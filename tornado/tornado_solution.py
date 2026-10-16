# %% [markdown]
# # Солнечные магнитные торнадо: полное решение (YOLO + временное сглаживание)
#
# Что улучшено относительно `baseline.ipynb`:
#
# 1. **Честная валидация**: отложенная часть train, разбитая по целым дням (как и тест),
#    по ней выбираются лучшая эпоха (по mAP50, а не по train loss) и все пороги.
# 2. **Модель крупнее и разрешение полное**: YOLO26s, `imgsz=768` (снимки 768×384 не сжимаются).
# 3. **Аугментации под домен**: без вертикального флипа (лимб всегда снизу), без цветовых
#    искажений (снимки серые), только яркость/масштаб/сдвиг/горизонтальный флип/мозаика.
# 4. **Временное сглаживание**: тестовые дни идут целиком, кадр каждые ~6–12 минут,
#    торнадо живут часами. Уверенность каждой рамки усредняется с совпадающими рамками
#    соседних кадров того же тайла, а пропуски в цепочках заполняются интерполяцией.
# 5. **Подбор порогов под micro F1** на валидации: порог уверенности, окно сглаживания,
#    «хотя бы одна рамка на снимок» (в train нет пустых снимков), отсечение рамок вне зоны лимба.
#
# Запуск: распакуйте `data_tornado_v2.zip` рядом со скриптом (или подключите Kaggle-датасет
# `d1ffic00lt/russian-olympiad-in-ai-2026-task-c`), затем `python tornado_solution.py`
# или выполните ячейки ноутбука по порядку. Итог: `submission.csv`.

# %%
# %pip install -q ultralytics==8.4.102 scipy
# Colab: скачать и распаковать данные (≈4 ГБ, 2–5 минут)
# !test -f data/train.csv || (curl -sSL -o d.zip https://www.kaggle.com/api/v1/datasets/download/d1ffic00lt/russian-olympiad-in-ai-2026-task-c && mkdir -p data && unzip -q d.zip -d data && rm d.zip)

# %%
import gc
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import yaml
from scipy.optimize import linear_sum_assignment

# %% [markdown]
# ## Настройки
#
# Значения по умолчанию рассчитаны на одну GPU с 16+ ГБ (T4/P100/V100/RTX 3090/4090/A100).
# При нехватке памяти уменьшите `BATCH`. Быстрый пробный прогон: `TORNADO_SMOKE=1`.

# %%
SMOKE = os.environ.get('TORNADO_SMOKE', '0') == '1'   # крошечный прогон для проверки кода

MODEL = 'yolo26s.pt'       # yolo26m.pt даст ещё немного качества при большем времени
EPOCHS = 30
IMGSZ = 768                # без потери разрешения по ширине
BATCH = 48                 # -1 = автоподбор под память GPU
PATIENCE = 12
TRAIN_STRIDE = 2           # брать каждый N-й кадр тайла (соседние кадры почти одинаковые); 2 ускоряет вдвое
VAL_DAY_FRACTION = 0.15    # доля дней train в отложенную валидацию
FULL_RETRAIN = False       # True: после подбора порогов дообучить модель на всех днях (≈ x2 время)
TTA = True                 # + предсказание на отражённых снимках, объединение через NMS (проверяется на валидации)
TRAIN_HOURS = 1.25         # жёсткий лимит времени обучения в часах (перекрывает EPOCHS); None — без лимита
PRED_CONF = 0.01           # сырые предсказания сохраняем с низким порогом, отсекаем потом
SEED = 0

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
WORKERS = min(8, os.cpu_count() or 1)

if SMOKE:
    MODEL, EPOCHS, IMGSZ, BATCH, PATIENCE, TRAIN_HOURS = 'yolo26n.pt', 1, 320, 8, 1, None
    FULL_RETRAIN = False

random.seed(SEED)
np.random.seed(SEED)
print('device:', DEVICE, '| smoke:', SMOKE)

# %% [markdown]
# ## Данные

# %%
def find_data_root():
    """Папка, в которой лежат train.csv, test.csv и tornado_yolo/."""
    candidates = []
    if os.environ.get('TORNADO_DATA'):
        candidates.append(Path(os.environ['TORNADO_DATA']))
    for root in (Path.cwd(), Path.cwd().parent, Path('/kaggle/input')):
        if root.exists():
            candidates.append(root)
            candidates.extend(p.parent for p in root.glob('**/tornado_yolo') if p.is_dir())
    for c in candidates:
        if (c / 'train.csv').exists() and (c / 'tornado_yolo' / 'train' / 'images').exists():
            return c.resolve()
    raise FileNotFoundError('Распакуйте data_tornado_v2.zip рядом с ноутбуком или задайте TORNADO_DATA.')


DATA_ROOT = find_data_root()
YOLO_ROOT = DATA_ROOT / 'tornado_yolo'
WORK = Path(os.environ.get('TORNADO_WORK', 'work')).resolve()
WORK.mkdir(parents=True, exist_ok=True)
print('data:', DATA_ROOT, '| work:', WORK)


def parse_ids(df):
    df = df.copy()
    stem = df['image_id'].str.rsplit('_', n=1)
    df['frame'] = stem.str[0]
    df['tile'] = stem.str[1]
    df['time'] = pd.to_datetime(df['frame'], format='%Y-%m-%dT%H%M%SZ')
    df['day'] = df['time'].dt.strftime('%Y-%m-%d')
    return df


train_df = parse_ids(pd.read_csv(DATA_ROOT / 'train.csv'))
test_df = parse_ids(pd.read_csv(DATA_ROOT / 'test.csv'))
train_df['gt'] = train_df['boxes'].map(json.loads)
print(len(train_df), 'train images,', train_df['day'].nunique(), 'days;',
      len(test_df), 'test images,', test_df['day'].nunique(), 'days')

# %%
# Разбиение train/val по целым дням, как устроен тест.
days = np.array(sorted(train_df['day'].unique()))
rng = np.random.default_rng(SEED)
val_days = set(rng.choice(days, size=max(1, int(round(len(days) * VAL_DAY_FRACTION))), replace=False))
train_df['is_val'] = train_df['day'].isin(val_days)
print('val days:', len(val_days), '| val images:', int(train_df['is_val'].sum()))


def thin_frames(df, stride):
    """Оставляет каждый stride-й кадр в последовательности каждого тайла."""
    if stride <= 1:
        return df
    df = df.sort_values(['tile', 'time'])
    return df[df.groupby(['day', 'tile']).cumcount() % stride == 0]


def build_split(name, df):
    """Папка в формате YOLO из симлинков на исходные снимки (данные не копируются)."""
    img_dir, lbl_dir = WORK / name / 'images', WORK / name / 'labels'
    shutil.rmtree(WORK / name, ignore_errors=True)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for iid in df['image_id']:
        (img_dir / f'{iid}.png').symlink_to(YOLO_ROOT / 'train' / 'images' / f'{iid}.png')
        shutil.copyfile(YOLO_ROOT / 'train' / 'labels' / f'{iid}.txt', lbl_dir / f'{iid}.txt')
    return img_dir


fit_df = thin_frames(train_df[~train_df['is_val']], TRAIN_STRIDE)
val_df = train_df[train_df['is_val']]
if SMOKE:
    fit_df = fit_df.sample(200, random_state=SEED)
    val_df = val_df[val_df['day'].isin(sorted(val_df['day'].unique())[:2])].head(150)

build_split('fit', fit_df)
build_split('val', val_df)


def write_yaml(name, train_split, val_split):
    path = WORK / f'{name}.yaml'
    path.write_text(yaml.safe_dump({'path': str(WORK), 'train': f'{train_split}/images',
                                    'val': f'{val_split}/images', 'names': {0: 'tornado'}}))
    return path


SPLIT_YAML = write_yaml('split', 'fit', 'val')
print('fit images:', len(fit_df), '| val images:', len(val_df))

# %% [markdown]
# ## Обучение
#
# Лучшая эпоха выбирается по mAP50 на отложенных днях: метрика задачи считает совпадением
# IoU ≥ 0.5, а стандартная fitness Ultralytics на 90% состоит из mAP50-95.

# %%
from ultralytics import YOLO
from ultralytics.utils import metrics as yolo_metrics

yolo_metrics.Metric.fitness = lambda self: self.map50

TRAIN_ARGS = dict(
    imgsz=IMGSZ, batch=BATCH, device=DEVICE, workers=WORKERS, seed=SEED,
    optimizer='AdamW', lr0=0.001, cos_lr=True, warmup_epochs=2, patience=PATIENCE,
    single_cls=True, amp=True, cache='ram', plots=False, verbose=False,
    # аугментации: снимки серые, лимб всегда снизу
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.35, degrees=0.0, shear=0.0, perspective=0.0,
    translate=0.1, scale=0.35, fliplr=0.5, flipud=0.0,
    mosaic=1.0, close_mosaic=8, mixup=0.0, copy_paste=0.0,
)
RUNS = WORK / 'runs'


def train(data_yaml, name, epochs, val=True):
    model = YOLO(MODEL)
    extra = {'time': TRAIN_HOURS} if TRAIN_HOURS else {}
    model.train(data=str(data_yaml), epochs=epochs, val=val, **extra, project=str(RUNS), name=name,
                exist_ok=True, **TRAIN_ARGS)
    weights = RUNS / name / 'weights'
    best = weights / 'best.pt'
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best if best.exists() else weights / 'last.pt'


SPLIT_WEIGHTS = train(SPLIT_YAML, 'split', EPOCHS)
print('weights:', SPLIT_WEIGHTS)

# %% [markdown]
# ## Сырые предсказания
#
# Сохраняем все рамки с уверенностью ≥ `PRED_CONF`, отбор делаем постобработкой.

# %%
def predict_raw(weights, image_paths, tta=False):
    """tta=True — предсказание на отражённых по горизонтали снимках (координаты возвращаются обратно)."""
    model = YOLO(str(weights))
    out = {}
    for start in range(0, len(image_paths), 64):
        chunk = list(image_paths[start:start + 64])
        imgs = [cv2.imread(str(p)) for p in chunk]
        if tta:
            imgs = [np.ascontiguousarray(im[:, ::-1]) for im in imgs]
        results = model.predict(imgs, imgsz=IMGSZ, conf=PRED_CONF, max_det=100,
                                device=DEVICE, half=DEVICE != 'cpu', verbose=False)
        for p, res in zip(chunk, results):
            boxes = res.boxes.xywhn.cpu().numpy().astype(np.float64)
            if tta:
                boxes[:, 0] = 1.0 - boxes[:, 0]
            out[Path(p).stem] = (boxes, res.boxes.conf.cpu().numpy().astype(np.float64))
    del model
    gc.collect()
    return out


val_paths = [YOLO_ROOT / 'train' / 'images' / f'{i}.png' for i in val_df['image_id']]
def merge_tta(a, b):
    return {k: (np.vstack([a[k][0], b[k][0]]), np.concatenate([a[k][1], b[k][1]])) for k in a}


val_raw = {False: predict_raw(SPLIT_WEIGHTS, val_paths)}
if TTA:
    val_raw[True] = merge_tta(val_raw[False], predict_raw(SPLIT_WEIGHTS, val_paths, tta=True))

# %% [markdown]
# ## Метрика и постобработка

# %%
def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def iou_matrix(a, b):
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def micro_f1(pred_by_id, gt_by_id):
    """micro F1 при IoU ≥ 0.5 с максимальным паросочетанием, как в условии."""
    tp = fp = fn = 0
    for iid, gt in gt_by_id.items():
        pred = pred_by_id.get(iid, [])
        if len(pred) and len(gt):
            match = iou_matrix(pred, gt) >= 0.5
            r, c = linear_sum_assignment(-match.astype(np.float64))
            m = int(match[r, c].sum())
        else:
            m = 0
        tp += m
        fp += len(pred) - m
        fn += len(gt) - m
    return 2 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def build_sequences(df):
    """Для каждого снимка — список соседних по времени снимков того же тайла (до ±30 мин)."""
    order = {}
    for _, g in df.sort_values('time').groupby(['day', 'tile']):
        ids, times = g['image_id'].tolist(), g['time'].to_numpy()
        for k, iid in enumerate(ids):
            order[iid] = (ids, times, k)
    return order


def neighbours(seq, iid, window, max_gap=np.timedelta64(30, 'm')):
    ids, times, k = seq[iid]
    res = []
    for j in range(max(0, k - window), min(len(ids), k + window + 1)):
        if j != k and abs(times[j] - times[k]) <= max_gap * max(1, abs(j - k)):
            res.append((ids[j], j - k))
    return res


def fill_gaps(raw, seq, conf_floor):
    """Добавляет рамку в кадр t, если в t-1 и t+1 есть согласованные рамки, а в t ничего похожего нет."""
    out = {k: (v[0].copy(), v[1].copy()) for k, v in raw.items()}
    for iid in raw:
        nb = dict((d, n) for n, d in neighbours(seq, iid, 1))
        if -1 not in nb or 1 not in nb:
            continue
        (bp, sp), (bn, sn) = raw[nb[-1]], raw[nb[1]]
        bp, sp = bp[sp >= conf_floor], sp[sp >= conf_floor]
        bn, sn = bn[sn >= conf_floor], sn[sn >= conf_floor]
        if not len(bp) or not len(bn):
            continue
        iou = iou_matrix(bp, bn)
        cur_b, cur_s = out[iid]
        add_b, add_s = [], []
        for i, j in zip(*np.nonzero(iou >= 0.5)):
            box = (bp[i] + bn[j]) / 2
            if len(cur_b) and iou_matrix(box[None], cur_b[cur_s >= conf_floor]).max(initial=0) >= 0.3:
                continue
            add_b.append(box)
            add_s.append(min(sp[i], sn[j]))
        if add_b:
            out[iid] = (np.vstack([cur_b, np.array(add_b)]), np.concatenate([cur_s, add_s]))
    return out


def temporal_scores(raw, seq, window, self_weight):
    """Новая уверенность = взвешенное среднее своей и лучших совпадающих рамок соседних кадров."""
    out = {}
    for iid, (boxes, scores) in raw.items():
        if not len(boxes) or window == 0:
            out[iid] = scores
            continue
        acc = scores * self_weight
        weight = self_weight
        for nid, _ in neighbours(seq, iid, window):
            nb, ns = raw.get(nid, (np.zeros((0, 4)), np.zeros(0)))
            if len(nb):
                iou = iou_matrix(boxes, nb)
                acc = acc + np.where(iou >= 0.3, ns[None, :], 0).max(axis=1)
            weight += 1
        out[iid] = acc / weight
    return out


def nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = iou_matrix(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=int)


_smooth_cache = {}


def smoothed(raw, seq, p):
    """Заполнение пропусков + временное сглаживание; кешируется, т.к. не зависит от порогов."""
    key = (id(raw), id(seq), p['window'], p['self_weight'], p['fill'], p['fill_floor'])
    if key not in _smooth_cache:
        r = fill_gaps(raw, seq, p['fill_floor']) if p['fill'] else raw
        _smooth_cache[key] = (r, temporal_scores(r, seq, p['window'], p['self_weight']))
    return _smooth_cache[key]


def postprocess(raw, seq, p):
    """raw: id -> (boxes xywhn, scores). Возвращает id -> список рамок [x, y, w, h]."""
    r, smooth = smoothed(raw, seq, p)
    result = {}
    for iid, (boxes, _) in r.items():
        s = smooth[iid]
        if len(boxes):
            valid = (boxes[:, 1] + boxes[:, 3] / 2 >= p['min_bottom']) & (s >= min(p['conf'], p['force_one']))
            boxes, s = boxes[valid], s[valid]
        if len(boxes):
            keep = nms(boxes, s, p['nms_iou'])
            boxes, s = boxes[keep], s[keep]
        sel = s >= p['conf']
        if not sel.any() and len(s) and s.max() >= p['force_one']:
            sel = s == s.max()
        boxes = boxes[sel]
        if len(boxes):
            boxes = np.clip(boxes, 1e-6, 1.0)          # координаты в [0,1], размеры > 0
        result[iid] = boxes.tolist()
    return result


# %% [markdown]
# ## Подбор постобработки на отложенных днях

# %%
val_seq = build_sequences(val_df)
val_gt = dict(zip(val_df['image_id'], val_df['gt']))

base = dict(conf=0.5, window=0, self_weight=1.0, fill=False, fill_floor=0.3,
            min_bottom=0.0, nms_iou=0.7, force_one=1.1)
print('как бейзлайн (conf 0.5, без TTA):', micro_f1(postprocess(val_raw[False], val_seq, base), val_gt))


def best_conf(raw, p, conf_grid=np.round(np.arange(0.05, 0.851, 0.025), 3)):
    scores = [(micro_f1(postprocess(raw, val_seq, dict(p, conf=float(c))), val_gt)[0], float(c)) for c in conf_grid]
    f1, conf = max(scores)
    return f1, dict(p, conf=conf)


def search(raw):
    """Поэтапный поиск: сглаживание -> NMS/зона лимба -> «хотя бы одна рамка»."""
    best = best_conf(raw, base)
    for window, self_weight in ((1, 1.0), (2, 1.0), (2, 2.0), (3, 1.0), (3, 2.0), (5, 1.0), (5, 2.0), (8, 2.0)):
        for fill in (False, True):
            best = max(best, best_conf(raw, dict(best[1], window=window, self_weight=self_weight, fill=fill)),
                       key=lambda x: x[0])
    for nms_iou in (0.4, 0.5, 0.6, 0.7):
        for min_bottom in (0.0, 0.8, 0.9):
            best = max(best, best_conf(raw, dict(best[1], nms_iou=nms_iou, min_bottom=min_bottom)),
                       key=lambda x: x[0])
    for frac in (0.3, 0.5, 0.7, 0.85):
        p = dict(best[1], force_one=best[1]['conf'] * frac)
        best = max(best, (micro_f1(postprocess(raw, val_seq, p), val_gt)[0], p), key=lambda x: x[0])
    return best


results = {tta: search(raw) for tta, raw in val_raw.items()}
for tta, (f1, p) in results.items():
    print(f'TTA={tta}: F1={f1:.4f}', p)
USE_TTA = max(results, key=lambda t: results[t][0])
BEST_F1, BEST_PARAMS = results[USE_TTA]
print('выбрано: TTA =', USE_TTA, '| val F1 =', round(BEST_F1, 4))
(WORK / 'postprocess.json').write_text(json.dumps({'tta': USE_TTA, 'val_f1': BEST_F1, **BEST_PARAMS}, indent=2))

# %% [markdown]
# ## (Опционально) дообучение на всех днях
#
# Модель учится на всех 134 днях с тем же числом эпох, что дала лучший результат на валидации;
# пороги берутся найденные выше.

# %%
FINAL_WEIGHTS = SPLIT_WEIGHTS
if FULL_RETRAIN:
    build_split('all', thin_frames(train_df, TRAIN_STRIDE))
    best_epoch = int(torch.load(SPLIT_WEIGHTS, map_location='cpu', weights_only=False).get('epoch', EPOCHS - 1)) + 1
    full_epochs = max(best_epoch, 10)
    TRAIN_ARGS['patience'] = full_epochs  # без валидации early stopping не нужен
    FINAL_WEIGHTS = train(write_yaml('all', 'all', 'val'), 'all', full_epochs, val=False)
    print('full weights:', FINAL_WEIGHTS, '| epochs:', full_epochs)

# %% [markdown]
# ## Предсказание теста и посылка

# %%
test_paths = [YOLO_ROOT / 'test' / 'images' / f'{i}.png' for i in test_df['image_id']]
test_raw = predict_raw(FINAL_WEIGHTS, test_paths)
if USE_TTA:
    test_raw = merge_tta(test_raw, predict_raw(FINAL_WEIGHTS, test_paths, tta=True))
test_seq = build_sequences(test_df)
test_boxes = postprocess(test_raw, test_seq, BEST_PARAMS)

submission = test_df[['image_id']].copy()
submission['boxes'] = [json.dumps([[float(v) for v in b] for b in test_boxes.get(i, [])][:1000],
                                  separators=(',', ':')) for i in test_df['image_id']]
assert submission['boxes'].str.len().max() <= 250000
assert submission['image_id'].tolist() == pd.read_csv(DATA_ROOT / 'test.csv')['image_id'].tolist()
OUTPUT_CSV = Path(os.environ.get('TORNADO_OUT', 'submission.csv'))
submission.to_csv(OUTPUT_CSV, index=False)
n_boxes = submission['boxes'].map(lambda s: len(json.loads(s)))
print(f'saved {OUTPUT_CSV}: {len(submission)} rows, {n_boxes.sum()} boxes, '
      f'{(n_boxes == 0).mean():.1%} empty | {datetime.now():%H:%M}')

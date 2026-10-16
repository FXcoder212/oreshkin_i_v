# %% [markdown]
# # Супер-разрешение ×4 спутниковых снимков (Colab)
#
# **Решение.** Остаточная свёрточная сеть RCAN-типа (блоки с канальным вниманием)
# поверх бикубика: сеть предсказывает только поправку к бикубическому увеличению,
# в начале обучения она в точности бикубик.
#
# 1. Фаза L1 — попиксельная L1-потеря, сеть выучивает деградацию 1024 → 256.
# 2. Фаза «метрика» — дифференцируемая копия платформенной метрики
#    (SSIM, MAE, лапласиан, Собель, HF): оптимизируем ровно то, что считают.
# 3. EMA-веса; самоансамбль ×8 и сила поправки выбираются
#    на валидации по точной numpy-метрике из `baseline.ipynb`.
# 4. Посылка SRP1 в parquet с самопроверкой.
#
# **Как запускать в Colab.**
# 1. Среда выполнения → Сменить среду → GPU (T4 хватает, A100/L4 быстрее).
# 2. Данные скачиваются сами с Kaggle (публичный датасет, ключ не нужен, ≈ 4.4 ГБ).
#    Если архив уже лежит на вашем Google Диске — укажите путь в `ZIP_ON_DRIVE`.
# 3. Среда выполнения → Выполнить всё. Чекпойнты пишутся в `WORK_ON_DRIVE`
#    каждые 1000 шагов; после отключения Colab просто запустите всё заново —
#    обучение продолжится с последнего чекпойнта.
#
# Время на T4: ≈ 3 часа на обучение, ≈ 10 минут на посылку.

# %%
import os
import subprocess
from pathlib import Path

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists() and not os.environ.get("SR_DATA")

ZIP_ON_DRIVE = "/content/drive/MyDrive/data.zip"   # необязательно: архив с данными на вашем Диске
WORK_ON_DRIVE = "/content/drive/MyDrive/sr_work"   # сюда пишутся чекпойнты и посылка
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/d1ffic00lt/russian-olympiad-in-ai-2026-task-a"
GDRIVE_FILE_ID = "1LCBvwkXYDp_F-d_EZKR4ZDijsWzYJ2jT"  # запасной вариант (бывает упирается в квоту Google)

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    os.environ.setdefault("SR_OUT", WORK_ON_DRIVE)
    local_zip = Path("/content/data.zip")
    if not any(Path("/content/data_unpacked").rglob("lr")):
        if Path(ZIP_ON_DRIVE).exists():
            print("копирую архив с Диска…")
            subprocess.run(["cp", ZIP_ON_DRIVE, str(local_zip)], check=True)
        elif subprocess.run(["curl", "-sSLf", "-o", str(local_zip), KAGGLE_URL]).returncode != 0:
            print("Kaggle не отдал архив, пробую Google Диск…")
            subprocess.run(["pip", "install", "-q", "gdown"], check=True)
            subprocess.run(["gdown", GDRIVE_FILE_ID, "-O", str(local_zip)], check=True)
        subprocess.run(["unzip", "-q", "-o", str(local_zip), "-d", "/content/data_unpacked"], check=True)
        local_zip.unlink()
    # папка, внутри которой лежат train/ val/ test/
    root = next(p.parent.parent for p in Path("/content/data_unpacked").rglob("lr") if p.parent.name == "train")
    os.environ.setdefault("SR_DATA", str(root))
    print("данные:", os.environ["SR_DATA"])

# %%
import math
import struct
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter, sobel

DATA = Path(os.environ.get("SR_DATA", "data"))
OUT = Path(os.environ.get("SR_OUT", "."))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True

PRESETS = {
    # полноразмерная модель для Colab GPU (~7.3 M параметров)
    "full": dict(feats=96, groups=6, blocks=6, patch=48, batch=16, steps_l1=70_000,
                 steps_metric=30_000, patch_metric=64, lr=2e-4, eval_every=5_000, eval_scenes=30,
                 final_eval_scenes=120),
    # маленькая модель, чтобы дождаться на CPU
    "cpu": dict(feats=32, groups=2, blocks=3, patch=32, batch=8, steps_l1=6_000, steps_metric=2_000,
                patch_metric=48, lr=4e-4, eval_every=2_000, eval_scenes=10, final_eval_scenes=40),
    # проверка, что всё запускается
    "smoke": dict(feats=8, groups=1, blocks=1, patch=16, batch=2, steps_l1=30, steps_metric=20,
                  patch_metric=24, lr=1e-3, eval_every=20, eval_scenes=2, final_eval_scenes=3),
}
CFG = dict(PRESETS[os.environ.get("SR_PRESET", "full" if DEVICE.type == "cuda" else "cpu")])
for _key in ("steps_l1", "steps_metric", "eval_every", "eval_scenes", "final_eval_scenes"):
    if os.environ.get(f"SR_{_key.upper()}"):
        CFG[_key] = int(os.environ[f"SR_{_key.upper()}"])
CFG.update(
    use_val_for_train=os.environ.get("SR_TRAIN_ON_VAL", "0") == "1",  # дообучить и на val (кроме отложенных)
    ema=0.999,
    save_every=1_000,
    seed=0,
)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(DEVICE, CFG)


def read(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def bicubic(low):
    return np.asarray(Image.fromarray(low).resize((1024, 1024), Image.Resampling.BICUBIC))


def ids_of(folder):
    return sorted(p.stem for p in sorted(Path(folder).glob("*.tif")))


train_ids = ids_of(DATA / "train" / "lr")
val_ids = ids_of(DATA / "val" / "lr")
test_ids = ids_of(DATA / "test" / "images")
print(f"train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

# %% [markdown]
# ## Метрика платформы (точная копия из baseline.ipynb)

# %%
V2_WEIGHTS = {"ssim": 0.33, "mae": 0.27, "gradient": 0.16, "laplacian": 0.22, "high_frequency": 0.02}
V2_COLLAPSE = {"std_zero_below": 0.05, "std_full_above": 0.25,
               "hf_zero_below": 0.02, "hf_full_above": 0.06, "epsilon": 0.000001}
LAPLACIAN = np.array([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=np.float32)


def ssim(x_chw, y_chw, data_range=255.0):
    x, y = x_chw.astype(np.float32), y_chw.astype(np.float32)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    blur = lambda v: gaussian_filter(v, sigma=(0, 1.5, 1.5))
    mx, my = blur(x), blur(y)
    vx, vy = blur(x * x) - mx * mx, blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    value = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(np.clip(value.mean(), 0, 1))


def ramp(ratio, low, full):
    if ratio < low:
        return 0.0
    return float(np.clip((ratio - low) / max(full - low, 1e-12), 0, 1))


def raw_score(prediction, target):
    """Вход HWC uint8, выход (сырой балл, составляющие)."""
    pred = np.moveaxis(np.asarray(prediction), -1, 0).astype(np.float32)
    gt = np.moveaxis(np.asarray(target), -1, 0).astype(np.float32)
    data_range = 255.0
    epsilon = float(V2_COLLAPSE["epsilon"]) * data_range
    nmae = float(np.clip(np.abs(pred - gt).mean() / data_range, 0, 1))
    grad = lambda v: np.hypot(sobel(v, axis=-2), sobel(v, axis=-1))
    gradient_nmae = float(np.clip(np.abs(grad(pred) - grad(gt)).mean() / (2 * np.sqrt(2) * data_range), 0, 1))
    lap = lambda v: convolve(v, LAPLACIAN, mode="reflect")
    laplacian_nmae = float(np.clip(np.abs(lap(pred) - lap(gt)).mean() / (4 * data_range), 0, 1))
    hf = lambda v: float(np.abs(v - gaussian_filter(v, sigma=(0, 1.2, 1.2))).mean())
    hf_ratio = hf(pred) / max(hf(gt), epsilon)
    hf_score = float(np.exp(-abs(math.log(max(hf_ratio, 1e-12)))))
    structure = ssim(pred, gt, data_range)
    value = (V2_WEIGHTS["ssim"] * structure
             + V2_WEIGHTS["mae"] * (1 - nmae)
             + V2_WEIGHTS["gradient"] * (1 - gradient_nmae)
             + V2_WEIGHTS["laplacian"] * (1 - laplacian_nmae)
             + V2_WEIGHTS["high_frequency"] * hf_score)
    std_ratio = float(np.std(pred)) / max(float(np.std(gt)), epsilon)
    penalty = min(ramp(std_ratio, V2_COLLAPSE["std_zero_below"], V2_COLLAPSE["std_full_above"]),
                  ramp(hf_ratio, V2_COLLAPSE["hf_zero_below"], V2_COLLAPSE["hf_full_above"]))
    parts = {"ssim": structure, "mae": 1 - nmae, "laplacian": 1 - laplacian_nmae,
             "gradient": 1 - gradient_nmae, "hf": hf_score}
    return float(np.clip(value * penalty, 0, 1)), parts


def scene_score(s, b):
    """Нормировка относительно бикубика: бикубик = 50, идеал = 100."""
    if s >= b:
        return 50 + 50 * (s - b) / max(1 - b, 1e-12)
    return 50 * s / max(b, 1e-12)

# %% [markdown]
# ## Дифференцируемая версия метрики (потеря для фазы 2)
#
# Тонкости, повторённые за scipy:
# * `sobel` на массиве (C, H, W) сглаживает [1, 2, 1] ещё и по оси каналов
#   (с отражением), поэтому каналы смешиваются матрицей [[3,1,0],[1,2,1],[0,1,3]];
# * гауссово ядро обрезается на радиусе int(4σ + 0.5).
#
# Края патча отражаются по-торчевому, а не по-scipy, — на обучение это почти
# не влияет, а валидация всегда считается точной numpy-версией.

# %%
def gaussian_kernel(sigma):
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g15", gaussian_kernel(1.5))
        self.register_buffer("g12", gaussian_kernel(1.2))
        self.register_buffer("mix", torch.tensor([[3., 1., 0.], [1., 2., 1.], [0., 1., 3.]]))
        d = torch.tensor([-1., 0., 1.])
        s = torch.tensor([1., 2., 1.])
        self.register_buffer("sx", torch.outer(s, d)[None, None])  # производная по ширине
        self.register_buffer("sy", torch.outer(d, s)[None, None])  # производная по высоте
        self.register_buffer("lap", torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None])

    @staticmethod
    def _conv(x, k):
        n, c, h, w = x.shape
        pad = k.shape[-1] // 2
        y = F.pad(x.reshape(n * c, 1, h, w), (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(y, k).reshape(n, c, h, w)

    @staticmethod
    def _blur(x, k1d):
        n, c, h, w = x.shape
        r = k1d.numel() // 2
        y = x.reshape(n * c, 1, h, w)
        y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k1d.view(1, 1, 1, -1))
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k1d.view(1, 1, -1, 1))
        return y.reshape(n, c, h, w)

    def _grad(self, x):
        x = torch.einsum("ij,njhw->nihw", self.mix, x)
        gx, gy = self._conv(x, self.sx), self._conv(x, self.sy)
        return torch.sqrt(gx * gx + gy * gy + 1e-10)

    def _ssim(self, x, y):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mx, my = self._blur(x, self.g15), self._blur(y, self.g15)
        vx = self._blur(x * x, self.g15) - mx * mx
        vy = self._blur(y * y, self.g15) - my * my
        cov = self._blur(x * y, self.g15) - mx * my
        value = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        return value.mean(dim=(1, 2, 3))

    def scores(self, pred, gt):
        """pred, gt: (N, 3, H, W) в [0, 1]. Сырой балл каждой картинки без защитного множителя."""
        dims = (1, 2, 3)
        mae = (pred - gt).abs().mean(dims)
        grad = (self._grad(pred) - self._grad(gt)).abs().mean(dims) / (2 * math.sqrt(2))
        lap = (self._conv(pred, self.lap) - self._conv(gt, self.lap)).abs().mean(dims) / 4
        hf_p = (pred - self._blur(pred, self.g12)).abs().mean(dims)
        hf_g = (gt - self._blur(gt, self.g12)).abs().mean(dims)
        hf = torch.exp(-torch.log((hf_p + 1e-6) / (hf_g + 1e-6)).abs())
        return (0.33 * self._ssim(pred, gt) + 0.27 * (1 - mae) + 0.16 * (1 - grad)
                + 0.22 * (1 - lap) + 0.02 * hf)

    def forward(self, pred, gt):
        return 1 - self.scores(pred, gt).mean()

# %% [markdown]
# ## Модель: RCAN-подобная сеть ×4
#
# `groups` групп по `blocks` остаточных блоков с канальным вниманием,
# длинные пропуски, апсемплинг двумя PixelShuffle ×2. Последний слой
# инициализирован нулями, поэтому стартовый выход — ровно бикубик.

# %%
class ChannelAttention(nn.Module):
    def __init__(self, feats, reduction=16):
        super().__init__()
        hidden = max(1, feats // reduction)
        self.body = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(feats, hidden, 1),
                                  nn.ReLU(inplace=True), nn.Conv2d(hidden, feats, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.body(x)


class RCAB(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(feats, feats, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(feats, feats, 3, padding=1), ChannelAttention(feats))

    def forward(self, x):
        return x + self.body(x)


class Group(nn.Module):
    def __init__(self, feats, blocks):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(feats) for _ in range(blocks)], nn.Conv2d(feats, feats, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class SRNet(nn.Module):
    """Выход = бикубик(вход) + поправка сети."""

    def __init__(self, feats=96, groups=6, blocks=6):
        super().__init__()
        self.head = nn.Conv2d(3, feats, 3, padding=1)
        self.body = nn.Sequential(*[Group(feats, blocks) for _ in range(groups)],
                                  nn.Conv2d(feats, feats, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(feats, feats * 4, 3, padding=1), nn.PixelShuffle(2),
                                nn.Conv2d(feats, feats * 4, 3, padding=1), nn.PixelShuffle(2))
        self.tail = nn.Conv2d(feats, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr, residual_scale=1.0):
        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        x = self.head(lr - 0.5)
        x = x + self.body(x)
        return base + residual_scale * self.tail(self.up(x))


model = SRNet(CFG["feats"], CFG["groups"], CFG["blocks"]).to(DEVICE).to(memory_format=torch.channels_last)
ema_model = SRNet(CFG["feats"], CFG["groups"], CFG["blocks"]).to(DEVICE).to(memory_format=torch.channels_last)
ema_model.load_state_dict(model.state_dict())
ema_model.requires_grad_(False)
print(f"параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# %% [markdown]
# ## Проверка: потеря-метрика совпадает с платформенной
#
# На целой картинке без защитного множителя торчевая версия должна давать тот же
# балл, что и numpy (с точностью до краёв в несколько пикселей).

# %%
_low = read(DATA / "val" / "lr" / f"{val_ids[0]}.tif")
_high = read(DATA / "val" / "hr" / f"{val_ids[0]}.tif")
_pred = bicubic(_low)
_to_t = lambda a: torch.from_numpy(np.array(a)).permute(2, 0, 1)[None].float().div(255)
with torch.no_grad():
    _torch_raw = MetricLoss().scores(_to_t(_pred), _to_t(_high)).item()
print(f"бикубик на {val_ids[0]}: numpy {raw_score(_pred, _high)[0]:.5f}, torch {_torch_raw:.5f}")

# %% [markdown]
# ## Данные для обучения
#
# Пары держим в памяти в uint8 (train ≈ 6.5 ГБ — влезает в обычный Colab),
# патчи режем случайно; аугментации — отражения и поворот на 90°.
# Первые `eval_scenes` сцен валидации всегда отложены для оценки.

# %%
def load_pairs(split, ids):
    lows, highs = [], []
    started = time.time()
    for i, scene in enumerate(ids):
        lows.append(read(DATA / split / "lr" / f"{scene}.tif"))
        highs.append(read(DATA / split / "hr" / f"{scene}.tif"))
        if (i + 1) % 250 == 0:
            print(f"  {split}: {i + 1}/{len(ids)}, {time.time() - started:.0f} с", flush=True)
    return lows, highs


eval_ids = val_ids[:CFG["eval_scenes"]]
train_lows, train_highs = [], []


def ensure_train_data():
    if train_lows:
        return
    limit = int(os.environ.get("SR_MAX_TRAIN", "0")) or None  # для быстрых проверок
    lows, highs = load_pairs("train", train_ids[:limit])
    train_lows.extend(lows)
    train_highs.extend(highs)
    if CFG["use_val_for_train"]:
        held = set(val_ids[:CFG["final_eval_scenes"]])
        lows, highs = load_pairs("val", [s for s in val_ids if s not in held])
        train_lows.extend(lows)
        train_highs.extend(highs)
    print(f"пар для обучения: {len(train_lows)}")


def sample_batch(batch, patch):
    lr = np.empty((batch, patch, patch, 3), np.uint8)
    hr = np.empty((batch, patch * 4, patch * 4, 3), np.uint8)
    for i in range(batch):
        k = np.random.randint(len(train_lows))
        low, high = train_lows[k], train_highs[k]
        y = np.random.randint(low.shape[0] - patch + 1)
        x = np.random.randint(low.shape[1] - patch + 1)
        lr[i] = low[y:y + patch, x:x + patch]
        hr[i] = high[4 * y:4 * (y + patch), 4 * x:4 * (x + patch)]
    lr = torch.from_numpy(lr).to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    hr = torch.from_numpy(hr).to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    if np.random.rand() < 0.5:
        lr, hr = lr.flip(-1), hr.flip(-1)
    if np.random.rand() < 0.5:
        lr, hr = lr.flip(-2), hr.flip(-2)
    if np.random.rand() < 0.5:
        lr, hr = lr.transpose(-1, -2), hr.transpose(-1, -2)
    return (lr.contiguous(memory_format=torch.channels_last),
            hr.contiguous(memory_format=torch.channels_last))

# %% [markdown]
# ## Инференс и валидация

# %%
@torch.no_grad()
def upscale(net, low, tta=True, residual_scale=1.0):
    """HWC uint8 256x256 -> HWC uint8 1024x1024."""
    net.eval()
    x = torch.from_numpy(np.array(low)).to(DEVICE).permute(2, 0, 1)[None].float() / 255
    variants = [(t, fh, fw) for t in (False, True) for fh in (False, True) for fw in (False, True)]
    if not tta:
        variants = variants[:1]
    acc = 0
    for t, fh, fw in variants:
        v = x
        if t:
            v = v.transpose(-1, -2)
        if fh:
            v = v.flip(-2)
        if fw:
            v = v.flip(-1)
        with torch.autocast(DEVICE.type, enabled=AMP):
            y = net(v.contiguous(memory_format=torch.channels_last), residual_scale).float()
        if fw:
            y = y.flip(-1)
        if fh:
            y = y.flip(-2)
        if t:
            y = y.transpose(-1, -2)
        acc = acc + y
    y = (acc / len(variants)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(np.round(y * 255), 0, 255).astype(np.uint8)


_val_cache = {}


def val_pair(scene):
    """(вход, эталон, сырой балл бикубика) — кэшируется, чтобы не читать заново."""
    if scene not in _val_cache:
        low = read(DATA / "val" / "lr" / f"{scene}.tif")
        high = read(DATA / "val" / "hr" / f"{scene}.tif")
        _val_cache[scene] = (low, high, raw_score(bicubic(low), high)[0])
    return _val_cache[scene]


def evaluate(net, ids, tta=False, residual_scale=1.0):
    scores = []
    for scene in ids:
        low, high, b = val_pair(scene)
        s, _ = raw_score(upscale(net, low, tta, residual_scale), high)
        scores.append(scene_score(s, b))
    return float(np.mean(scores))

# %% [markdown]
# ## Обучение с возобновлением
#
# Сквозной счётчик шагов: сначала `steps_l1` шагов L1, затем `steps_metric`
# шагов потери-метрики. Каждые `save_every` шагов полное состояние
# (веса, EMA, оптимизатор, расписание) пишется в `train_state.pt`.
# Если уже есть `model.pt` — обучение пропускается (удалите файл, чтобы переобучить).

# %%
PHASES = [
    dict(name="L1", steps=CFG["steps_l1"], patch=CFG["patch"], lr=CFG["lr"]),
    dict(name="metric", steps=CFG["steps_metric"], patch=CFG["patch_metric"], lr=CFG["lr"] / 4),
]


def make_optimizer(phase):
    opt = torch.optim.Adam(model.parameters(), lr=phase["lr"], betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=phase["lr"], total_steps=phase["steps"],
                                                pct_start=0.02, anneal_strategy="cos",
                                                div_factor=10, final_div_factor=100)
    return opt, sched


def save_state(path, step, phase_index, opt, sched, scaler):
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "phase": phase_index, "model": model.state_dict(),
                "ema": ema_model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "cfg": CFG}, tmp)
    os.replace(tmp, path)  # атомарно: обрыв посреди записи не портит чекпойнт


def train():
    ensure_train_data()
    state_path = OUT / "train_state.pt"
    metric_loss = MetricLoss().to(DEVICE)
    losses = [lambda p, t: (p - t).abs().mean(), metric_loss]
    step, state = 0, None
    if state_path.exists():
        state = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema"])
        step = state["step"]
        print(f"продолжаю с шага {step}")
    offsets = np.cumsum([0] + [p["steps"] for p in PHASES])
    for index, phase in enumerate(PHASES):
        if step >= offsets[index + 1] or phase["steps"] <= 0:
            continue
        opt, sched = make_optimizer(phase)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
        if state is not None and state["phase"] == index and step > offsets[index]:
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
        started, running, count = time.time(), 0.0, 0
        while step < offsets[index + 1]:
            model.train()
            lr_t, hr_t = sample_batch(CFG["batch"], phase["patch"])
            with torch.autocast(DEVICE.type, enabled=AMP):
                pred = model(lr_t)
            loss = losses[index](pred.float(), hr_t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for e, p in zip(ema_model.parameters(), model.parameters()):
                    e.lerp_(p, 1 - CFG["ema"])
            step += 1
            running, count = running + loss.item(), count + 1
            if step % CFG["save_every"] == 0 or step == offsets[index + 1]:
                save_state(state_path, step, index, opt, sched, scaler)
            if step % CFG["eval_every"] == 0 or step == offsets[index + 1]:
                score = evaluate(ema_model, eval_ids)
                speed = count / (time.time() - started)
                print(f"[{phase['name']}] шаг {step}/{offsets[-1]}  потеря {running / count:.5f}  "
                      f"Score(EMA, {len(eval_ids)} сцен) {score:.3f}  {speed:.1f} шаг/с", flush=True)
                started, running, count = time.time(), 0.0, 0
    torch.save(ema_model.state_dict(), OUT / "model.pt")


final_path = OUT / "model.pt"
if final_path.exists():
    print("найдена обученная модель", final_path)
else:
    train()
ema_model.load_state_dict(torch.load(final_path, map_location=DEVICE))
net = ema_model

# %% [markdown]
# ## Выбор самоансамбля и силы поправки
#
# Самоансамбль ×8 усредняет предсказания и этим сглаживает мелкую текстуру, а метрика
# (лапласиан, градиенты, HF) как раз сравнивает текстуру с эталоном. На маленькой модели
# он даже снижал балл, поэтому не включаем его вслепую: перебираем «с ансамблем / без» и
# коэффициент поправки сети и берём лучшее по точной метрике на отложенных сценах.

# %%
rows = [{"tta": tta, "scale": s, "Score": evaluate(net, eval_ids, tta=tta, residual_scale=s)}
        for tta in (False, True) for s in (0.9, 1.0, 1.1, 1.2)]
choice = pd.DataFrame(rows)
print(choice.to_string(index=False))
best = choice.loc[choice["Score"].idxmax()]
best_tta, best_scale = bool(best["tta"]), float(best["scale"])

final_ids = val_ids[:CFG["final_eval_scenes"]]
print(f"самоансамбль {best_tta}, коэффициент {best_scale}; Score на {len(final_ids)} сценах val: "
      f"{evaluate(net, final_ids, tta=best_tta, residual_scale=best_scale):.3f} (бикубик = 50)")

# %% [markdown]
# ## Посылка

# %%
HEADER = struct.Struct(">4sBHHBdI")


def encode(image):
    """RGB uint8 (1024, 1024, 3) -> бинарный payload SRP1."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    assert image.shape == (1024, 1024, 3), image.shape
    delta = image.copy()
    delta[:, 1:] = image[:, 1:] - image[:, :-1]
    head = HEADER.pack(b"SRP1", 2, 1024, 1024, 3, 0.0, image.size)
    return head + zlib.compress(delta.tobytes(), 9)


def decode(payload):
    magic, kind, height, width, channels, _, raw_size = HEADER.unpack_from(payload)
    assert magic == b"SRP1" and kind == 2
    assert (height, width, channels, raw_size) == (1024, 1024, 3, 1024 * 1024 * 3)
    delta = np.frombuffer(zlib.decompress(payload[HEADER.size:]), dtype=np.uint8)
    delta = delta.reshape(height, width, channels)
    return (np.cumsum(delta.astype(np.uint32), axis=1) & 255).astype(np.uint8)


payloads, first = [], None
for scene in test_ids:
    prediction = upscale(net, read(DATA / "test" / "images" / f"{scene}.tif"), best_tta, best_scale)
    if first is None:
        first = prediction
    payloads.append(encode(prediction))

schema = pa.schema([pa.field("id", pa.string()), pa.field("image", pa.binary())])
table = pa.Table.from_arrays([pa.array(test_ids), pa.array(payloads, type=pa.binary())], schema=schema)
submission = OUT / "submission.parquet"
pq.write_table(table, submission, compression="zstd", compression_level=9,
               use_dictionary=False, write_statistics=True, row_group_size=1)
size = submission.stat().st_size

check = pq.read_table(submission)
assert check.column_names == ["id", "image"], check.column_names
assert check.num_rows == len(test_ids) == 150 or os.environ.get("SR_PRESET") == "smoke", check.num_rows
assert pa.types.is_binary(check.schema.field("image").type)
assert set(check.column("id").to_pylist()) == set(test_ids)
assert len(set(check.column("id").to_pylist())) == len(test_ids)
assert size < 300_000_000, size
for payload in check.column("image").to_pylist():
    restored = decode(payload)
    assert restored.shape == (1024, 1024, 3) and restored.dtype == np.uint8
    assert restored.std() > 1.0, "вырожденное предсказание"
assert np.array_equal(decode(check.column("image")[0].as_py()), first)
print(f"{submission}: {check.num_rows} строк, {size} байт — посылка готова")

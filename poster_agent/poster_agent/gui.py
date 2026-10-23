"""Оконный интерфейс агента афиш (PySide6)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import anthropic
from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from .agent import PosterAgent, PosterResult, load_plan
from .airfail import AirFailClient
from .config import Settings, config_dir
from .frame import Frame, load_frame

APP_NAME = "Афиши по плану мероприятий"
FRAME_EXT = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_PLAN_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".docx", ".xlsx"}
ATTACH_PLAN_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
EFFORTS = [("Быстро", "low"), ("Обычно", "medium"), ("Тщательно", "high"), ("Максимально", "xhigh")]

STYLE = """
QMainWindow, QDialog { background: #f5f6fa; }
QGroupBox { font-weight: 600; border: 1px solid #dfe3eb; border-radius: 10px; margin-top: 14px;
            background: white; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #2b3a55; }
QPushButton { padding: 6px 14px; border-radius: 6px; border: 1px solid #c9d0dc; background: #ffffff; }
QPushButton:hover { background: #eef2fb; }
QPushButton#primary { background: #3d6df2; color: white; border: none; font-weight: 600;
                      padding: 10px; font-size: 15px; }
QPushButton#primary:hover { background: #2f5ad6; }
QPushButton#primary:disabled { background: #a9b9e8; }
QPushButton#danger { color: #c0392b; }
QLabel#hint { color: #6b7385; }
QLabel#drop { border: 2px dashed #b8c2d6; border-radius: 10px; color: #6b7385; background: #fafbfe; }
QListWidget { background: white; border: 1px solid #dfe3eb; border-radius: 10px; }
QPlainTextEdit { background: white; border: 1px solid #dfe3eb; border-radius: 6px; }
"""


# ---------------------------------------------------------------- виджеты

class DropLabel(QLabel):
    """Область, куда можно перетащить файл или кликнуть для выбора."""

    fileDropped = Signal(str)
    clicked = Signal()

    def __init__(self, text: str):
        super().__init__(text)
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setWordWrap(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, e):
        self.clicked.emit()


class DropGroup(QGroupBox):
    """Группа, принимающая перетаскиваемый файл."""

    fileDropped = Signal(str)

    def __init__(self, title: str):
        super().__init__(title)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ключи и подключение")
        self.setMinimumWidth(520)
        self.s = settings
        lay = QVBoxLayout(self)

        keys = QGroupBox("API-ключи")
        f = QFormLayout(keys)
        self.anthropic = self._secret(settings.anthropic_api_key)
        self.airfail = self._secret(settings.airfail_api_key)
        f.addRow("Claude (Anthropic):", self.anthropic)
        f.addRow("air.fail:", self.airfail)
        hint = QLabel('Ключ Claude: <a href="https://console.anthropic.com/settings/keys">console.anthropic.com</a>. '
                      'Ключ air.fail: личный кабинет <a href="https://app.air.fail">app.air.fail</a> → API ключи.')
        hint.setObjectName("hint")
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        f.addRow(hint)
        lay.addWidget(keys)

        adv = QGroupBox("air.fail — дополнительно")
        g = QFormLayout(adv)
        self.endpoint = QLineEdit(settings.airfail_endpoint)
        self.auth = QComboBox()
        self.auth.addItems(["Authorization: <ключ>", "Authorization: Bearer <ключ>"])
        self.auth.setCurrentIndex(1 if settings.airfail_auth_scheme == "Bearer" else 0)
        self.info = QLineEdit(settings.airfail_info_json)
        self.info.setPlaceholderText('например {"quality": "high"}')
        g.addRow("Адрес генерации:", self.endpoint)
        g.addRow("Заголовок:", self.auth)
        g.addRow("Параметры info (JSON):", self.info)
        lay.addWidget(adv)

        note = QLabel(f"Настройки хранятся локально: {config_dir()}")
        note.setObjectName("hint")
        note.setWordWrap(True)
        lay.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _secret(value: str) -> QLineEdit:
        e = QLineEdit(value)
        e.setEchoMode(QLineEdit.Password)
        act = e.addAction(QIcon(), QLineEdit.TrailingPosition)
        act.setText("👁")
        act.triggered.connect(lambda: e.setEchoMode(
            QLineEdit.Normal if e.echoMode() == QLineEdit.Password else QLineEdit.Password))
        return e

    def _save(self):
        info = self.info.text().strip()
        if info:
            try:
                json.loads(info)
            except ValueError:
                QMessageBox.warning(self, APP_NAME, "Параметры info должны быть корректным JSON.")
                return
        self.s.anthropic_api_key = self.anthropic.text().strip()
        self.s.airfail_api_key = self.airfail.text().strip()
        self.s.airfail_endpoint = self.endpoint.text().strip() or self.s.airfail_endpoint
        self.s.airfail_auth_scheme = "Bearer" if self.auth.currentIndex() == 1 else ""
        self.s.airfail_info_json = info
        self.s.save()
        self.accept()


# ---------------------------------------------------------------- фоновая работа

class AgentWorker(QThread):
    log = Signal(str)
    poster = Signal(object)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, agent: PosterAgent, plan: list[dict], api_key: str):
        super().__init__()
        self.agent, self.plan, self.api_key = agent, plan, api_key
        agent.log = self.log.emit
        agent.on_poster = self.poster.emit

    def run(self):
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            self.done.emit(self.agent.run(self.plan, client=client))
        except anthropic.AuthenticationError:
            self.failed.emit("Ключ Claude не подошёл. Проверьте его в «Ключи и подключение».")
        except anthropic.APIConnectionError as e:
            self.failed.emit(f"Нет связи с Claude API: {e}")
        except Exception as e:  # noqa: BLE001 — показываем пользователю любую ошибку
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))


# ---------------------------------------------------------------- главное окно

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        self.frame: Frame | None = None
        self.frame_path: str | None = None
        self.plan_attachment: str | None = None
        self.worker: AgentWorker | None = None
        self.current_out: Path | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self._build_toolbar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 860])
        self.setCentralWidget(splitter)
        self._update_start()

    # ----- построение интерфейса

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for text, slot in (("🔑  Ключи и подключение", self.open_settings),
                           ("📂  Папка с афишами", self.open_out_dir),
                           ("❓  Как пользоваться", self.show_help)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)
        self.addToolBar(tb)

    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # 1. Рамка
        g1 = QGroupBox("1. Рамка")
        l1 = QVBoxLayout(g1)
        self.frame_view = DropLabel("Перетащите сюда рамку (PNG)\nили нажмите, чтобы выбрать файл\n\n"
                                    "Окно под картинку — прозрачное\nили залито зелёным #00FF00")
        self.frame_view.setMinimumHeight(300)
        self.frame_view.clicked.connect(self.choose_frame)
        self.frame_view.fileDropped.connect(self.set_frame)
        self.frame_info = QLabel("")
        self.frame_info.setObjectName("hint")
        self.frame_info.setWordWrap(True)
        box_row = QHBoxLayout()
        self.box_edit = QLineEdit()
        self.box_edit.setPlaceholderText("окно вручную: x, y, ширина, высота")
        btn_box = QPushButton("Применить")
        btn_box.clicked.connect(lambda: self.frame_path and self.set_frame(self.frame_path))
        box_row.addWidget(self.box_edit)
        box_row.addWidget(btn_box)
        l1.addWidget(self.frame_view)
        l1.addWidget(self.frame_info)
        l1.addLayout(box_row)
        lay.addWidget(g1)

        # 2. План
        g2 = DropGroup("2. План мероприятий")
        g2.fileDropped.connect(self.set_plan)
        l2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        b_open = QPushButton("Открыть файл…")
        b_open.clicked.connect(self.choose_plan)
        b_clear = QPushButton("Очистить")
        b_clear.clicked.connect(self.clear_plan)
        row.addWidget(b_open)
        row.addWidget(b_clear)
        row.addStretch()
        self.plan_edit = QPlainTextEdit()
        self.plan_edit.setPlaceholderText("Вставьте план сюда или перетащите файл "
                                          "(.txt .md .csv .docx .xlsx .pdf)…\n\n"
                                          "Например:\n7 ноября, 12:00 — День открытых дверей, главный зал…")
        self.plan_edit.setAcceptDrops(False)
        self.plan_edit.textChanged.connect(self._update_start)
        self.plan_attach_label = QLabel("")
        self.plan_attach_label.setObjectName("hint")
        self.plan_attach_label.hide()
        l2.addLayout(row)
        l2.addWidget(self.plan_edit)
        l2.addWidget(self.plan_attach_label)
        lay.addWidget(g2, 1)

        # 3. Параметры
        g3 = QGroupBox("3. Параметры")
        f = QFormLayout(g3)
        self.cb_text = QCheckBox("Писать название и дату на картинке")
        self.cb_text.setChecked(self.settings.with_text)
        self.cb_dry = QCheckBox("Только промпты (без генерации картинок)")
        self.effort = QComboBox()
        for label, value in EFFORTS:
            self.effort.addItem(label, value)
        self.effort.setCurrentIndex(max(0, self.effort.findData(self.settings.effort)))
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(self.settings.out_dir)
        b_out = QPushButton("…")
        b_out.setFixedWidth(36)
        b_out.clicked.connect(self.choose_out)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(b_out)
        f.addRow(self.cb_text)
        f.addRow(self.cb_dry)
        f.addRow("Тщательность:", self.effort)
        f.addRow("Сохранять в:", out_row)
        lay.addWidget(g3)

        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Кнопки — всегда видны под прокручиваемой областью.
        panel = QWidget()
        panel.setMinimumWidth(380)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(8, 0, 8, 8)
        pl.addWidget(scroll, 1)
        self.btn_start = QPushButton("▶  Создать афиши")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("■  Остановить")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.hide()
        pl.addWidget(self.btn_start)
        pl.addWidget(self.btn_stop)
        return panel

    def _build_right(self) -> QWidget:
        split = QSplitter(Qt.Vertical)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(6, 6, 6, 0)
        head = QLabel("Готовые афиши")
        head.setStyleSheet("font-size: 16px; font-weight: 600; color: #2b3a55;")
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(220, 300))
        self.gallery.setGridSize(QSize(240, 350))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setWordWrap(True)
        self.gallery.itemDoubleClicked.connect(self._open_item)
        self.gallery.currentItemChanged.connect(self._show_item)
        tl.addWidget(head)
        tl.addWidget(self.gallery)

        self.tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        prompt_tab = QWidget()
        pl = QVBoxLayout(prompt_tab)
        self.prompt_view = QPlainTextEdit()
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setPlaceholderText("Выберите афишу, чтобы увидеть её промпт")
        b_copy = QPushButton("Копировать промпт")
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.prompt_view.toPlainText()))
        pl.addWidget(self.prompt_view)
        pl.addWidget(b_copy, alignment=Qt.AlignRight)
        self.tabs.addTab(self.log_view, "Журнал агента")
        self.tabs.addTab(prompt_tab, "Промпт")

        split.addWidget(top)
        split.addWidget(self.tabs)
        split.setSizes([560, 220])
        return split

    # ----- рамка

    def choose_frame(self):
        path, _ = QFileDialog.getOpenFileName(self, "Рамка", "", "Изображения (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.set_frame(path)

    def set_frame(self, path: str):
        if Path(path).suffix.lower() not in FRAME_EXT:
            QMessageBox.warning(self, APP_NAME, "Рамка должна быть картинкой: PNG, JPG или WEBP.")
            return
        box = None
        if self.box_edit.text().strip():
            try:
                box = tuple(int(v) for v in self.box_edit.text().replace(" ", "").split(","))
                assert len(box) == 4
            except (ValueError, AssertionError):
                QMessageBox.warning(self, APP_NAME, "Окно задаётся четырьмя числами: x, y, ширина, высота.")
                return
        try:
            self.frame = load_frame(path, box=box)
        except Exception as e:  # noqa: BLE001
            self.frame = None
            self.frame_view.setPixmap(QPixmap())
            self.frame_view.setText(str(e))
            self.frame_info.setText("")
            self._update_start()
            return
        self.frame_path = path
        self._draw_frame_preview(path)
        x, y, w, h = self.frame.box
        self.frame_info.setText(f"{Path(path).name}: окно {w}×{h} px, отступ ({x}, {y}). Красная рамка — "
                                "куда встанет картинка.")
        self._update_start()

    def _draw_frame_preview(self, path: str):
        pm = QPixmap(path)
        # Шахматка под прозрачностью + контур окна.
        canvas = QPixmap(pm.size())
        canvas.fill(Qt.white)
        p = QPainter(canvas)
        cell = max(8, pm.width() // 60)
        for yy in range(0, pm.height(), cell):
            for xx in range(0, pm.width(), cell):
                if (xx // cell + yy // cell) % 2:
                    p.fillRect(xx, yy, cell, cell, QColor("#d9dde5"))
        p.drawPixmap(0, 0, pm)
        pen = QPen(QColor("#e53935"))
        pen.setWidth(max(3, pm.width() // 200))
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRect(*self.frame.box)
        p.end()
        self.frame_view.setPixmap(canvas.scaled(max(200, self.frame_view.width() - 20), 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ----- план

    def choose_plan(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "План мероприятий", "",
            "Планы (*.txt *.md *.csv *.tsv *.json *.docx *.xlsx *.pdf *.png *.jpg *.jpeg);;Все файлы (*)")
        if path:
            self.set_plan(path)

    def set_plan(self, path: str):
        ext = Path(path).suffix.lower()
        try:
            if ext in ATTACH_PLAN_EXT:
                self.plan_attachment = path
                self.plan_attach_label.setText(f"📎 {Path(path).name} — будет передан агенту целиком. "
                                               "В поле можно дописать пояснения.")
                self.plan_attach_label.show()
            else:
                blocks = load_plan(path)
                self.plan_edit.setPlainText(blocks[0]["text"])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Не удалось прочитать план: {e}")
        self._update_start()

    def clear_plan(self):
        self.plan_attachment = None
        self.plan_attach_label.hide()
        self.plan_edit.clear()
        self._update_start()

    def _plan_blocks(self) -> list[dict]:
        blocks = load_plan(self.plan_attachment) if self.plan_attachment else []
        text = self.plan_edit.toPlainText().strip()
        if text:
            blocks.append({"type": "text", "text": text})
        return blocks

    # ----- параметры

    def choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для афиш", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def open_settings(self) -> bool:
        return SettingsDialog(self.settings, self).exec() == QDialog.Accepted

    def open_out_dir(self):
        d = self.current_out or Path(self.out_edit.text())
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def show_help(self):
        QMessageBox.information(self, APP_NAME, (
            "1. Нажмите «Ключи и подключение» и введите ключи Claude и air.fail (один раз).\n\n"
            "2. Перетащите рамку. Область под картинку должна быть прозрачной (PNG) или залитой "
            "чистым зелёным #00FF00 — либо задайте окно вручную.\n\n"
            "3. Вставьте план мероприятий текстом или откройте файл (Word, Excel, PDF, текст).\n\n"
            "4. Нажмите «Создать афиши». Агент придумает промпт для каждого мероприятия, сгенерирует "
            "картинку, вставит её в рамку, проверит результат и при необходимости переделает.\n\n"
            "Двойной клик по афише открывает файл. Для каждой афиши в папке сохраняются промпт, "
            "исходная картинка и афиша в рамке."))

    # ----- запуск

    def _update_start(self):
        ready = self.frame is not None and (bool(self.plan_edit.toPlainText().strip()) or bool(self.plan_attachment))
        self.btn_start.setEnabled(ready and self.worker is None)
        if self.worker is None:
            self.statusBar().showMessage("Всё готово — нажмите «Создать афиши»" if ready else
                                         "Выберите рамку и план мероприятий")

    def start(self):
        s = self.settings
        dry = self.cb_dry.isChecked()
        if not s.anthropic_api_key or (not dry and not s.airfail_api_key):
            QMessageBox.information(self, APP_NAME, "Сначала укажите API-ключи.")
            if not self.open_settings():
                return
            if not s.anthropic_api_key or (not dry and not s.airfail_api_key):
                return

        s.with_text = self.cb_text.isChecked()
        s.effort = self.effort.currentData()
        s.out_dir = self.out_edit.text().strip() or s.out_dir
        s.save()

        from datetime import datetime

        self.current_out = Path(s.out_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_client = None if dry else AirFailClient(
            api_key=s.airfail_api_key, endpoint=s.airfail_endpoint, auth_scheme=s.airfail_auth_scheme,
            extra_info=json.loads(s.airfail_info_json) if s.airfail_info_json else {})
        try:
            plan = self._plan_blocks()
            agent = PosterAgent(frame=self.frame, out_dir=self.current_out, image_client=image_client,
                                with_text=s.with_text, effort=s.effort)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(e))
            return

        self.gallery.clear()
        self.log_view.clear()
        self.tabs.setCurrentIndex(0)
        self.worker = AgentWorker(agent, plan, s.anthropic_api_key)
        self.worker.log.connect(self._log)
        self.worker.poster.connect(self._add_poster)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._worker_gone)
        self.worker.start()

        self.btn_start.hide()
        self.btn_stop.show()
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage(f"Агент работает… Результаты: {self.current_out}")

    def stop(self):
        if self.worker:
            self.worker.agent.stop_event.set()
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("Останавливаю после текущего шага…")

    def _log(self, text: str):
        self.log_view.appendPlainText(text)

    def _add_poster(self, res: PosterResult):
        label = f"{res.title}\nпопытка {res.attempt}"
        if res.poster_path:
            item = QListWidgetItem(QIcon(QPixmap(res.poster_path)), label)
            item.setData(Qt.UserRole, res.poster_path)
        elif res.error:
            item = QListWidgetItem(f"⚠ {label}\n{res.error[:120]}")
        else:
            item = QListWidgetItem(f"📝 {label}\n(только промпт)")
        item.setData(Qt.UserRole + 1, res.prompt)
        item.setToolTip(res.prompt)
        self.gallery.addItem(item)
        self.gallery.scrollToItem(item)

    def _show_item(self, item: QListWidgetItem | None, _prev=None):
        if item:
            self.prompt_view.setPlainText(item.data(Qt.UserRole + 1) or "")
            self.tabs.setCurrentIndex(1)

    def _open_item(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _finished(self, text: str):
        n = sum(1 for r in self.worker.agent.results if r.poster_path)
        self.statusBar().showMessage(f"Готово: {n} афиш(и) в {self.current_out}")

    def _failed(self, message: str):
        self.statusBar().showMessage("Ошибка — подробности в журнале")
        QMessageBox.critical(self, APP_NAME, message)

    def _worker_gone(self):
        self.worker = None
        self.btn_stop.hide()
        self.btn_start.show()
        self._update_start()

    def closeEvent(self, e):
        if self.worker:
            if QMessageBox.question(self, APP_NAME, "Агент ещё работает. Остановить и выйти?") != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.agent.stop_event.set()
            self.worker.wait(5000)
        e.accept()


def _resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / name


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    icon = _resource("assets/icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

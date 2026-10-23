# PyInstaller: pyinstaller packaging/PosterAgent.spec  (запускать из папки poster_agent)
import sys
from pathlib import Path

root = Path(SPECPATH).parent

a = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root)],
    datas=[(str(root / "assets" / "icon.png"), "assets"),
           (str(root / "examples"), "examples")],
    hiddenimports=["docx", "openpyxl"],
    excludes=["streamlit", "tkinter", "matplotlib", "numpy", "pandas", "IPython",
              "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="PosterAgent",
    console=False,
    icon=str(root / "assets" / ("icon.ico" if sys.platform == "win32" else "icon.png")),
    upx=False,
)

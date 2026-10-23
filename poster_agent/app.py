"""Веб-интерфейс: streamlit run app.py

Пользователь загружает рамку и план мероприятий, агент делает афиши.
"""

import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from poster_agent import AirFailClient, PosterAgent, load_frame, load_plan

st.set_page_config(page_title="Афиши по плану мероприятий", layout="wide")
st.title("Афиши по плану мероприятий")

with st.sidebar:
    st.header("Ключи")
    anthropic_key = st.text_input("ANTHROPIC_API_KEY", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                  type="password")
    airfail_key = st.text_input("AIRFAIL_API_KEY", value=os.environ.get("AIRFAIL_API_KEY", ""), type="password")
    st.header("Настройки")
    with_text = st.checkbox("Надписи на афише (название, дата)", value=True)
    dry_run = st.checkbox("Только промпты (без генерации)", value=False)
    effort = st.select_slider("Тщательность агента", ["low", "medium", "high", "xhigh"], value="medium")
    manual_box = st.text_input("Окно вручную x,y,w,h (необязательно)")

col1, col2 = st.columns(2)
frame_file = col1.file_uploader("Рамка (PNG с прозрачным окном или окном #00FF00)",
                                type=["png", "jpg", "jpeg", "webp"])
plan_file = col2.file_uploader("План мероприятий", type=["txt", "md", "csv", "json", "pdf", "docx", "xlsx"])
plan_text = col2.text_area("…или вставьте план текстом", height=200)

if frame_file:
    col1.image(frame_file, caption="Рамка", width=300)

if st.button("Сделать афиши", type="primary", disabled=not (frame_file and (plan_file or plan_text.strip()))):
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if airfail_key:
        os.environ["AIRFAIL_API_KEY"] = airfail_key

    work = Path(tempfile.mkdtemp(prefix="posters_"))
    frame_path = work / f"frame{Path(frame_file.name).suffix}"
    frame_path.write_bytes(frame_file.getvalue())
    if plan_file:
        plan_path = work / f"plan{Path(plan_file.name).suffix}"
        plan_path.write_bytes(plan_file.getvalue())
    else:
        plan_path = work / "plan.txt"
        plan_path.write_text(plan_text, encoding="utf-8")

    try:
        box = tuple(int(v) for v in manual_box.split(",")) if manual_box.strip() else None
        frame = load_frame(str(frame_path), box=box)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.info(f"Окно в рамке: {frame.box[2]}×{frame.box[3]} px, начиная с ({frame.box[0]}, {frame.box[1]})")

    log_box = st.status("Агент работает…", expanded=True)
    gallery = st.container()

    def on_poster(res):
        if res.poster_path:
            gallery.image(res.poster_path, caption=f"{res.title} — попытка {res.attempt}", width=320)
        elif res.error:
            gallery.warning(f"{res.title}: {res.error}")
        else:
            gallery.code(res.prompt, language=None)

    agent = PosterAgent(
        frame=frame,
        out_dir=work / "out",
        image_client=None if dry_run else AirFailClient.from_env(),
        with_text=with_text,
        effort=effort,
        log=log_box.write,
        on_poster=on_poster,
    )
    try:
        agent.run(load_plan(plan_path))
        log_box.update(label="Готово", state="complete")
    except Exception as e:  # показываем ошибку, но сохраняем уже сделанное
        log_box.update(label="Ошибка", state="error")
        st.exception(e)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted((work / "out").iterdir()):
            z.write(f, f.name)
    st.download_button("Скачать всё (zip)", buf.getvalue(), file_name="posters.zip")

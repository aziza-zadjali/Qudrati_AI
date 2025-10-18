# -*- coding: utf-8 -*-
# pages/7_spacial_questions.py
# Spatial IQ Generator (Folding + Mimic Sample). ASCII-only.

import io, time, json
from pathlib import Path
import streamlit as st
from PIL import Image

from spatial_iq import (
    make_rng, extract_style_from_image,
    generate_folding_challenge, generate_matrix_reasoning,
)

# -------- helper to show images compatibly across Streamlit versions --------

def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)

# -------- UI --------

st.set_page_config(page_title="Spatial IQ Generator", layout="wide")
st.title("Spatial IQ Question Generator")

with st.sidebar:
    st.header("Mode")
    mode = st.selectbox("Choose", ["Folding Challenge", "Mimic Sample (Matrix)"], index=0)

    st.header("Difficulty and Seed")
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    seed_str = st.text_input("Seed (optional)", value="")
    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except Exception:
            seed = abs(hash(seed_str)) % (2**31)

    st.header("Visual Options")
    allow_diag = st.checkbox("Allow diagonal folds (folding mode)", value=True)
    bg_hex = st.text_input("Background hex", value="#FFFFFF")
    paper_hex = st.text_input("Paper fill hex", value="#FAFAFA")
    outline_hex = st.text_input("Outline hex", value="#141414")
    text_hex = st.text_input("Text hex", value="#141414")
    stroke_w = st.number_input("Stroke width", min_value=2, max_value=12, value=5, step=1)

    def hex_to_rgb(h):
        try:
            h = h.strip().lstrip("#")
            if len(h) == 3: h = "".join([c*2 for c in h])
            if len(h) != 6: return (255,255,255)
            return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
        except Exception:
            return (255,255,255)

    style = {
        "bg": hex_to_rgb(bg_hex),
        "paper_fill": hex_to_rgb(paper_hex),
        "outline": hex_to_rgb(outline_hex),
        "text_color": hex_to_rgb(text_hex),
        "stroke_width": int(stroke_w),
    }

    sample_bytes = None
    if mode.startswith("Mimic"):
        st.header("Sample for style (optional)")
        f = st.file_uploader("Upload sample image", type=["png","jpg","jpeg"], accept_multiple_files=False)
        if f:
            sample_bytes = f.getvalue()

    st.markdown("---")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)
    st.subheader("Batch")
    batch_n = st.number_input("How many to generate", min_value=2, max_value=100, value=10, step=1)
    batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

rng = make_rng(seed)

# -------- create one item --------

def make_one():
    if mode.startswith("Folding"):
        pack = generate_folding_challenge(rng, difficulty=difficulty, allow_diagonal=allow_diag, style=style)
        q_id = f"folding-{int(time.time()*1000)}"
        qtype = "folding"
        problem_img = pack["problem_img"]
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
    else:
        style2 = style.copy()
        if sample_bytes:
            try:
                im = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
                ext = extract_style_from_image(im)
                # blend: keep brand bg/outline if provided; use sample fills for variety
                style2["fills"] = ext.get("fills", [(30,30,30)])
            except Exception:
                pass
        pack = generate_matrix_reasoning(rng, difficulty=difficulty, style=style2)
        q_id = f"mimic-{int(time.time()*1000)}"
        qtype = "mimic"
        from spatial_iq import compose_grid  # local import to avoid circular import at top
        problem_img = compose_grid(pack["grid_imgs"], pack["grid_size"])
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]

    labels = [chr(ord('A') + i) for i in range(len(choices_imgs))]
    labeled = []
    for i, img in enumerate(choices_imgs):
        overlay = img.copy()
        d = ImageDraw.Draw(overlay)
        try:
            font = _load_font(32)  # reuse from spatial_iq if imported; else fallback below
        except Exception:
            from spatial_iq import _load_font as _lf
            font = _lf(32)
        # label box
        d.rectangle([10, 10, 74, 64], fill=(245,245,245), outline=(30,30,30), width=2)
        # center label
        try:
            left, top, right, bottom = d.textbbox((0,0), labels[i], font=font)
            tw, th = right - left, bottom - top
        except Exception:
            try:
                tw = int(d.textlength(labels[i], font=font))
            except Exception:
                tw = 18
            th = 24
        tx = 10 + (64 - tw) / 2
        ty = 10 + (54 - th) / 2
        d.text((tx, ty), labels[i], fill=(10,10,10), font=font)
        labeled.append(overlay)

    return {
        "id": q_id, "type": qtype, "difficulty": difficulty, "seed": seed or 0,
        "prompt": prompt_text, "rule_description": rule_desc,
        "labels": labels, "correct_index": correct_index,
        "problem_img": problem_img, "choices_imgs": labeled, "meta": meta
    }

# First render or on click
if ("qpack" not in st.session_state) or gen_btn:
    from PIL import ImageDraw  # local import to keep top minimal
    st.session_state.qpack = make_one()

qp = st.session_state.get("qpack")
if qp:
    colQ, colA = st.columns([2.1, 1.4])
    with colQ:
        st.subheader("Question")
        show_image(qp["problem_img"])
        st.write(qp["prompt"])
    with colA:
        st.subheader("Choices")
        chosen = st.radio("Select your answer:", qp["labels"], index=0, horizontal=True, label_visibility="collapsed")
        cols2 = st.columns(2)
        for i, img in enumerate(qp["choices_imgs"]):
            with cols2[i % 2]:
                show_image(img, caption=f"Option {qp['labels'][i]}")
        if st.button("Check answer"):
            if chosen == qp["labels"][qp["correct_index"]]:
                st.success(f"Correct. Answer: {qp['labels'][qp['correct_index']]}")
            else:
                st.error(f"Not quite. Correct answer: {qp['labels'][qp['correct_index']]}")
            st.markdown("Why: " + qp["rule_description"])

    st.markdown("---")
    st.subheader("Export")
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, img in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); img.save(cb, format="PNG")
            zf.writestr(f"choice_{qp['labels'][i]}.png", cb.getvalue())
        meta = {
            "id": qp["id"], "type": qp["type"], "difficulty": qp["difficulty"],
            "seed": qp["seed"], "prompt": qp["prompt"], "rule_description": qp["rule_description"],
            "correct_label": qp["labels"][qp["correct_index"]],
            "choices": [{"label": l, "is_correct": (idx == qp["correct_index"])} for idx, l in enumerate(qp["labels"])],
            "meta": qp["meta"]
        }
        zf.writestr("question.json", json.dumps(meta, indent=2))
    st.download_button("Download ZIP", data=buf.getvalue(), file_name=f"{qp['id']}.zip", mime="application/zip")

    # Batch
    if batch_btn:
        import zipfile
        batch_buf = io.BytesIO()
        with zipfile.ZipFile(batch_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            master = []
            for k in range(int(batch_n)):
                rng_local = make_rng((seed or 0) + k + 1)
                # re-seed per item via global RNG used inside functions
                # quick patch: rebuild via new qpack but keep current style/options
                from spatial_iq import make_rng as _mr
                pack_rng = _mr((seed or 0) + k + 1)
                # temporarily swap
                local = make_one()
                pb = io.BytesIO(); local["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{local['id']}/problem.png", pb.getvalue())
                for i, img in enumerate(local["choices_imgs"]):
                    cb = io.BytesIO(); img.save(cb, format="PNG")
                    zf.writestr(f"{local['id']}/choice_{local['labels'][i]}.png", cb.getvalue())
                meta = {
                    "id": local["id"], "type": local["type"], "difficulty": local["difficulty"],
                    "seed": local["seed"], "prompt": local["prompt"], "rule_description": local["rule_description"],
                    "correct_label": local["labels"][local["correct_index"]],
                    "choices": [{"label": l, "is_correct": (idx == local["correct_index"])} for idx, l in enumerate(local["labels"])],
                    "meta": local["meta"]
                }
                zf.writestr(f"{local['id']}/question.json", json.dumps(meta, indent=2))
                master.append(meta)
            zf.writestr("index.json", json.dumps(master, indent=2))
        st.download_button("Download Batch ZIP", data=batch_buf.getvalue(),
                           file_name=f"batch_{qp['type']}_{int(time.time())}.zip", mime="application/zip")
``

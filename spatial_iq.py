# -*- coding: utf-8 -*-
# spatial_iq.py
# Core generators for Folding Challenge and Matrix Reasoning.
# ASCII-only, Pillow 10+ safe.

import math, time, random, io
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont

# ----------- utilities -----------

def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _load_font(font_size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            return ImageFont.load_default()

def text_image(text: str, size=(380, 380), font_size=42,
               color=(30, 30, 30), bg=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    font = _load_font(font_size)
    try:
        # Pillow 10+: textbbox
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        x = (size[0] - w) / 2 - left
        y = (size[1] - h) / 2 - top
    except Exception:
        # Fallback using textlength + approximate height
        try:
            w = int(d.textlength(text, font=font))
        except Exception:
            w = int(len(text) * font_size * 0.6)
        h = int(font_size * 1.2)
        x = (size[0] - w) / 2
        y = (size[1] - h) / 2
    d.text((x, y), text, fill=color, font=font)
    return img

def _rotate_points(points, angle_deg, origin):
    ox, oy = origin
    ang = math.radians(angle_deg)
    out = []
    for x, y in points:
        qx = ox + math.cos(ang) * (x - ox) - math.sin(ang) * (y - oy)
        qy = oy + math.sin(ang) * (x - ox) + math.cos(ang) * (y - oy)
        out.append((qx, qy))
    return out

def _draw_polygon(draw, pts, fill, outline, width):
    draw.polygon(pts, fill=fill)
    pts_closed = list(pts) + [pts[0]]
    try:
        draw.line(pts_closed, fill=outline, width=width, joint="curve")
    except TypeError:
        draw.line(pts_closed, fill=outline, width=width)

def draw_shape(draw, shape: str, center: Tuple[int, int], size: int,
               rotation_deg: float = 0, fill=(20, 20, 20), outline=(20, 20, 20), width: int = 4):
    cx, cy = center
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)
    elif shape == "square":
        pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    elif shape == "pentagon":
        pts = []
        for k in range(5):
            ang = 90 + 72 * k
            pts.append((cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))))
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    else:
        draw.line([cx - r, cy, cx + r, cy], fill=outline, width=width)
        draw.line([cx, cy - r, cx, cy + r], fill=outline, width=width)

def compose_grid(images: List[Image.Image], grid_size: Tuple[int, int], pad: int = 16, bg=(255, 255, 255)) -> Image.Image:
    rows, cols = grid_size
    assert len(images) == rows * cols
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError("Tile size mismatch")
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (w + pad)
            y = pad + r * (h + pad)
            canvas.paste(images[r * cols + c], (x, y))
    return canvas

# ----------- style extraction for mimic -----------

def extract_style_from_image(img: Image.Image, n_colors: int = 6) -> Dict:
    small = img.convert("RGB")
    if max(small.size) > 512:
        scale = 512 / max(small.size)
        small = small.resize((int(small.width * scale), int(small.height * scale)), Image.BILINEAR)
    pal = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
    palette = pal.getpalette()[:n_colors * 3]
    counts = pal.getcolors()
    colors_freq = []
    if counts:
        for count, idx in counts:
            rgb = tuple(palette[idx * 3: idx * 3 + 3])
            colors_freq.append((count, rgb))
        colors_freq.sort(key=lambda x: x[0], reverse=True)
    bg = colors_freq[0][1] if colors_freq else (255, 255, 255)

    def lum(c): return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    unique = [rgb for _, rgb in colors_freq] or [(30, 30, 30), (240, 240, 240)]
    outline = min(unique, key=lum)
    fills = [c for c in unique if c != bg] or [(30, 30, 30), (0, 88, 155), (200, 0, 0)]
    stroke_width = 5 if img.width < 800 else (6 if img.width < 1400 else 8)
    text_color = (20, 20, 20) if lum(bg) > 186 else (240, 240, 240)
    paper_fill = tuple(min(255, int(bg[i] * 1.02)) for i in range(3))
    return {"bg": bg, "outline": outline, "fills": fills, "stroke_width": stroke_width,
            "text_color": text_color, "paper_fill": paper_fill}

# ----------- matrix generator -----------

DEFAULT_COLORS = [
    (30, 30, 30),
    (0, 88, 155),
    (200, 0, 0),
    (0, 140, 70),
    (220, 120, 0),
    (120, 0, 160),
]
DEFAULT_SHAPES = ["circle", "square", "triangle", "pentagon"]

def generate_matrix_reasoning(rng: random.Random, img_size=(380, 380), cell_shape_size=160,
                              difficulty="Medium", style: Optional[Dict] = None) -> Dict:
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline = (style.get("outline") if style else None) or (10, 10, 10)
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    stroke = (style.get("stroke_width") if style else None) or 4
    txt_col = (style.get("text_color") if style else None) or (30, 30, 30)

    if difficulty == "Easy":
        rules_to_use = 1; step_choices = [0, 45, 90]
    elif difficulty == "Hard":
        rules_to_use = 3; step_choices = [30, 45, 60, 90]
    else:
        rules_to_use = 2; step_choices = [30, 45, 60, 90]

    use_rotation = use_count = use_color = use_size = False
    for s in rng.sample(["rotation", "count", "color", "size"], rules_to_use):
        if s == "rotation": use_rotation = True
        elif s == "count":  use_count = True
        elif s == "color":  use_color = True
        elif s == "size":   use_size = True

    rotation_step = rng.choice(step_choices) if use_rotation else 0
    base_count = rng.randint(1, 2) if use_count else 1
    base_size = cell_shape_size
    size_step = rng.choice([-20, 20]) if use_size else 0
    base_color_idx = rng.randrange(len(fills)) if use_color else 0
    color_by_row = (rng.random() < 0.5) if use_color else True
    shape = rng.choice(DEFAULT_SHAPES)

    grid_imgs = []
    rule_desc_parts = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                grid_imgs.append(text_image("?", size=img_size, font_size=int(img_size[0] * 0.32),
                                            color=txt_col, bg=bg))
                continue
            img = Image.new("RGB", img_size, bg)
            d = ImageDraw.Draw(img)
            rot = (c * rotation_step) % 360 if use_rotation else 0
            count = base_count + r if use_count else 1
            size_px = max(50, base_size + r * size_step) if use_size else base_size
            if use_color:
                shift = r if color_by_row else c
                color_idx = (base_color_idx + shift) % len(fills)
            else:
                color_idx = base_color_idx
            cols_cnt = int(math.ceil(math.sqrt(count)))
            rows_cnt = int(math.ceil(count / cols_cnt))
            grid_w, grid_h = img_size
            margin = 30
            cell_w = (grid_w - 2 * margin) // cols_cnt
            cell_h = (grid_h - 2 * margin) // rows_cnt
            mini = min(size_px, int(0.8 * min(cell_w, cell_h)))
            k = 0
            for rr in range(rows_cnt):
                for cc in range(cols_cnt):
                    if k >= count: break
                    cx = margin + cc * cell_w + cell_w // 2
                    cy = margin + rr * cell_h + cell_h // 2
                    draw_shape(d, shape, (cx, cy), mini, rotation_deg=rot,
                               fill=fills[color_idx], outline=outline, width=stroke)
                    k += 1
            grid_imgs.append(img)

    correct_rot = (2 * rotation_step) % 360 if use_rotation else 0
    correct_count = base_count + 2 if use_count else 1
    correct_size = max(50, base_size + 2 * size_step) if use_size else base_size
    if use_color:
        shift = 2 if color_by_row else 2
        color_idx_correct = (base_color_idx + shift) % len(fills)
    else:
        color_idx_correct = base_color_idx

    def render_multi(count, rot, sz, color_idx):
        img = Image.new("RGB", img_size, bg)
        d = ImageDraw.Draw(img)
        cols_cnt = int(math.ceil(math.sqrt(count)))
        rows_cnt = int(math.ceil(count / cols_cnt))
        grid_w, grid_h = img_size
        margin = 30
        cell_w = (grid_w - 2 * margin) // cols_cnt
        cell_h = (grid_h - 2 * margin) // rows_cnt
        mini2 = min(sz, int(0.8 * min(cell_w, cell_h)))
        k2 = 0
        for rr in range(rows_cnt):
            for cc in range(cols_cnt):
                if k2 >= count: break
                cx = margin + cc * cell_w + cell_w // 2
                cy = margin + rr * cell_h + cell_h // 2
                draw_shape(d, shape, (cx, cy), mini2, rotation_deg=rot,
                           fill=fills[color_idx], outline=outline, width=stroke)
                k2 += 1
        return img

    correct_img = render_multi(correct_count, correct_rot, correct_size, color_idx_correct)

    def make_distractor(var: str):
        cnt, rot, sz, col = correct_count, correct_rot, correct_size, color_idx_correct
        if var == "rotation":
            step = rotation_step if rotation_step != 0 else rng.choice([30, 45, 60, 90])
            rot = (rot + rng.choice([-step, step])) % 360
        elif var == "count":
            cnt = max(1, cnt + rng.choice([-1, 1]))
        elif var == "size":
            sz = max(50, sz + rng.choice([-20, 20]))
        elif var == "color":
            col = (col + rng.choice([1, -1])) % len(fills)
        return render_multi(cnt, rot, sz, col)

    distractor_kinds = []
    if use_rotation: distractor_kinds.append("rotation")
    if use_count:    distractor_kinds.append("count")
    if use_size:     distractor_kinds.append("size")
    if use_color:    distractor_kinds.append("color")
    while len(distractor_kinds) < 3:
        distractor_kinds.append(rng.choice(["rotation", "count", "size", "color"]))

    choices_imgs = [correct_img] + [make_distractor(k) for k in distractor_kinds]
    rng.shuffle(choices_imgs)

    if use_rotation: rule_desc_parts.append(f"Rotation increases by {rotation_step} deg.")
    if use_count:    rule_desc_parts.append("Count increases by 1 down rows.")
    if use_color:    rule_desc_parts.append("Color alternates consistently.")
    if use_size:     rule_desc_parts.append(f"Size changes by {size_step:+d} px per row.")
    rule_desc = " ".join(rule_desc_parts) if rule_desc_parts else "Follow the visual pattern."
    prompt_text = "Which option completes the 3x3 matrix?"

    return {
        "grid_imgs": grid_imgs,
        "grid_size": (3, 3),
        "choices_imgs": choices_imgs,
        "correct_index": choices_imgs.index(correct_img),
        "rule_desc": rule_desc,
        "prompt": prompt_text,
        "meta": {"type": "matrix"}
    }

# ----------- folding challenge (with diagonal folds) -----------

def reflect_point(p, axis):
    x, y = p
    if axis == "V":  # x=0
        return (-x, y)
    if axis == "H":  # y=0
        return (x, -y)
    if axis == "D1":  # y=x
        return (y, x)
    if axis == "D2":  # y=-x
        return (-y, -x)
    return (x, y)

def unfold_points(base_point: Tuple[float, float], folds_axes: List[str]) -> List[Tuple[float, float]]:
    pts = [base_point]
    for axis in reversed(folds_axes):
        new_pts = [reflect_point(p, axis) for p in pts]
        pts = pts + new_pts
    # dedupe with rounding
    seen, out = set(), []
    for x, y in pts:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key); out.append((x, y))
    return out

def _arrow(d: ImageDraw.ImageDraw, start, end, color, width):
    d.line([start, end], fill=color, width=width)
    vx, vy = end[0] - start[0], end[1] - start[1]
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    head_len = max(14, width * 3)
    head_w = max(10, width * 2)
    tip = (end[0], end[1])
    base = (end[0] - ux * head_len, end[1] - uy * head_len)
    p1 = (base[0] + px * head_w, base[1] + py * head_w)
    p2 = (base[0] - px * head_w, base[1] - py * head_w)
    d.polygon([tip, p1, p2], fill=color)

def draw_fold_icon(direction: str, size=(180, 180), bg=(255, 255, 255),
                   paper_fill=(250, 250, 250), outline=(20, 20, 20), stroke=4) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = 18
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    cx, cy = W // 2, H // 2
    Llen = min(W, H) // 3
    if direction == "L":
        start, end = (cx + Llen // 2, cy), (cx - Llen, cy)
    elif direction == "R":
        start, end = (cx - Llen // 2, cy), (cx + Llen, cy)
    elif direction == "U":
        start, end = (cx, cy + Llen // 2), (cx, cy - Llen)
    elif direction == "D":
        start, end = (cx, cy - Llen // 2), (cx, cy + Llen)
    elif direction == "UL":
        start, end = (cx + Llen // 2, cy + Llen // 2), (cx - Llen, cy - Llen)
    elif direction == "UR":
        start, end = (cx - Llen // 2, cy + Llen // 2), (cx + Llen, cy - Llen)
    elif direction == "DL":
        start, end = (cx + Llen // 2, cy - Llen // 2), (cx - Llen, cy + Llen)
    else:  # DR
        start, end = (cx - Llen // 2, cy - Llen // 2), (cx + Llen, cy + Llen)
    _arrow(d, start, end, outline, stroke)
    return img

def draw_paper_with_holes(size=(420, 420), holes=None, paper_margin=40, hole_radius=12,
                          bg=(255, 255, 255), paper_fill=(250, 250, 250),
                          outline=(20, 20, 20), stroke=5) -> Image.Image:
    holes = holes or []
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    left, top, right, bottom = paper_margin, paper_margin, W - paper_margin, H - paper_margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    for (x, y) in holes:
        px = (x + 1) / 2.0 * (right - left) + left
        py = (1 - (y + 1) / 2.0) * (bottom - top) + top
        d.ellipse([px - hole_radius, py - hole_radius, px + hole_radius, py + hole_radius],
                  fill=outline, outline=outline, width=1)
    return img

def draw_folded_with_punch(point_folded: Tuple[float, float], size=(220, 220),
                           bg=(255, 255, 255), paper_fill=(250, 250, 250),
                           outline=(20, 20, 20), stroke=4, text_color=(20, 20, 20)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = 16
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    x, y = point_folded
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    r = 10
    d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((left, top - 22), "Punch", fill=text_color, font=font)
    return img

def generate_folding_challenge(rng: random.Random, difficulty="Medium",
                               allow_diagonal=True, style: Optional[Dict] = None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    stroke = (style.get("stroke_width") if style else None) or 5
    text_color = (style.get("text_color") if style else None) or (20, 20, 20)

    if difficulty == "Easy":
        n_folds = rng.choice([1, 2])
    elif difficulty == "Hard":
        n_folds = 3
    else:
        n_folds = rng.choice([2, 3])

    dirs_card = ["L", "R", "U", "D"]
    dirs_diag = ["UL", "UR", "DL", "DR"]
    dirs_all = dirs_card + (dirs_diag if allow_diagonal else [])

    folds = []
    for i in range(n_folds):
        cand = rng.choice(dirs_all) if i == 0 else rng.choice([d for d in dirs_all if d != folds[-1]])
        folds.append(cand)

    def dir_to_axis(d):
        if d in ("L", "R"): return "V"
        if d in ("U", "D"): return "H"
        if d in ("UL", "DR"): return "D1"
        return "D2"

    axes = [dir_to_axis(d) for d in folds]

    px = rng.uniform(-0.65, 0.65)
    py = rng.uniform(-0.65, 0.65)
    point_folded = (px, py)
    holes = unfold_points(point_folded, axes)

    fold_icons = [draw_fold_icon(d, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke) for d in folds]
    row1 = compose_grid(fold_icons, (1, len(fold_icons))) if fold_icons else text_image("No folds", bg=bg, color=outline)
    punch_panel = draw_folded_with_punch(point_folded, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, text_color=text_color)

    W = max(row1.size[0], punch_panel.size[0]) + 32
    H = row1.size[1] + punch_panel.size[1] + 48
    problem_img = Image.new("RGB", (W, H), bg)
    x1 = (W - row1.size[0]) // 2
    problem_img.paste(row1, (x1, 16))
    x2 = (W - punch_panel.size[0]) // 2
    problem_img.paste(punch_panel, (x2, row1.size[1] + 32))

    correct_img = draw_paper_with_holes(holes=holes, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        holes_d1 = unfold_points(point_folded, axes[:-1])
    else:
        holes_d1 = [(px, py)]
    d1 = draw_paper_with_holes(holes=holes_d1, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        wrong_axes = axes.copy()
        idx = rng.randrange(len(wrong_axes))
        wrong_axes[idx] = "H" if wrong_axes[idx] == "V" else ("V" if wrong_axes[idx] == "H" else ("D2" if wrong_axes[idx] == "D1" else "D1"))
        holes_d2 = unfold_points(point_folded, wrong_axes)
    else:
        holes_d2 = [(-px, py)]
    d2 = draw_paper_with_holes(holes=holes_d2, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    d3 = correct_img.rotate(90, expand=False)

    choices = [correct_img, d1, d2, d3]
    rng.shuffle(choices)
    correct_index = choices.index(correct_img)

    rule_desc = "Unfold across each fold line; each fold mirrors hole positions."
    return {
        "problem_img": problem_img,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "prompt": "Paper is folded in the shown order. A hole is punched as marked. Which option shows the fully unfolded paper?",
        "rule_desc": rule_desc,
        "meta": {"type": "folding", "folds": folds}
    }

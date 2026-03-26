"""Avatar and silhouette face obfuscation.

On-device face de-identification using SAM-guided compositing:
- Silhouette: mask filled with solid color (no texture = no FaceNet match)
- Avatar: procedural cartoon face with anti-aliased SAM mask edges
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SILHOUETTE_COLORS = [
    (180, 200, 220),
    (200, 180, 200),
    (220, 200, 180),
    (180, 220, 200),
    (200, 200, 180),
    (220, 180, 200),
    (190, 210, 190),
    (210, 190, 210),
]


def pick_color(detection_id: str = "") -> Tuple[int, int, int]:
    idx = int(hashlib.md5(detection_id.encode()).hexdigest()[:8], 16) % len(SILHOUETTE_COLORS)
    return SILHOUETTE_COLORS[idx]


def _sample_skin_tone(image: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Sample average color from masked face region, shift to pastel."""
    try:
        pixels = image[mask > 0]
        if len(pixels) < 10:
            return None
        avg = pixels.mean(axis=0).astype(int)
        pastel = tuple(int(min(250, avg[i] * 0.85 + 230 * 0.15)) for i in range(3))
        return pastel
    except Exception:
        return None


def _sample_skin_tone_from_crop(crop: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Sample skin tone from inner face region (forehead/cheek area, avoids hair/clothes)."""
    try:
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return None
        # Inner 40% width, centered at 35% height (forehead/cheek, not hair or chin)
        iy = int(h * 0.25)
        ix = int(w * 0.30)
        ih = int(h * 0.30)
        iw = int(w * 0.40)
        inner = crop[iy:iy + ih, ix:ix + iw]
        if inner.size < 30:
            return None
        avg = inner.reshape(-1, 3).mean(axis=0).astype(int)
        # Lighten by just 15% — keep 85% of original color for clear differentiation
        pastel = tuple(int(min(250, avg[i] * 0.85 + 230 * 0.15)) for i in range(3))
        return pastel
    except Exception:
        return None


def _feathered_alpha(mask: np.ndarray, feather_radius: int = 3) -> np.ndarray:
    """Create anti-aliased alpha from binary mask using Gaussian blur at edges."""
    alpha = (mask > 0).astype(np.uint8) * 255
    if feather_radius <= 0:
        return alpha.astype(np.float32) / 255.0
    alpha_pil = Image.fromarray(alpha).filter(
        ImageFilter.GaussianBlur(radius=min(feather_radius, 6))
    )
    return np.array(alpha_pil).astype(np.float32) / 255.0


# Silhouette 

def apply_silhouette_mask(image: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    color = params.get("silhouette_color")
    if color is None:
        color = _sample_skin_tone(image, mask) or pick_color(params.get("detection_id", ""))
    feather = max(1, min(6, int(np.sqrt(mask.sum() / max(np.count_nonzero(mask), 1)))))
    alpha = _feathered_alpha(mask, feather)
    color_arr = np.array(color, dtype=np.float32)
    for c in range(3):
        image[:, :, c] = (color_arr[c] * alpha + image[:, :, c].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return image


def apply_silhouette_bbox(image: Image.Image, x1: int, y1: int, x2: int, y2: int, params: dict):
    color = params.get("silhouette_color")
    if color is None:
        try:
            face_crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
            color = _sample_skin_tone_from_crop(face_crop)
        except Exception:
            color = None
    if color is None:
        color = pick_color(params.get("detection_id", ""))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse([1, 1, w - 2, h - 2], fill=(*color, 255), outline=_darker(color, 40), width=1)
    _draw_face_features(draw, w, h, color)
    image.paste(overlay, (x1, y1), overlay)


# Avatar 

def apply_avatar_mask(image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int], params: dict) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return apply_silhouette_mask(image, mask, params)

    skin = _sample_skin_tone(image, mask) or pick_color(params.get("detection_id", ""))
    avatar = _generate_avatar(w, h, {**params, "_skin_color": skin})
    avatar_np = np.array(avatar.convert("RGB"))

    y_end = min(y1 + h, image.shape[0])
    x_end = min(x1 + w, image.shape[1])
    h_actual = y_end - y1
    w_actual = x_end - x1

    mask_region = mask[y1:y_end, x1:x_end]
    feather = max(1, min(4, w // 20))
    alpha = _feathered_alpha(mask_region, feather)

    for c in range(3):
        orig = image[y1:y_end, x1:x_end, c].astype(np.float32)
        avt = avatar_np[:h_actual, :w_actual, c].astype(np.float32)
        image[y1:y_end, x1:x_end, c] = (avt * alpha[:h_actual, :w_actual] + orig * (1 - alpha[:h_actual, :w_actual])).astype(np.uint8)
    return image


def apply_avatar_bbox(image: Image.Image, x1: int, y1: int, x2: int, y2: int, params: dict):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    # Sample skin tone from the original face region
    try:
        face_crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
        skin = _sample_skin_tone_from_crop(face_crop)
    except Exception:
        skin = None
    if skin is None:
        skin = pick_color(params.get("detection_id", ""))
    avatar = _generate_avatar(w, h, {**params, "_skin_color": skin})
    # Render at 4x for crisp anti-aliased edges, then downscale
    scale = 4
    mask_img = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(mask_img).ellipse([0, 0, w * scale - 1, h * scale - 1], fill=255)
    mask_img = mask_img.resize((w, h), Image.LANCZOS)
    image.paste(avatar.convert("RGB"), (x1, y1), mask_img)


def _generate_avatar(w: int, h: int, params: dict) -> Image.Image:
    avatar_path = params.get("avatar_image_path")
    if avatar_path and Path(avatar_path).exists():
        return Image.open(avatar_path).convert("RGBA").resize((w, h))

    color = params.get("_skin_color") or pick_color(params.get("detection_id", ""))
    # Render at 3x internal resolution for crisp features, then downscale
    scale = 3
    sw, sh = w * scale, h * scale
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    border_w = max(2, sw // 30)
    draw.ellipse([border_w, border_w, sw - border_w - 1, sh - border_w - 1],
                 fill=(*color, 255), outline=_darker(color, 40), width=border_w)
    _draw_face_features(draw, sw, sh, color)
    # Downscale to target size — LANCZOS produces crisp, sharp features
    img = img.resize((w, h), Image.LANCZOS)
    return img


# Face feature rendering 

def _draw_face_features(draw: ImageDraw.Draw, w: int, h: int, bg_color: Tuple[int, int, int]):
    if w < 12 or h < 12:
        return
    if w < 25 or h < 25:
        _draw_minimal_features(draw, w, h, bg_color)
        return

    dark = _darker(bg_color, 80)
    white = (255, 255, 255, 255)
    line_w = max(1, w // 35)

    # Eyebrows
    brow_y = int(h * 0.30)
    brow_w = max(4, int(w * 0.14))
    lbx = int(w * 0.30)
    rbx = int(w * 0.58)
    draw.arc([lbx, brow_y, lbx + brow_w, brow_y + max(3, int(h * 0.06))],
             start=190, end=350, fill=dark, width=line_w)
    draw.arc([rbx, brow_y, rbx + brow_w, brow_y + max(3, int(h * 0.06))],
             start=190, end=350, fill=dark, width=line_w)

    # Eyes — white sclera + colored iris + dark pupil + highlight
    eye_y = int(h * 0.38)
    eye_w = max(5, int(w * 0.12))
    eye_h = max(4, int(h * 0.10))
    lx = int(w * 0.30)
    rx = int(w * 0.60)

    for ex in (lx, rx):
        # Sclera (white)
        draw.ellipse([ex, eye_y, ex + eye_w, eye_y + eye_h], fill=white, outline=dark, width=1)
        # Iris (darker bg color)
        iris_r = max(2, eye_w // 3)
        icx = ex + eye_w // 2
        icy = eye_y + eye_h // 2
        iris_color = _darker(bg_color, 50)
        draw.ellipse([icx - iris_r, icy - iris_r, icx + iris_r, icy + iris_r], fill=(*iris_color, 255))
        # Pupil
        pr = max(1, iris_r // 2)
        draw.ellipse([icx - pr, icy - pr, icx + pr, icy + pr], fill=dark)
        # Highlight dot
        if eye_w > 6:
            hr = max(1, pr // 2)
            draw.ellipse([icx - hr - 1, icy - hr - 1, icx - hr + hr, icy - hr + hr], fill=white)

    # Nose — subtle vertical line with tip
    if h > 40:
        nx = w // 2
        nose_top = int(h * 0.46)
        nose_bot = int(h * 0.56)
        nose_color = _darker(bg_color, 30)
        draw.line([(nx, nose_top), (nx, nose_bot)], fill=nose_color, width=line_w)
        draw.ellipse([nx - 1, nose_bot - 1, nx + 1, nose_bot + 1], fill=nose_color)

    # Mouth
    my = int(h * 0.64)
    mw = max(6, int(w * 0.28))
    mh = max(4, int(h * 0.10))
    mx = int(w * 0.5 - mw / 2)
    mouth_color = _darker(bg_color, 60)
    draw.arc([mx, my, mx + mw, my + mh], start=10, end=170, fill=mouth_color, width=max(1, line_w + 1))

    # Cheek blush (on larger faces)
    if w > 50:
        blush_r = max(3, int(w * 0.06))
        blush_color = (min(255, bg_color[0] + 20), max(0, bg_color[1] - 10), max(0, bg_color[2] - 5), 60)
        ly_blush = int(h * 0.52)
        draw.ellipse([lx - 2, ly_blush, lx - 2 + blush_r * 2, ly_blush + blush_r], fill=blush_color)
        draw.ellipse([rx + eye_w - blush_r, ly_blush, rx + eye_w + blush_r, ly_blush + blush_r], fill=blush_color)


def _draw_minimal_features(draw: ImageDraw.Draw, w: int, h: int, bg_color: Tuple[int, int, int]):
    """Simplified features for small faces (12-25px)."""
    dark = _darker(bg_color, 90)
    dot_r = max(1, w // 10)
    # Two dot eyes
    lx = int(w * 0.35)
    rx = int(w * 0.60)
    ey = int(h * 0.40)
    draw.ellipse([lx, ey, lx + dot_r, ey + dot_r], fill=dark)
    draw.ellipse([rx, ey, rx + dot_r, ey + dot_r], fill=dark)
    # Tiny mouth arc
    my = int(h * 0.62)
    mw = max(3, int(w * 0.2))
    mx = int(w * 0.5 - mw / 2)
    draw.arc([mx, my, mx + mw, my + max(2, int(h * 0.08))],
             start=0, end=180, fill=dark, width=1)


def _darker(color: Tuple[int, int, int], amount: int = 40) -> Tuple[int, ...]:
    return tuple(max(0, c - amount) for c in color)

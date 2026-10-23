"""Portrait-specific conditioning for image-to-3D backends.

The generic ingest path preserves the whole foreground object.  Portrait 3D
models are much more sensitive to how much of the conditioning image is spent
on the face, hair and shoulders, so portrait mode creates a second, deterministic
conditioning image:

* find the face (MediaPipe FaceLandmarker when available; OpenCV Haar fallback),
* crop a head-and-shoulders window around it,
* keep the original alpha matte,
* square-pad without stretching,
* resize to the preset's ``gen_resolution``, and
* apply a small amount of local contrast/sharpening so moustache, beard, hair
  and eyelid edges survive the image encoder.

No generative model is used here.  The transform is deliberately conservative:
identity-bearing pixels are never repainted or hallucinated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from core.config import Config
from core.logging import get_logger

log = get_logger("portrait_conditioning")


@dataclass(frozen=True)
class PortraitConditionResult:
    out_path: Path
    detector: str
    face_bbox: Optional[tuple[int, int, int, int]]
    crop_box: tuple[int, int, int, int]
    source_size: tuple[int, int]
    output_size: tuple[int, int]


def prepare_portrait_condition(
    cfg: Config,
    image_path: Path,
    out_path: Path,
    resolution: int,
) -> PortraitConditionResult:
    """Create a head-and-shoulders RGBA conditioner for portrait reconstruction."""
    with Image.open(image_path) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGBA")

    w, h = img.size
    face_bbox, detector = detect_face_bbox(img, cfg)
    crop_box = _portrait_crop_box(img, face_bbox, cfg)
    cropped = img.crop(crop_box)

    # Improve small identity-bearing edges without turning the source into a
    # stylised image.  Work on RGB only, then restore the untouched alpha.
    alpha = cropped.getchannel("A")
    rgb = cropped.convert("RGB")
    contrast = float(cfg.get("portrait.conditioning.contrast", 1.06))
    sharpen = float(cfg.get("portrait.conditioning.sharpen", 1.20))
    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(max(0.1, contrast))
    if sharpen > 1.0:
        # UnsharpMask is more predictable than repeated SHARPEN filters at
        # high resolution and avoids ringing on already-sharp photographs.
        percent = int(min(250, max(0, (sharpen - 1.0) * 120 + 100)))
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.1, percent=percent, threshold=3))
    cropped = rgb.convert("RGBA")
    cropped.putalpha(alpha)

    result = _fit_square_rgba(
        cropped,
        max(128, int(resolution)),
        padding_fraction=float(cfg.get("portrait.conditioning.padding_fraction", 0.035)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    log.info(
        "portrait condition: detector=%s face=%s crop=%s source=%sx%s output=%sx%s",
        detector, face_bbox, crop_box, w, h, result.width, result.height,
    )
    return PortraitConditionResult(
        out_path=out_path,
        detector=detector,
        face_bbox=face_bbox,
        crop_box=crop_box,
        source_size=(w, h),
        output_size=result.size,
    )


def detect_face_bbox(
    img: Image.Image,
    cfg: Optional[Config] = None,
) -> tuple[Optional[tuple[int, int, int, int]], str]:
    """Return the largest face bounding box in pixel coordinates.

    MediaPipe is preferred because it handles non-frontal portraits well.  A
    bundled OpenCV Haar cascade is the zero-download fallback, which is useful
    in tests and before the MediaPipe task file has been fetched.
    """
    box = _mediapipe_face_bbox(img)
    if box is not None:
        return box, "mediapipe"
    box = _opencv_face_bbox(img)
    if box is not None:
        return box, "opencv_haar"
    return None, "alpha_bbox"


def _mediapipe_face_bbox(img: Image.Image) -> Optional[tuple[int, int, int, int]]:
    try:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        from scripts.preprocess.select_keyframes import _landmarker_model_path  # noqa: PLC0415
    except (ImportError, AttributeError):
        return None

    model = _landmarker_model_path()
    if model is None:
        return None
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            num_faces=1,
        )
        rgb = np.asarray(img.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        w, h = img.size
        x0 = int(max(0, min(xs) * w))
        y0 = int(max(0, min(ys) * h))
        x1 = int(min(w, max(xs) * w))
        y1 = int(min(h, max(ys) * h))
        return _valid_box((x0, y0, x1, y1), w, h)
    except Exception as exc:  # noqa: BLE001 - optional helper must not break ingest
        log.debug("mediapipe portrait face detection failed: %s", exc)
        return None


def _opencv_face_bbox(img: Image.Image) -> Optional[tuple[int, int, int, int]]:
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        rgb = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Equalisation helps on dark studio portraits without modifying the
        # actual conditioning image.
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(max(32, img.width // 12), max(32, img.height // 12)),
        )
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        return _valid_box((int(x), int(y), int(x + fw), int(y + fh)), img.width, img.height)
    except Exception as exc:  # noqa: BLE001
        log.debug("opencv portrait face detection failed: %s", exc)
        return None


def _valid_box(
    box: tuple[int, int, int, int], w: int, h: int,
) -> Optional[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def _alpha_bbox(img: Image.Image) -> Optional[tuple[int, int, int, int]]:
    alpha = np.asarray(img.getchannel("A"))
    ys, xs = np.where(alpha > 12)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _portrait_crop_box(
    img: Image.Image,
    face_bbox: Optional[tuple[int, int, int, int]],
    cfg: Config,
) -> tuple[int, int, int, int]:
    """Build a square-ish head/shoulders crop; coordinates may extend outside image."""
    w, h = img.size
    if face_bbox is None:
        subject = _alpha_bbox(img)
        if subject is None:
            return (0, 0, w, h)
        x0, y0, x1, y1 = subject
        sw, sh = x1 - x0, y1 - y0
        pad = float(cfg.get("portrait.conditioning.subject_padding", 0.06))
        px, py = int(sw * pad), int(sh * pad)
        return (x0 - px, y0 - py, x1 + px, y1 + py)

    x0, y0, x1, y1 = face_bbox
    fh = max(1.0, float(y1 - y0))
    cx = (x0 + x1) / 2.0
    face_widths = float(cfg.get("portrait.conditioning.face_widths", 2.15))
    above = float(cfg.get("portrait.conditioning.above_face", 0.45))
    below = float(cfg.get("portrait.conditioning.below_face", 0.75))

    crop_w = fh * face_widths
    top = y0 - fh * above
    bottom = y1 + fh * below
    crop_h = bottom - top
    side = max(crop_w, crop_h)
    # Keep the face slightly above centre so shoulders survive the square crop.
    centre_y = top + crop_h * 0.50
    left = cx - side / 2.0
    right = cx + side / 2.0
    top = centre_y - side / 2.0
    bottom = centre_y + side / 2.0
    return tuple(int(round(v)) for v in (left, top, right, bottom))


def _fit_square_rgba(img: Image.Image, resolution: int, padding_fraction: float) -> Image.Image:
    padding_fraction = min(0.20, max(0.0, padding_fraction))
    inner = max(1, int(round(resolution * (1.0 - 2.0 * padding_fraction))))
    fitted = img.copy()
    # ``thumbnail`` never enlarges a smaller crop, which would accidentally
    # leave a 600 px portrait floating inside a 1024 px condition image.  Scale
    # both up and down so ``gen_resolution`` truly controls occupied pixels.
    longest = max(1, max(fitted.size))
    scale = inner / longest
    fitted = fitted.resize(
        (max(1, round(fitted.width * scale)), max(1, round(fitted.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (resolution, resolution), (255, 255, 255, 0))
    x = (resolution - fitted.width) // 2
    y = (resolution - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))
    return canvas

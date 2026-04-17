"""Draw bounding boxes and labels on schematic images.

Two main entry points:

- :func:`overlay_netlist` draws boxes from a predicted :class:`Netlist`,
  color-coded by component type. Components without a bbox are listed in
  a legend along the bottom so they aren't silently dropped.
- :func:`overlay_yolo` draws ground-truth YOLO bboxes in a single colour.

:func:`side_by_side` stitches two annotated images together horizontally
with optional titles, for pred-vs-GT comparisons.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from circuit_extract.overlay.annotations import YoloAnnotation
from circuit_extract.schema import Component, Netlist


def _color_for_type(component_type: str) -> tuple[int, int, int]:
    """Deterministic pastel-ish colour per component type."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(component_type)) % 360
    hue = seed / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _load_font(size: int = 12) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    width: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

    # Label with a filled background so it's readable over dense line art.
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    pad = 2
    bg_y2 = y1
    bg_y1 = max(0, bg_y2 - th - 2 * pad)
    draw.rectangle((x1, bg_y1, x1 + tw + 2 * pad, bg_y2), fill=color)
    draw.text((x1 + pad, bg_y1 + pad), label, fill=(0, 0, 0), font=font)


def _open_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


# ---------------------------------------------------------------------------
# Netlist overlay (predictions)
# ---------------------------------------------------------------------------


def _components_with_bboxes(
    netlist: Netlist,
) -> tuple[list[Component], list[Component]]:
    with_bbox = [c for c in netlist.components if c.bbox is not None]
    without_bbox = [c for c in netlist.components if c.bbox is None]
    return with_bbox, without_bbox


def _footer_height(n_lines: int, line_height: int) -> int:
    if n_lines == 0:
        return 0
    return n_lines * line_height + 20  # top/bottom padding


def overlay_netlist(
    image: str | Path | Image.Image,
    netlist: Netlist,
    *,
    title: str | None = None,
) -> Image.Image:
    """Render predicted component bboxes on top of a schematic image.

    Components without a bbox are listed in a footer so nothing is silently
    dropped from the visualisation.
    """
    img = _open_image(image)
    font = _load_font(14)
    small = _load_font(11)
    line_h = 16

    with_bbox, without_bbox = _components_with_bboxes(netlist)

    title_lines = [title] if title else []
    footer_lines: list[str] = []
    if without_bbox:
        footer_lines.append(f"no-bbox ({len(without_bbox)}):")
        # Pack component ids into lines of ~80 chars
        chunk: list[str] = []
        for c in without_bbox:
            label = f"{c.id}[{c.type}]" + (f"={c.value}" if c.value else "")
            chunk.append(label)
            if sum(len(s) + 2 for s in chunk) > 80:
                footer_lines.append("  " + ", ".join(chunk))
                chunk = []
        if chunk:
            footer_lines.append("  " + ", ".join(chunk))

    title_h = _footer_height(len(title_lines), line_h)
    footer_h = _footer_height(len(footer_lines), line_h)

    canvas = Image.new("RGB", (img.width, img.height + title_h + footer_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 6), title, fill=(0, 0, 0), font=font)

    for comp in with_bbox:
        assert comp.bbox is not None
        bbox = (comp.bbox.x1, comp.bbox.y1 + title_h, comp.bbox.x2, comp.bbox.y2 + title_h)
        label = f"{comp.id}:{comp.type}"
        _draw_labeled_box(draw, bbox, label, _color_for_type(comp.type), font)

    for i, line in enumerate(footer_lines):
        draw.text(
            (10, img.height + title_h + 6 + i * line_h),
            line,
            fill=(0, 0, 0),
            font=small,
        )

    return canvas


# ---------------------------------------------------------------------------
# YOLO overlay (ground truth)
# ---------------------------------------------------------------------------


def overlay_yolo(
    image: str | Path | Image.Image,
    annotations: list[YoloAnnotation],
    *,
    title: str | None = None,
) -> Image.Image:
    """Render ground-truth YOLO bounding boxes on top of a schematic image."""
    img = _open_image(image)
    font = _load_font(14)
    line_h = 16

    title_h = line_h + 10 if title else 0
    canvas = Image.new("RGB", (img.width, img.height + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 6), title, fill=(0, 0, 0), font=font)

    for ann in annotations:
        bbox = (ann.x1, ann.y1 + title_h, ann.x2, ann.y2 + title_h)
        label = ann.class_name
        if ann.orientation and ann.orientation != "R0":
            label += f"[{ann.orientation}]"
        _draw_labeled_box(draw, bbox, label, _color_for_type(ann.class_name), font)

    return canvas


# ---------------------------------------------------------------------------
# Side-by-side stitching
# ---------------------------------------------------------------------------


def side_by_side(left: Image.Image, right: Image.Image, gap: int = 16) -> Image.Image:
    """Stitch two images horizontally with a white gap in between."""
    h = max(left.height, right.height)
    w = left.width + gap + right.width
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas

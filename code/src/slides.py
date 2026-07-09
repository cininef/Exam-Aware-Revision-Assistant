from __future__ import annotations

from pathlib import Path

import fitz

from . import config

SLIDE_DIR = config.RAW_DIR / "course_slides"
CACHE_DIR = config.CACHE_DIR / "slide_pages"


def resolve_slide_path(source: str) -> Path:
    path = SLIDE_DIR / source
    if not path.is_file():
        raise FileNotFoundError(f"Slide not found: {source}")
    return path


def render_page_png(source: str, page: int, scale: float = 1.5) -> bytes:
    if page < 1:
        raise ValueError("Page number must be >= 1.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{Path(source).stem}_p{page}_s{scale:.1f}.png"
    cache_path = CACHE_DIR / cache_key
    if cache_path.is_file():
        return cache_path.read_bytes()

    pdf_path = resolve_slide_path(source)
    doc = fitz.open(pdf_path)
    try:
        if page > doc.page_count:
            raise ValueError(f"Page {page} out of range for {source} ({doc.page_count} pages).")
        pix = doc.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    cache_path.write_bytes(png_bytes)
    return png_bytes

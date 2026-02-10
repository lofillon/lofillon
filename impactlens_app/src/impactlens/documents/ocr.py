from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from typing import Iterable

import easyocr
import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def get_reader(langs: tuple[str, ...] = ("en", "fr")) -> easyocr.Reader:
    # EasyOCR is heavy to init; cache it.
    return easyocr.Reader(list(langs), gpu=False)


def ocr_image_bytes(image_bytes: bytes, *, langs: Iterable[str] = ("en", "fr")) -> str:
    reader = get_reader(tuple(langs))
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    # detail=0 returns text strings only; paragraph=True groups lines.
    lines: list[str] = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join([l for l in lines if l.strip()])


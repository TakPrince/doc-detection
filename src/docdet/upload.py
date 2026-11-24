"""Validation helpers for user-uploaded specimen images."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not a safe supported image."""


def decode_uploaded_image(data: bytes) -> Image.Image:
    """Validate and decode an uploaded PNG or JPEG image."""
    if not data:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The uploaded image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise UploadValidationError("Upload a valid PNG or JPEG image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise UploadValidationError("The uploaded image dimensions are too large.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except UploadValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadValidationError("Upload a valid PNG or JPEG image.") from exc


__all__ = [
    "ALLOWED_IMAGE_FORMATS",
    "MAX_IMAGE_PIXELS",
    "MAX_UPLOAD_BYTES",
    "UploadValidationError",
    "decode_uploaded_image",
]

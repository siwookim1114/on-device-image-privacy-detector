"""
Input validation and sanitisation utilities for the FastAPI backend.

Provides:
  - ``validate_image_upload``  — enforces MIME type, file size, and pixel
    dimension limits on uploaded image files.
  - ``sanitize_chat_message``  — length-caps, strips control characters, and
    blocks prompt-injection patterns in user chat messages.

All validation failures raise ``fastapi.HTTPException`` with the standard
error envelope defined in API_CONTRACT.md:

    {
      "error": {
        "code": "UNSUPPORTED_FILE" | "VALIDATION_ERROR",
        "message": "...",
        "details": { ... }
      }
    }
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Optional

from fastapi import HTTPException, UploadFile, status

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False
# Constants

ALLOWED_MIME: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
})

# Maximum raw file size accepted before any processing.
MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB

# Maximum pixel dimension in either axis (width or height).
MAX_DIMENSION: int = 8192

# Maximum number of characters in a single chat message.
MAX_CHAT_LENGTH: int = 2000
# Prompt-injection detection patterns

# Compiled once at import time for performance.  Any match is treated as a
# potential injection attempt and results in a 400 VALIDATION_ERROR.
_RAW_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"you\s+are\s+now",
    r"system\s*:",
    r"forget\s+(everything|your|all)",
    r"disregard\s+(previous|above|all)",
    r"override\s+(safety|security|rules?|protection)",
    r"jailbreak",
    r"pretend\s+you\s+are",
]

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in _RAW_INJECTION_PATTERNS
]
# Standard error envelope factory


def _error_response(code: str, message: str, details: Optional[dict] = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }
# Image upload validation


async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Validate an uploaded image file and return its raw bytes.

    Checks (in order):
      1. MIME type is in ``ALLOWED_MIME``.
      2. Total file size does not exceed ``MAX_FILE_SIZE`` (20 MB).
      3. Image can be decoded by Pillow (guards against corrupt / malformed files).
      4. Neither dimension exceeds ``MAX_DIMENSION`` (8 192 px).

    Raises
    ------
    HTTPException 422 UNSUPPORTED_FILE
        When MIME type is not accepted.
    HTTPException 400 VALIDATION_ERROR
        When size or dimension limits are exceeded, or the file cannot be
        decoded as an image.

    Returns
    -------
    bytes
        Raw file content ready for downstream processing.
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_error_response(
                code="UNSUPPORTED_FILE",
                message=(
                    f"File type '{content_type}' is not supported. "
                    f"Accepted types: {', '.join(sorted(ALLOWED_MIME))}."
                ),
                details={"received_content_type": content_type},
            ),
        )
    # Read in chunks to avoid loading a potentially giant file in one shot
    # before we know it is within the limit.
    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = await file.read(1024 * 256)  # 256 KB per read
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_error_response(
                    code="VALIDATION_ERROR",
                    message=(
                        f"File exceeds the maximum allowed size of "
                        f"{MAX_FILE_SIZE // (1024 * 1024)} MB."
                    ),
                    details={"max_bytes": MAX_FILE_SIZE, "received_bytes_so_far": total_bytes},
                ),
            )
        chunks.append(chunk)

    raw_bytes: bytes = b"".join(chunks)
    if not _PIL_AVAILABLE:
        # If Pillow is not installed we can still accept the bytes; dimension
        # checks will be skipped with a warning.  This should not happen in a
        # production environment.
        import logging
        logging.getLogger(__name__).warning(
            "Pillow is not installed; image dimension validation is disabled."
        )
        return raw_bytes

    try:
        img = _PILImage.open(io.BytesIO(raw_bytes))
        img.verify()  # Detects truncated / corrupt files without full decode
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_response(
                code="VALIDATION_ERROR",
                message="The uploaded file could not be decoded as a valid image.",
                details={"reason": str(exc)},
            ),
        ) from exc

    # Re-open after verify() because verify() leaves the file pointer at EOF.
    try:
        img = _PILImage.open(io.BytesIO(raw_bytes))
        width, height = img.size
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_response(
                code="VALIDATION_ERROR",
                message="Failed to read image dimensions.",
                details={"reason": str(exc)},
            ),
        ) from exc

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_response(
                code="VALIDATION_ERROR",
                message=(
                    f"Image dimensions {width}x{height} exceed the maximum "
                    f"allowed size of {MAX_DIMENSION}x{MAX_DIMENSION} pixels."
                ),
                details={
                    "max_dimension": MAX_DIMENSION,
                    "received_width": width,
                    "received_height": height,
                },
            ),
        )

    return raw_bytes
# Chat message sanitisation


def sanitize_chat_message(message: str) -> str:
    """
    Sanitise a user-supplied chat message before forwarding it to the
    Coordinator Agent.

    Steps applied (in order):
      1. Type guard — rejects non-string input.
      2. Length limit — rejects messages longer than ``MAX_CHAT_LENGTH``.
      3. Unicode normalisation (NFC) for consistent matching.
      4. Control character stripping (removes C0/C1 control codes except
         newline and tab, which are legitimate in multi-line messages).
      5. Prompt-injection pattern scan — raises 400 if a pattern matches.

    Returns
    -------
    str
        The sanitised message, safe to pass to the LLM pipeline.

    Raises
    ------
    HTTPException 400 VALIDATION_ERROR
        On length violation or detected injection pattern.
    """
    if not isinstance(message, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_response(
                code="VALIDATION_ERROR",
                message="Chat message must be a string.",
            ),
        )
    if len(message) > MAX_CHAT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_response(
                code="VALIDATION_ERROR",
                message=(
                    f"Message length {len(message)} exceeds the maximum of "
                    f"{MAX_CHAT_LENGTH} characters."
                ),
                details={"max_length": MAX_CHAT_LENGTH, "received_length": len(message)},
            ),
        )
    message = unicodedata.normalize("NFC", message)
    # Keep printable characters, newline (\n, \r), and horizontal tab (\t).
    # Removes DEL (0x7F), C1 range (0x80–0x9F), and C0 below 0x09.
    sanitised_chars: list[str] = []
    for ch in message:
        cp = ord(ch)
        # Allow printable ASCII / Unicode, plus \t \n \r
        if ch in ("\t", "\n", "\r") or (cp >= 0x20 and cp != 0x7F and not (0x80 <= cp <= 0x9F)):
            sanitised_chars.append(ch)
        # All other control characters are silently dropped

    sanitised = "".join(sanitised_chars).strip()
    for pattern in INJECTION_PATTERNS:
        if pattern.search(sanitised):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_error_response(
                    code="VALIDATION_ERROR",
                    message="Message contains content that cannot be processed.",
                    details={"reason": "Potential prompt injection detected."},
                ),
            )

    return sanitised

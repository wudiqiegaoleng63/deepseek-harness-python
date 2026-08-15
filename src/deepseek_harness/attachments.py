"""Content-addressed image attachments used by the session API."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .models import JsonValue

IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
_ATTACHMENT_ID = re.compile(r"^sha256:([0-9a-f]{64})$")


class AttachmentError(Exception):
    """A stable image admission or retrieval failure."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    attachment_id: str
    media_type: str
    bytes: int
    width: int
    height: int
    name: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "attachmentId": self.attachment_id,
            "mediaType": self.media_type,
            "bytes": self.bytes,
            "width": self.width,
            "height": self.height,
        }
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True, slots=True)
class StoredImage:
    ref: ImageAttachment
    data: bytes


class AttachmentStore:
    """Private, deduplicated image objects below one state root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_image_bytes: int = 10 * 1024 * 1024,
        max_images_per_message: int = 4,
        max_message_image_bytes: int = 20 * 1024 * 1024,
        max_image_pixels: int = 100_000_000,
    ) -> None:
        self.root = Path(root).expanduser().resolve() / "attachments" / "v1"
        self.max_image_bytes = max_image_bytes
        self.max_images_per_message = max_images_per_message
        self.max_message_image_bytes = max_message_image_bytes
        self.max_image_pixels = max_image_pixels

    def validate(self, data: bytes, media_type: str) -> ImageAttachment:
        if media_type not in IMAGE_MEDIA_TYPES:
            raise AttachmentError(f"unsupported image media type: {media_type}", "INVALID_IMAGE")
        if not data:
            raise AttachmentError("image is empty", "INVALID_IMAGE")
        if len(data) > self.max_image_bytes:
            raise AttachmentError("image exceeds the configured byte limit", "IMAGE_TOO_LARGE")
        detected_type, width, height = _image_metadata(data)
        if detected_type != media_type:
            raise AttachmentError(
                "declared image type does not match its bytes", "IMAGE_TYPE_MISMATCH"
            )
        if width * height > self.max_image_pixels:
            raise AttachmentError(
                "image exceeds the configured decoded-pixel limit", "IMAGE_TOO_MANY_PIXELS"
            )
        return ImageAttachment("", media_type, len(data), width, height)

    def save(
        self,
        data: bytes,
        media_type: str,
        *,
        name: str | None = None,
    ) -> ImageAttachment:
        metadata = self.validate(data, media_type)
        digest = hashlib.sha256(data).hexdigest()
        bucket = self.root / "objects" / digest[:2]
        bucket.mkdir(parents=True, exist_ok=True)
        target = bucket / digest
        if not target.exists():
            temporary = bucket / f".{digest}.{os.getpid()}.tmp"
            temporary.write_bytes(data)
            try:
                os.replace(temporary, target)
            except FileExistsError:
                temporary.unlink(missing_ok=True)
        clean_name = _display_name(name)
        return ImageAttachment(
            f"sha256:{digest}",
            metadata.media_type,
            metadata.bytes,
            metadata.width,
            metadata.height,
            clean_name,
        )

    def read(self, ref: ImageAttachment) -> StoredImage:
        match = _ATTACHMENT_ID.fullmatch(ref.attachment_id)
        if match is None:
            raise AttachmentError("attachment reference is invalid", "INVALID_ATTACHMENT_REF")
        target = self.root / "objects" / match.group(1)[:2] / match.group(1)
        try:
            data = target.read_bytes()
        except FileNotFoundError as exc:
            raise AttachmentError("attachment object is missing", "ATTACHMENT_NOT_FOUND") from exc
        if hashlib.sha256(data).hexdigest() != match.group(1):
            raise AttachmentError(
                "stored attachment failed integrity verification", "ATTACHMENT_CORRUPT"
            )
        metadata = self.validate(data, ref.media_type)
        if (
            metadata.bytes != ref.bytes
            or metadata.width != ref.width
            or metadata.height != ref.height
        ):
            raise AttachmentError(
                "stored attachment metadata does not match its reference", "ATTACHMENT_CORRUPT"
            )
        return StoredImage(ref, data)

    @staticmethod
    def decode_base64(value: str) -> bytes:
        if not value:
            raise AttachmentError("image upload is not canonical base64", "INVALID_IMAGE_BASE64")
        try:
            data = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise AttachmentError(
                "image upload is not canonical base64", "INVALID_IMAGE_BASE64"
            ) from exc
        if base64.b64encode(data).decode("ascii") != value:
            raise AttachmentError("image upload is not canonical base64", "INVALID_IMAGE_BASE64")
        return data


def _display_name(value: str | None) -> str | None:
    if value is None:
        return None
    leaf = re.split(r"[/\\]", value)[-1]
    clean = "".join(char for char in leaf if ord(char) >= 32 and ord(char) != 127).strip()
    return clean[:255] or None


def _image_metadata(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        if width and height:
            return "image/png", width, height
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        if width and height:
            return "image/gif", width, height
    if data[:2] == b"\xff\xd8":
        dimensions = _jpeg_dimensions(data)
        if dimensions is not None:
            return "image/jpeg", *dimensions
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        dimensions = _webp_dimensions(data)
        if dimensions is not None:
            return "image/webp", *dimensions
    raise AttachmentError("unsupported or malformed image data", "INVALID_IMAGE")


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    offset = 2
    sof_markers = (
        set(range(0xC0, 0xC4))
        | set(range(0xC5, 0xC8))
        | set(range(0xC9, 0xCC))
        | set(range(0xCD, 0xD0))
    )
    while offset + 3 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in sof_markers and length >= 7:
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width and height:
                return width, height
        offset += length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        marker = data.find(b"\x9d\x01\x2a", 20)
        if marker >= 0 and marker + 7 <= len(data):
            width = int.from_bytes(data[marker + 3 : marker + 5], "little") & 0x3FFF
            height = int.from_bytes(data[marker + 5 : marker + 7], "little") & 0x3FFF
            if width and height:
                return width, height
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = 1 + (bits & 0x3FFF)
        height = 1 + ((bits >> 14) & 0x3FFF)
        return width, height
    return None


__all__ = [
    "AttachmentError",
    "AttachmentStore",
    "IMAGE_MEDIA_TYPES",
    "ImageAttachment",
    "StoredImage",
]

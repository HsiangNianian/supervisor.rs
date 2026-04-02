"""Multimodal message types for images, audio, and files.

Extends the typed message system with content types for handling
non-text data in agent communication.

Example::

    from supervisor.multimodal import ImageMessage, AudioMessage, FileMessage

    img = ImageMessage(
        sender="camera",
        recipient="vision",
        image_url="https://example.com/photo.jpg",
        mime_type="image/jpeg",
    )
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from pydantic import Field

from supervisor.typed_message import TypedMessage


class ImageMessage(TypedMessage):
    """Message containing image data.

    Supports both URL references and base64-encoded inline data.

    Attributes:
        image_url: URL of the image (for remote images).
        image_data: Base64-encoded image data (for inline images).
        mime_type: MIME type (e.g., ``"image/jpeg"``, ``"image/png"``).
        width: Image width in pixels (optional).
        height: Image height in pixels (optional).
        alt_text: Alternative text description.
    """

    msg_type: str = "image"
    image_url: str = ""
    image_data: str = ""
    mime_type: str = "image/png"
    width: Optional[int] = None
    height: Optional[int] = None
    alt_text: str = ""

    @classmethod
    def from_file(
        cls,
        path: str,
        sender: str = "",
        recipient: str = "",
        mime_type: str = "image/png",
    ) -> "ImageMessage":
        """Create an ImageMessage from a local file.

        Args:
            path: Path to the image file.
            sender: Sender agent name.
            recipient: Recipient agent name.
            mime_type: MIME type of the image.

        Returns:
            An :class:`ImageMessage` with base64-encoded data.
        """
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return cls(
            sender=sender,
            recipient=recipient,
            content=f"[image:{path}]",
            image_data=data,
            mime_type=mime_type,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        sender: str = "",
        recipient: str = "",
        alt_text: str = "",
    ) -> "ImageMessage":
        """Create an ImageMessage from a URL.

        Args:
            url: Image URL.
            sender: Sender agent name.
            recipient: Recipient agent name.
            alt_text: Alternative text description.

        Returns:
            An :class:`ImageMessage` with the URL reference.
        """
        return cls(
            sender=sender,
            recipient=recipient,
            content=f"[image:{url}]",
            image_url=url,
            alt_text=alt_text,
        )


class AudioMessage(TypedMessage):
    """Message containing audio data.

    Attributes:
        audio_url: URL of the audio file.
        audio_data: Base64-encoded audio data.
        mime_type: MIME type (e.g., ``"audio/wav"``, ``"audio/mp3"``).
        duration_seconds: Duration of the audio in seconds.
        transcript: Optional text transcript.
    """

    msg_type: str = "audio"
    audio_url: str = ""
    audio_data: str = ""
    mime_type: str = "audio/wav"
    duration_seconds: Optional[float] = None
    transcript: str = ""

    @classmethod
    def from_file(
        cls,
        path: str,
        sender: str = "",
        recipient: str = "",
        mime_type: str = "audio/wav",
    ) -> "AudioMessage":
        """Create an AudioMessage from a local file.

        Args:
            path: Path to the audio file.
            sender: Sender agent name.
            recipient: Recipient agent name.
            mime_type: MIME type of the audio.

        Returns:
            An :class:`AudioMessage` with base64-encoded data.
        """
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return cls(
            sender=sender,
            recipient=recipient,
            content=f"[audio:{path}]",
            audio_data=data,
            mime_type=mime_type,
        )


class FileMessage(TypedMessage):
    """Message containing a file attachment.

    Attributes:
        file_url: URL of the file.
        file_data: Base64-encoded file content.
        filename: Original filename.
        mime_type: MIME type of the file.
        file_size: File size in bytes.
    """

    msg_type: str = "file"
    file_url: str = ""
    file_data: str = ""
    filename: str = ""
    mime_type: str = "application/octet-stream"
    file_size: Optional[int] = None

    @classmethod
    def from_file(
        cls,
        path: str,
        sender: str = "",
        recipient: str = "",
        mime_type: str = "application/octet-stream",
    ) -> "FileMessage":
        """Create a FileMessage from a local file.

        Args:
            path: Path to the file.
            sender: Sender agent name.
            recipient: Recipient agent name.
            mime_type: MIME type of the file.

        Returns:
            A :class:`FileMessage` with base64-encoded data and filename.
        """
        from pathlib import Path as P

        p = P(path)
        with open(path, "rb") as f:
            raw = f.read()
            data = base64.b64encode(raw).decode("utf-8")
        return cls(
            sender=sender,
            recipient=recipient,
            content=f"[file:{p.name}]",
            file_data=data,
            filename=p.name,
            mime_type=mime_type,
            file_size=len(raw),
        )


class MultimodalMessage(TypedMessage):
    """Message containing mixed content types.

    Allows combining text, images, audio, and files in a single message.

    Attributes:
        parts: List of content parts, each a dict with ``"type"`` key.
    """

    msg_type: str = "multimodal"
    parts: List[Dict[str, Any]] = Field(default_factory=list)

    def add_text(self, text: str) -> "MultimodalMessage":
        """Add a text part.

        Args:
            text: Text content.

        Returns:
            Self for chaining.
        """
        self.parts.append({"type": "text", "content": text})
        return self

    def add_image(
        self, url: str = "", data: str = "", mime_type: str = "image/png"
    ) -> "MultimodalMessage":
        """Add an image part.

        Args:
            url: Image URL.
            data: Base64-encoded image data.
            mime_type: MIME type.

        Returns:
            Self for chaining.
        """
        part: Dict[str, Any] = {"type": "image", "mime_type": mime_type}
        if url:
            part["url"] = url
        if data:
            part["data"] = data
        self.parts.append(part)
        return self

    def add_audio(
        self, url: str = "", data: str = "", mime_type: str = "audio/wav"
    ) -> "MultimodalMessage":
        """Add an audio part.

        Args:
            url: Audio URL.
            data: Base64-encoded audio data.
            mime_type: MIME type.

        Returns:
            Self for chaining.
        """
        part: Dict[str, Any] = {"type": "audio", "mime_type": mime_type}
        if url:
            part["url"] = url
        if data:
            part["data"] = data
        self.parts.append(part)
        return self

    def add_file(
        self,
        filename: str,
        url: str = "",
        data: str = "",
        mime_type: str = "application/octet-stream",
    ) -> "MultimodalMessage":
        """Add a file part.

        Args:
            filename: Original filename.
            url: File URL.
            data: Base64-encoded file data.
            mime_type: MIME type.

        Returns:
            Self for chaining.
        """
        part: Dict[str, Any] = {
            "type": "file",
            "filename": filename,
            "mime_type": mime_type,
        }
        if url:
            part["url"] = url
        if data:
            part["data"] = data
        self.parts.append(part)
        return self

    def get_text_parts(self) -> List[str]:
        """Return all text content from parts."""
        return [p["content"] for p in self.parts if p.get("type") == "text"]

    def get_image_parts(self) -> List[Dict[str, Any]]:
        """Return all image parts."""
        return [p for p in self.parts if p.get("type") == "image"]

    def get_audio_parts(self) -> List[Dict[str, Any]]:
        """Return all audio parts."""
        return [p for p in self.parts if p.get("type") == "audio"]

    def get_file_parts(self) -> List[Dict[str, Any]]:
        """Return all file parts."""
        return [p for p in self.parts if p.get("type") == "file"]


__all__ = [
    "AudioMessage",
    "FileMessage",
    "ImageMessage",
    "MultimodalMessage",
]

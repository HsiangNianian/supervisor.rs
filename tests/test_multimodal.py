"""Tests for the multimodal module (Phase 7.1)."""

import base64
import json

import pytest

from supervisor.multimodal import (
    AudioMessage,
    FileMessage,
    ImageMessage,
    MultimodalMessage,
)


class TestImageMessage:
    """Tests for the ImageMessage class."""

    def test_create_from_url(self):
        msg = ImageMessage.from_url(
            "https://example.com/photo.jpg",
            sender="camera",
            recipient="vision",
            alt_text="A test photo",
        )
        assert msg.msg_type == "image"
        assert msg.image_url == "https://example.com/photo.jpg"
        assert msg.alt_text == "A test photo"
        assert msg.sender == "camera"

    def test_create_from_file(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
        msg = ImageMessage.from_file(
            str(img_file), sender="camera", recipient="vision"
        )
        assert msg.msg_type == "image"
        assert msg.image_data != ""
        # Verify it's valid base64
        decoded = base64.b64decode(msg.image_data)
        assert decoded[:4] == b"\x89PNG"

    def test_defaults(self):
        msg = ImageMessage(sender="a", recipient="b", content="test")
        assert msg.mime_type == "image/png"
        assert msg.width is None
        assert msg.height is None

    def test_serialization(self):
        msg = ImageMessage(
            sender="a",
            recipient="b",
            content="image",
            image_url="https://example.com/img.png",
        )
        data = msg.to_json()
        restored = ImageMessage.model_validate_json(data)
        assert restored.image_url == "https://example.com/img.png"


class TestAudioMessage:
    """Tests for the AudioMessage class."""

    def test_create_from_file(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 20)
        msg = AudioMessage.from_file(str(audio_file), sender="mic", recipient="asr")
        assert msg.msg_type == "audio"
        assert msg.audio_data != ""

    def test_defaults(self):
        msg = AudioMessage(sender="a", recipient="b", content="test")
        assert msg.mime_type == "audio/wav"
        assert msg.duration_seconds is None
        assert msg.transcript == ""

    def test_with_transcript(self):
        msg = AudioMessage(
            sender="a",
            recipient="b",
            content="audio",
            transcript="Hello world",
        )
        assert msg.transcript == "Hello world"


class TestFileMessage:
    """Tests for the FileMessage class."""

    def test_create_from_file(self, tmp_path):
        doc = tmp_path / "readme.txt"
        doc.write_text("Hello, World!")
        msg = FileMessage.from_file(str(doc), sender="uploader", recipient="processor")
        assert msg.msg_type == "file"
        assert msg.filename == "readme.txt"
        assert msg.file_size == 13
        decoded = base64.b64decode(msg.file_data)
        assert decoded == b"Hello, World!"

    def test_defaults(self):
        msg = FileMessage(sender="a", recipient="b", content="test")
        assert msg.mime_type == "application/octet-stream"
        assert msg.file_size is None


class TestMultimodalMessage:
    """Tests for the MultimodalMessage class."""

    def test_add_text(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_text("Hello")
        assert len(msg.parts) == 1
        assert msg.parts[0]["type"] == "text"
        assert msg.parts[0]["content"] == "Hello"

    def test_add_image(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_image(url="https://example.com/img.png")
        assert len(msg.parts) == 1
        assert msg.parts[0]["type"] == "image"

    def test_add_audio(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_audio(url="https://example.com/audio.wav")
        assert len(msg.parts) == 1
        assert msg.parts[0]["type"] == "audio"

    def test_add_file(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_file("report.pdf", url="https://example.com/report.pdf")
        assert len(msg.parts) == 1
        assert msg.parts[0]["type"] == "file"
        assert msg.parts[0]["filename"] == "report.pdf"

    def test_chaining(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        result = (
            msg.add_text("Hello")
            .add_image(url="https://example.com/img.png")
            .add_audio(data="base64data")
        )
        assert result is msg
        assert len(msg.parts) == 3

    def test_get_parts_by_type(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_text("Hello")
        msg.add_text("World")
        msg.add_image(url="https://example.com/img.png")
        msg.add_audio(url="https://example.com/audio.wav")
        msg.add_file("doc.pdf")

        assert len(msg.get_text_parts()) == 2
        assert msg.get_text_parts() == ["Hello", "World"]
        assert len(msg.get_image_parts()) == 1
        assert len(msg.get_audio_parts()) == 1
        assert len(msg.get_file_parts()) == 1

    def test_serialization(self):
        msg = MultimodalMessage(sender="a", recipient="b", content="multi")
        msg.add_text("Hello")
        msg.add_image(url="https://example.com/img.png")
        data = msg.to_json()
        restored = MultimodalMessage.model_validate_json(data)
        assert len(restored.parts) == 2

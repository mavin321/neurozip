def text_to_bytes(text: str) -> bytes:
    """Encode text to UTF-8 bytes."""
    return text.encode("utf-8", errors="replace")


def file_to_bytes(path: str) -> bytes:
    """Load a text or binary file as raw bytes."""
    with open(path, "rb") as f:
        return f.read()

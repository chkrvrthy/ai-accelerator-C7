from __future__ import annotations

# Converts binary image data into base64 text
import base64

# Helps detect image types like png/jpg
import mimetypes

# Modern way to work with file paths
from pathlib import Path

# Allows flexible input types
from typing import Any


# ---------------------------------------------------------
# Convert an image file into a base64 data URL
# ---------------------------------------------------------
def image_file_to_data_url(file_path: str) -> str:

    # Convert string path into Path object
    path = Path(file_path)

    # Detect image MIME type
    # Example:
    # image/png
    # image/jpeg
    mime_type, _ = mimetypes.guess_type(path.name)

    # Validate that the file is actually an image
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Please provide a supported image file.")

    # Read image bytes and convert to base64 text
    encoded_text = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    # Return AI-compatible data URL
    return f"data:{mime_type};base64,{encoded_text}"


# ---------------------------------------------------------
# Extract file path from Gradio upload formats
# ---------------------------------------------------------
def get_uploaded_file_path(file_value: Any) -> str | None:
    """
    Normalize different Gradio file formats into a path string.
    """

    # -----------------------------------------------------
    # CASE 1:
    # File already comes as a string path
    # Example:
    # "cat.png"
    # -----------------------------------------------------
    if isinstance(file_value, str):
        return file_value

    # -----------------------------------------------------
    # CASE 2:
    # File comes as a dictionary
    # Example:
    # {"path": "cat.png"}
    # or
    # {"name": "cat.png"}
    # -----------------------------------------------------
    if isinstance(file_value, dict):

        # Try "path" first
        # If missing, try "name"
        return (
            file_value.get("path")
            or file_value.get("name")
        )

    # -----------------------------------------------------
    # CASE 3:
    # File comes as an object
    # Example:
    # file.path
    # file.name
    # -----------------------------------------------------

    # getattr safely checks attributes
    return (
        getattr(file_value, "path", None)
        or getattr(file_value, "name", None)
    )


# ---------------------------------------------------------
# Build OpenRouter multimodal user content
# ---------------------------------------------------------
def build_user_content(
    message: dict[str, Any]
) -> str | list[dict[str, Any]]:

    """
    Convert user text + uploaded images into
    OpenRouter vision message format.
    """

    # Get user text
    # If missing, use empty string
    text = (message.get("text") or "").strip()

    # Get uploaded files list
    # If missing, use empty list
    files = message.get("files") or []

    # Final multimodal content list
    content: list[dict[str, Any]] = []

    # -----------------------------------------------------
    # Add text block if user typed something
    # -----------------------------------------------------
    if text:
        content.append({
            "type": "text",
            "text": text
        })

    # -----------------------------------------------------
    # Process uploaded images
    # -----------------------------------------------------
    for file_value in files:

        # Extract file path
        file_path = get_uploaded_file_path(file_value)

        # Continue only if valid path exists
        if file_path:

            # Convert image to base64 data URL
            image_url = image_file_to_data_url(file_path)

            # Add image block for AI model
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })

    # -----------------------------------------------------
    # If user sent nothing
    # -----------------------------------------------------
    if not content:
        return "Please send text or upload an image."

    # -----------------------------------------------------
    # If only text exists,
    # return plain string instead of list
    # -----------------------------------------------------
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]

    # -----------------------------------------------------
    # If image exists but no text,
    # add automatic instruction
    # -----------------------------------------------------
    if not text:
        content.insert(0, {
            "type": "text",
            "text": "Please analyze this image."
        })

    # Return multimodal content
    return content


# ---------------------------------------------------------
# Build full chat history for OpenRouter
# ---------------------------------------------------------
def build_multimodal_messages(
    history: list[dict[str, Any]],
    current_message: dict[str, Any],
) -> list[dict[str, Any]]:

    messages: list[dict[str, Any]] = []

    # Add previous history
    for item in history:

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        messages.append({
            "role": role,
            "content": content
        })

    # Add latest user message
    messages.append({
        "role": "user",
        "content": build_user_content(current_message)
    })

    return messages


# ---------------------------------------------------------
# MAIN BLOCK
# Runs only when this file is executed directly
# ---------------------------------------------------------
if __name__ == "__main__":

    print("Running multimodal message builder...\n")

    # -----------------------------------------------------
    # Previous conversation history
    # -----------------------------------------------------
    history = [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi! How can I help?"
        }
    ]

    # -----------------------------------------------------
    # Current user message
    # -----------------------------------------------------
    current_message = {
        "text": "Please analyze this image",
        "files": []
    }

    # -----------------------------------------------------
    # Build final messages
    # -----------------------------------------------------
    messages = build_multimodal_messages(
        history,
        current_message
    )

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------
    print(messages)
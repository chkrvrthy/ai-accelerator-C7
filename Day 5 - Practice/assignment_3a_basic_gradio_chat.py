from __future__ import annotations

import os
from typing import Any

import gradio as gr
from openai import OpenAI

from assignment_2_multimodal_messages import build_multimodal_messages


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
APP_TITLE = "Basic Multimodal Chat"


def stream_basic_chat(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    api_key: str,
):
    """Stream a multimodal response into Gradio."""

    # Use textbox API key OR environment variable
    api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()

    # If no API key found
    if not api_key:
        yield "Add your OpenRouter API key first."
        return

    # Create OpenAI/OpenRouter client
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key
    )

    # Create streaming chat completion
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=build_multimodal_messages(history, message),
        stream=True,
        extra_body={
            "provider": {
                "data_collection": "deny"
            }
        }
    )

    # Store growing response
    answer = ""

    # Stream chunks one-by-one
    for chunk in response:

        # Extract text delta safely
        delta = chunk.choices[0].delta.content

        # If chunk has content
        if delta:

            # Add chunk to answer
            answer += delta

            # Send updated answer to Gradio UI
            yield answer


def build_demo() -> gr.ChatInterface:
    """Create the basic Gradio app."""

    return gr.ChatInterface(
        fn=stream_basic_chat,

        # Enable image + text support
        multimodal=True,

        # App title
        title=APP_TITLE,

        # Multimodal textbox for images
        textbox=gr.MultimodalTextbox(
            file_types=["image"]
        ),

        # Extra input for API key
        additional_inputs=[
            gr.Textbox(
                label="OpenRouter API Key",
                type="password"
            )
        ]
    )


# Build app
demo = build_demo()


# Run app
if __name__ == "__main__":
    demo.launch(inbrowser=True,share=True)
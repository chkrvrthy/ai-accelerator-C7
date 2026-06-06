from __future__ import annotations

import os
from typing import Any

import gradio as gr
from openai import OpenAI

from assignment_2_multimodal_messages import build_multimodal_messages


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MODEL_CHOICES = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]
APP_TITLE = "Kalyan's Advanced Multimodal Chat"


def stream_advanced_chat(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Stream a configurable multimodal response into Gradio."""

    # Use textbox key first, otherwise fallback to environment variable
    api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()

    # Stop if no API key is found
    if not api_key:
        yield "Add your OpenRouter API key first."
        return

    # Create OpenAI-compatible client for OpenRouter
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    # Create streaming chat completion request
    response = client.chat.completions.create(
        model=(model or DEFAULT_MODEL).strip(),
        messages=build_multimodal_messages(history, message),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_body={"provider": {"data_collection": "deny"}},
    )

    # Build streamed answer progressively
    answer = ""

    # Stream chunks from model
    for chunk in response:

        # Some chunks may not contain content
        delta = chunk.choices[0].delta.content

        # Append content if present
        if delta:
            answer += delta

            # Yield growing answer to Gradio UI
            yield answer


def build_demo() -> gr.ChatInterface:
    """Create the advanced Gradio app with configurable controls."""

    # Password field for API key
    api_key_input = gr.Textbox(
        label="OpenRouter API Key",
        type="password",
    )

    # Dropdown for model selection
    model_input = gr.Dropdown(
        choices=MODEL_CHOICES,
        value=DEFAULT_MODEL,
        label="Model",
    )

    # Slider for creativity/randomness
    temperature_input = gr.Slider(
        minimum=0,
        maximum=1.5,
        value=0.7,
        step=0.1,
        label="Temperature",
    )

    # Slider for response size
    max_tokens_input = gr.Slider(
        minimum=64,
        maximum=2048,
        value=512,
        step=64,
        label="Max Tokens",
    )

    # Create Gradio chat application
    return gr.ChatInterface(
        fn=stream_advanced_chat,
        multimodal=True,
        additional_inputs=[
            api_key_input,
            model_input,
            temperature_input,
            max_tokens_input,
        ],
        title=APP_TITLE,
    )


# Build app instance
demo = build_demo()


# Start app
if __name__ == "__main__":
    demo.launch(inbrowser=True,share=True)

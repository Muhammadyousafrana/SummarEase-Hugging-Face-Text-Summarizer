
import re
import threading

import gradio as gr
from transformers import pipeline

# --------------------------------------------------
# PyTorch and GPU detection
# --------------------------------------------------

try:
    import torch

    HAS_TORCH = True
    GPU_AVAILABLE = torch.cuda.is_available()
    TORCH_ERROR = None

except Exception as e:
    torch = None
    HAS_TORCH = False
    GPU_AVAILABLE = False
    TORCH_ERROR = f"{type(e).__name__}: {e}"


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_CHOICES = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-6-6",
    "t5-small",
    "google/pegasus-xsum",
]

# Keep loaded models on the server rather than in Gradio session state.
MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(model_name: str, use_gpu: bool):
    """Load the requested summarization model and cache it."""

    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch could not be imported. "
            f"Details: {TORCH_ERROR}. "
            "Check the torch installation and Space build logs."
        )

    # Never select a GPU that is unavailable.
    device = 0 if use_gpu and GPU_AVAILABLE else -1
    cache_key = f"{model_name}::device={device}"

    with MODEL_CACHE_LOCK:
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key], (
                f"Model loaded (cached): {model_name} "
                f"on {'GPU' if device >= 0 else 'CPU'}"
            )

        try:
            summarizer = pipeline(
                task="summarization",
                model=model_name,
                device=device,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                f"{type(e).__name__}: {e}. "
                "Verify requirements.txt and model compatibility."
            ) from e

        MODEL_CACHE[cache_key] = summarizer

    status = (
        f"Model loaded: {model_name} "
        f"on {'GPU' if device >= 0 else 'CPU'}"
    )

    if use_gpu and not GPU_AVAILABLE:
        status += " (GPU unavailable; using CPU)"

    return summarizer, status


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def chunk_text(text: str, max_chars: int = 1000):
    """Split text into manageable chunks while preserving sentences."""

    text = text.strip()

    if not text:
        return []

    if max_chars < 1:
        raise ValueError("Chunk size must be greater than zero.")

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_length = len(sentence)

        # Split exceptionally long sentences into smaller pieces.
        if sentence_length > max_chars:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0

            for start in range(0, sentence_length, max_chars):
                chunks.append(sentence[start:start + max_chars])

            continue

        additional_length = sentence_length
        if current_sentences:
            additional_length += 1  # Space between sentences.

        if current_length + additional_length <= max_chars:
            current_sentences.append(sentence)
            current_length += additional_length
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            current_sentences = [sentence]
            current_length = sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# --------------------------------------------------
# Summarization
# --------------------------------------------------

def generate_summary(summarizer, text: str, max_length: int):
    """Generate one summary with safe generation parameters."""

    min_length = min(20, max(1, max_length // 4))
    min_length = min(min_length, max_length - 1)

    result = summarizer(
        text,
        max_length=int(max_length),
        min_length=int(min_length),
        do_sample=False,
        truncation=True,
    )

    if not result or not result[0].get("summary_text"):
        raise RuntimeError("The model returned an empty summary.")

    return result[0]["summary_text"].strip()


def summarize_with_chunking(
    text: str,
    summarizer,
    max_chars: int,
    chunk_summary_len: int,
    final_summary_len: int,
):
    """Summarize chunks individually, then combine their summaries."""

    chunks = chunk_text(text, max_chars=max_chars)

    if not chunks:
        return ""

    partial_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        try:
            partial_summary = generate_summary(
                summarizer,
                chunk,
                int(chunk_summary_len),
            )
            partial_summaries.append(partial_summary)

        except Exception as e:
            raise RuntimeError(
                f"Failed to summarize chunk {index}/{len(chunks)}: "
                f"{type(e).__name__}: {e}"
            ) from e

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined_text = "\n\n".join(partial_summaries)

    # Summarize the combined partial summaries.
    try:
        final_summary = generate_summary(
            summarizer,
            combined_text,
            int(final_summary_len),
        )
        return final_summary

    except Exception:
        # Preserve the valid partial summaries if the final pass fails.
        return combined_text


# --------------------------------------------------
# Main Gradio actions
# --------------------------------------------------

def summarize_action(
    text,
    model_name,
    device_choice,
    max_chars,
    chunk_summary_len,
    final_summary_len,
):
    if not text or not text.strip():
        return "", "Please enter some text to summarize."

    use_gpu = (
        device_choice in ("Auto", "GPU")
        and GPU_AVAILABLE
    )

    try:
        summarizer, status = load_model(model_name, use_gpu)

    except Exception as e:
        return "", f"Error loading model: {e}"

    try:
        summary = summarize_with_chunking(
            text=text,
            summarizer=summarizer,
            max_chars=int(max_chars),
            chunk_summary_len=int(chunk_summary_len),
            final_summary_len=int(final_summary_len),
        )

        return summary, status

    except Exception as e:
        return "", f"Error during summarization: {e}"


def preload_model(model_name, device_choice):
    """Load the selected model when the user changes settings."""

    use_gpu = (
        device_choice in ("Auto", "GPU")
        and GPU_AVAILABLE
    )

    try:
        _, status = load_model(model_name, use_gpu)
        return status

    except Exception as e:
        return f"Error loading model: {e}"


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SummarEase — Text Summarizer",
) as demo:

    gr.Markdown("# SummarEase")
    gr.Markdown(
        "Summarize long text using pretrained Hugging Face "
        "Transformer models."
    )

    with gr.Row():

        with gr.Column(scale=3):

            text_input = gr.Textbox(
                label="Input text",
                lines=14,
                placeholder=(
                    "Paste an article, paragraph, or long text here..."
                ),
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "Summarize",
                    variant="primary",
                )
                copy_btn = gr.Button("Copy summary")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[0],
                    label="Summarization model",
                )

                device_radio = gr.Radio(
                    choices=["Auto", "CPU", "GPU"],
                    value="Auto" if GPU_AVAILABLE else "CPU",
                    label="Device",
                )

            with gr.Row():
                max_chars_slider = gr.Slider(
                    label="Chunk size (characters)",
                    minimum=500,
                    maximum=4000,
                    value=1000,
                    step=100,
                )

                chunk_summary_len = gr.Slider(
                    label="Per-chunk summary max tokens",
                    minimum=50,
                    maximum=250,
                    value=100,
                    step=10,
                )

                final_summary_len = gr.Slider(
                    label="Final summary max tokens",
                    minimum=30,
                    maximum=250,
                    value=130,
                    step=10,
                )

        with gr.Column(scale=2):

            gr.Markdown("### Output")

            summary_output = gr.Textbox(
                label="Summary",
                lines=14,
                interactive=False,
                elem_id="summary-output",
            )

            initial_status = (
                "GPU available"
                if GPU_AVAILABLE
                else (
                    "CPU mode (GPU unavailable)"
                    if HAS_TORCH
                    else f"PyTorch import failed: {TORCH_ERROR}"
                )
            )

            status_box = gr.Textbox(
                label="Model status",
                interactive=False,
                value=initial_status,
            )

    # Summarize button
    submit_btn.click(
        fn=summarize_action,
        inputs=[
            text_input,
            model_dropdown,
            device_radio,
            max_chars_slider,
            chunk_summary_len,
            final_summary_len,
        ],
        outputs=[summary_output, status_box],
        api_name="summarize",
    )

    # Press Enter in the input textbox to summarize.
    text_input.submit(
        fn=summarize_action,
        inputs=[
            text_input,
            model_dropdown,
            device_radio,
            max_chars_slider,
            chunk_summary_len,
            final_summary_len,
        ],
        outputs=[summary_output, status_box],
    )

    # Copy summary using the Gradio JavaScript event API.
    copy_btn.click(
        fn=None,
        js="""() => {
            const container = document.getElementById('summary-output');
            const textarea = container?.querySelector('textarea');

            if (!textarea) {
                return [];
            }

            return [textarea.value];
        }""",
        outputs=[],
    )

    # Preload the selected model when settings change.
    model_dropdown.change(
        fn=preload_model,
        inputs=[model_dropdown, device_radio],
        outputs=[status_box],
    )

    device_radio.change(
        fn=preload_model,
        inputs=[model_dropdown, device_radio],
        outputs=[status_box],
    )


if __name__ == "__main__":
    demo.queue().launch()

import re
import gradio as gr
from transformers import pipeline
import torch
GPU_AVAILABLE = torch.cuda.is_available()

# Optional GPU detection
try:
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    GPU_AVAILABLE = False

# Models to offer in the dropdown
MODEL_CHOICES = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-6-6",
    "t5-small",
    "google/pegasus-xsum",
]

# Simple model cache stored in Gradio state

def load_model(model_name: str, use_gpu: bool, cache: dict):
    device = 0 if use_gpu and GPU_AVAILABLE else -1
    key = f"{model_name}::device={device}"

    if key in cache:
        return cache[key], (
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
            f"Could not load '{model_name}'. "
            f"Check your requirements.txt and model compatibility. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    cache[key] = summarizer

    return summarizer, (
        f"Model loaded: {model_name} "
        f"on {'GPU' if device >= 0 else 'CPU'}"
    )

def chunk_text(text: str, max_chars: int = 1000):
    """Split text into chunks of up to max_chars while trying to preserve sentences."""
    # Split into sentences using simple punctuation based heuristic
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for s in sentences:
        s_len = len(s)
        if current_len + s_len <= max_chars:
            current.append(s)
            current_len += s_len
        else:
            if current:
                chunks.append(" ".join(current))
            # If single sentence longer than max_chars, hard split
            if s_len > max_chars:
                for i in range(0, s_len, max_chars):
                    chunks.append(s[i:i+max_chars])
                current = []
                current_len = 0
            else:
                current = [s]
                current_len = s_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_with_chunking(text: str, summarizer, max_chars: int, chunk_summary_len: int, final_summary_len: int):
    # Create chunks
    chunks = chunk_text(text, max_chars=max_chars)
    partial_summaries = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=chunk_summary_len, min_length=max(20, chunk_summary_len//4), do_sample=False, truncation=True)
            partial_summaries.append(out[0]["summary_text"].strip())
        except Exception:
            # As a fallback, truncate the chunk and try again
            truncated = c[: max_chars]
            out = summarizer(truncated, max_length=chunk_summary_len, min_length=max(20, chunk_summary_len//4), do_sample=False, truncation=True)
            partial_summaries.append(out[0]["summary_text"].strip())

    if not partial_summaries:
        return ""

    # If there is more than one partial summary, combine and summarize again
    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n".join(partial_summaries)
    try:
        final = summarizer(combined, max_length=final_summary_len, min_length=max(30, final_summary_len//4), do_sample=False, truncation=True)
        return final[0]["summary_text"].strip()
    except Exception:
        # If final summarization fails for some reason, return combined partials
        return combined


def summarize_action(text: str, model_name: str, device_choice: str, max_chars: int, chunk_summary_len: int, final_summary_len: int, cache: dict):
    if not text or not text.strip():
        return "", "Please enter some text to summarize.", cache

    use_gpu = False
    if device_choice == "Auto":
        use_gpu = GPU_AVAILABLE
    elif device_choice == "GPU":
        use_gpu = True
    else:
        use_gpu = False

    # Load (or fetch cached) model
    try:
        summarizer, status = load_model(model_name, use_gpu, cache)
    except Exception as e:
        return "", f"Error loading model: {e}", cache

    # Summarize with chunking
    try:
        summary = summarize_with_chunking(text, summarizer, max_chars=max_chars, chunk_summary_len=chunk_summary_len, final_summary_len=final_summary_len)
    except Exception as e:
        return "", f"Error during summarization: {e}", cache

    return summary, status, cache


# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="SummarEase — Text Summarizer") as demo:
    gr.Markdown("# SummarEase")
    gr.Markdown("A Gradio demo that converts long text into concise summaries using Hugging Face Transformers.")

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(label="Input text", lines=14, placeholder="Paste a paragraph, article, or long text here...")
            with gr.Row():
                submit_btn = gr.Button("Summarize", variant="primary")
                copy_btn = gr.Button("Copy summary")

            # Controls
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
                device_radio = gr.Radio(choices=["Auto", "CPU", "GPU"], value=("Auto" if GPU_AVAILABLE else "CPU"), label="Device")

            with gr.Row():
                max_chars_slider = gr.Slider(label="Chunk size (characters)", minimum=500, maximum=4000, value=1000, step=100)
                chunk_summary_len = gr.Slider(label="Per-chunk summary max tokens", minimum=50, maximum=250, value=100, step=10)
                final_summary_len = gr.Slider(label="Final summary max tokens", minimum=30, maximum=250, value=130, step=10)

        with gr.Column(scale=2):
            gr.Markdown("### Output")
            summary_output = gr.Textbox(label="Summary", lines=14, interactive=False, elem_id="summary-output")
            status_box = gr.Textbox(label="Model status", interactive=False, value=("GPU available" if GPU_AVAILABLE else "GPU not available"))

    # Hidden state to cache loaded pipeline objects
    model_cache = gr.State({})

    # When user clicks Summarize -> call summarize_action
    submit_btn.click(
        fn=summarize_action,
        inputs=[text_input, model_dropdown, device_radio, max_chars_slider, chunk_summary_len, final_summary_len, model_cache],
        outputs=[summary_output, status_box, model_cache],
        api_name="summarize",
    )

    # Allow Enter key in textbox to submit
    text_input.submit(
        fn=summarize_action,
        inputs=[text_input, model_dropdown, device_radio, max_chars_slider, chunk_summary_len, final_summary_len, model_cache],
        outputs=[summary_output, status_box, model_cache],
    )

    # Copy button uses a small JS snippet to access the output elem by id
    
    copy_btn.click(
        fn=None,
        js="""() => {
            const container = document.getElementById('summary-output');
            const textarea = container?.querySelector('textarea');
            if (textarea) {
                navigator.clipboard.writeText(textarea.value);
            }
        }"""
    )


    # Provide a lightweight model loader action so users can pre-load a model and see status
    def preload_model(model_name: str, device_choice: str, cache: dict):
        use_gpu = False
        if device_choice == "Auto":
            use_gpu = GPU_AVAILABLE
        elif device_choice == "GPU":
            use_gpu = True
        else:
            use_gpu = False
        try:
            _, status = load_model(model_name, use_gpu, cache)
            return status, cache
        except Exception as e:
            return f"Error loading model: {e}", cache

    model_dropdown.change(fn=preload_model, inputs=[model_dropdown, device_radio, model_cache], outputs=[status_box, model_cache])
    device_radio.change(fn=preload_model, inputs=[model_dropdown, device_radio, model_cache], outputs=[status_box, model_cache])


if __name__ == "__main__":
    demo.launch()

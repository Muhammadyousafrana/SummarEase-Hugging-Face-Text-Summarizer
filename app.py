import gradio as gr
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-cnn"

try:
    summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
except Exception:
    summarizer = pipeline("summarization", device=-1)


EXAMPLES = [
    (
        "Artificial intelligence is transforming the way people work, learn, and communicate. By analyzing large amounts of information quickly, AI systems can support decision-making, automate repetitive tasks, and uncover patterns that may not be obvious to humans. In education, healthcare, and business, these tools are becoming increasingly valuable for improving productivity and insight.",
    ),
    (
        "Climate change continues to affect ecosystems, weather patterns, and human communities across the globe. Rising temperatures, melting ice, and shifting rainfall patterns can increase the risk of floods, droughts, and crop failures. Governments, scientists, and organizations are working together to reduce emissions, adapt to changing conditions, and build more resilient societies.",
    ),
    (
        "Remote work became a major part of daily life for many companies after digital collaboration tools improved significantly. Employees gained flexibility, while employers explored new models for productivity and communication. Although remote work offers convenience and cost savings, it also presents challenges related to teamwork, engagement, and work-life balance.",
    ),
]


def summarize_text(text: str):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to summarize.")

    cleaned_text = text.strip()
    result = summarizer(
        cleaned_text,
        max_length=130,
        min_length=30,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


with gr.Blocks(theme=gr.themes.Soft(), title="SummarEase") as demo:
    gr.Markdown(
        "# SummarEase\n"
        "A simple text summarization app powered by Hugging Face Transformers."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input text",
                lines=12,
                placeholder="Paste a paragraph, article, or any long text here...",
            )
            submit_btn = gr.Button("Summarize", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Quick examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                cache_examples=True,
            )

    summary_output = gr.Textbox(
        label="Summary",
        lines=8,
        interactive=False,
        placeholder="Your generated summary will appear here...",
    )

    submit_btn.click(fn=summarize_text, inputs=text_input, outputs=summary_output)
    text_input.submit(fn=summarize_text, inputs=text_input, outputs=summary_output)


if __name__ == "__main__":
    demo.launch()

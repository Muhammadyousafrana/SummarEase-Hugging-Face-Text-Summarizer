---
title: hugging-face-demo
emoji: 🐠
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
licence: cc
license: cc
---


[![Deploy to Hugging Face Spaces](https://github.com/Muhammadyousafrana/hugging-face-demo/actions/workflows/main.yml/badge.svg)](https://github.com/Muhammadyousafrana/hugging-face-demo/actions/workflows/main.yml)

# SummarEase — Hugging Face Text Summarizer

SummarEase is a lightweight Gradio demo that uses the Hugging Face Transformers summarization pipeline to convert long text into concise summaries. It's designed for quick local testing and for easy deployment to Hugging Face Spaces.

Demo (example): https://huggingface.co/spaces/MYousafRana/demo
(Replace the link above with your actual Space URL if different.)

## Features
- Simple web UI built with Gradio
- Uses `transformers.pipeline("summarization")` for inference
- Ready to deploy on Hugging Face Spaces via the included GitHub Actions workflow

## Quick start (local)
1. Clone the repository:
   ```bash
   git clone https://github.com/Muhammadyousafrana/hugging-face-demo.git
   cd hugging-face-demo

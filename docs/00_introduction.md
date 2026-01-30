# Introduction

**Sentinel-SLM** is a production-ready, multilingual content moderation system designed to protect Large Language Model (LLM) deployments. It acts as a firewall for your LLM, filtering out malicious inputs (Jailbreaks) and harmful outputs (Hate, Violence, etc.).

## Philosophy

Most safety guardrails are either:
1.  **Too Heavy**: Large 7B+ models that add 500ms+ latency.
2.  **Too Simple**: Keyword filters that are easily bypassed.
3.  **English-Only**: Leaving multilingual deployments vulnerable.

Sentinel-SLM bridges this gap by using **Small Language Models (SLMs)**—specifically LiquidAI's LFM2-350M—specialized with distinct "Rails" to achieve high accuracy with low latency (<50ms).

## The Dual-Rail System

The system is composed of two independent models:

*   **Rail A (Input Guard)**: A binary classifier trained to detect prompt injections and jailbreaks. It sits *before* your LLM.
*   **Rail B (Policy Guard)**: A multi-label classifier trained to detect 7 distinct categories of policy violations. It can check user inputs or LLM outputs.

## Key Features

*   **Speed**: Built on 350M parameter models, efficient enough for CPU/Edge inference.
*   **Privacy**: Run entirely locally (no API calls to OpenAI/Anthropic).
*   **Multilingual**: Trained on a diverse mix of 20+ languages.
*   **No False Positives**: Tuned to allow "benign" safety discussions while blocking attacks.

---

[Next: Architecture →](01_architecture.md)

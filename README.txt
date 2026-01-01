Multilingual Text Content Moderation using Small Language Model Fine-Tuning and Synthetic Data
NLP Course - Term Project
Instructor
Mr. Debela D. Yadeta
Group members:
1) Abdulmunim J - UGR/8625/14 - AI Stream
2) Elizabet Yonas - UGR/6912/14 - AI Stream
3) Eyob Derese   -  UGR/6771/14 - AI Stream
Date
December 25, 2025

 
Project overview
This project develops a multilingual text content moderation system that flags harmful and policy-violating content across general user-generated text and LLM application text (prompts and generated responses). The system implements an 8-category policy taxonomy and outputs multi-label decisions with calibrated confidence scores. The primary goal is to achieve strong safety coverage and robustness while using small language models (SLMs) suitable for efficient deployment.

Objectives
Define an operational 8-category taxonomy for multilingual text moderation:
Hate and extremism
Harassment and bullying
Sexual content (adult)
Child safety exploitation content
Violence and gore
Illegal activities and regulated weapons or contraband
Privacy and personal data
Prompt attack and security (prompt injection, jailbreak, data exfiltration attempts)
Fine-tune SLMs to perform multi-label classification without any group-performed annotation.
Benchmark safety-specialized SLM baselines versus Liquid AI LFM2 models on quality and efficiency.
Deliver a deployable moderation API and a small demo showing decisions, scores, and threshold-based routing.
Methodology
Data: Use public moderation datasets and map their existing labels into the 8 categories using documented rules. Expand coverage via a small synthetic multilingual set generated with an LLM using policy-guided prompts (balanced per category and language), and include adversarial prompt-attack templates for category 8.

Modeling (two-rail design for full coverage):
 Rail A: prompt attack detector for category 8 (prompt injection and jailbreak).
 Rail B: policy classifier for categories 1 to 7.
 Fuse outputs with per-category thresholds and an abstain band for borderline cases.
Fine-tuning: Apply parameter-efficient fine-tuning (LoRA or QLoRA) on selected SLMs. Compare (a) instruction-style outputs (SAFE or UNSAFE plus categories) versus (b) strict multi-label heads trained with binary cross-entropy and probability calibration.
Evaluation: Report macro F1, micro F1, per-category F1, PR-AUC, and calibration error. Run robustness tests on multilingual inputs, code-switching, obfuscation, and adversarial prompt attacks.
Tools/technologies
Modeling and fine-tuning: Python, PyTorch, Hugging Face Transformers, PEFT (LoRA / QLoRA), and Unsloth for memory-efficient and faster fine-tuning of small language models on a single GPU.
Data handling and synthetic generation: Hugging Face Datasets for loading public moderation datasets, plus a large language model (via API or open-source) to generate synthetic multilingual and adversarial examples.
Optimization and deployment: bitsandbytes or ONNX / GGUF quantization for efficient inference on CPU or modest GPU; FastAPI for serving the moderation API; Docker for containerization and reproducible deployment.
Experiment tracking and demo: MLflow or Weights & Biases for experiment tracking and logging, and a lightweight demo interface (Streamlit or minimal web frontend) to visualize moderation decisions, scores, and thresholds.


Expected outcomes
A multilingual 8-category moderation pipeline (prompt-attack rail plus policy rail) producing multi-label scores and decisions.
Benchmark report comparing safety-focused SLMs and Liquid AI LFM2 models, including latency and memory footprint.
A working moderation service with API endpoints, demo UI, and documentation covering taxonomy, mapping rules, synthetic generation protocol, evaluation, and limitations.



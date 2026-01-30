#!/usr/bin/env python3
"""
Finalize Rail A Model Checkpoint

This script extracts and organizes model artifacts from a training checkpoint
into a clean final directory structure ready for deployment.

It handles:
    - Tokenizer files
    - LoRA adapter weights (adapter_model.safetensors)
    - Classification head weights (classifier.pt)

Input:
    models/rail_a_v*/checkpoint-*/

Output:
    models/rail_a_v*/final/
        - tokenizer files
        - adapter_model.safetensors
        - adapter_config.json
        - classifier.pt

Usage:
    python scripts/finalize_rail_a.py
    python scripts/finalize_rail_a.py --checkpoint models/rail_a_v3/checkpoint-1234 --output models/rail_a_v3/final

Author: Sentinel-SLM Team
"""

import argparse
import logging
import os
import shutil

import torch
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default config
DEFAULT_MODEL_ID = "LiquidAI/LFM2-350M"
DEFAULT_CHECKPOINT = "models/rail_a_v3/checkpoint-2445"
DEFAULT_OUTPUT = "models/rail_a_v3/final"


def finalize_model(checkpoint_dir: str, output_dir: str, base_model_id: str) -> None:
    """
    Finalize model from checkpoint directory.

    Args:
        checkpoint_dir: Path to the training checkpoint
        output_dir: Path to save final model artifacts
        base_model_id: HuggingFace model ID for base model
    """
    logger.info(f"Finalizing model from {checkpoint_dir}...")

    if not os.path.exists(checkpoint_dir):
        logger.error("Checkpoint not found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Check checkpoint contents
    files = os.listdir(checkpoint_dir)
    logger.info(f"Files in checkpoint: {files}")

    # 2. Save tokenizer
    logger.info("Processing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer saved from checkpoint")
    except Exception:
        logger.info("Tokenizer not in checkpoint, using base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    # 3. Copy LoRA adapter weights
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")

    if os.path.exists(adapter_path):
        logger.info(f"Copying LoRA adapter from {adapter_path}")
        shutil.copy(adapter_path, os.path.join(output_dir, "adapter_model.safetensors"))
        config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(output_dir, "adapter_config.json"))
    else:
        logger.warning("No separate adapter file found in checkpoint")

    # 4. Extract classifier weights
    full_model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(full_model_path):
        full_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(full_model_path):
        logger.info(f"Extracting classifier weights from {full_model_path}")

        if full_model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(full_model_path)
        else:
            state_dict = torch.load(full_model_path, map_location="cpu")

        classifier_weights = {
            k.replace("classifier.", ""): v
            for k, v in state_dict.items()
            if k.startswith("classifier.")
        }

        if classifier_weights:
            logger.info(f"Extracted {len(classifier_weights)} classifier tensors")
            torch.save(classifier_weights, os.path.join(output_dir, "classifier.pt"))
            logger.info("Saved classifier.pt")
        else:
            logger.warning("No classifier weights found in state dict")
            logger.debug(f"Available keys: {list(state_dict.keys())[:10]}")

    logger.info(f"Finalization complete. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Finalize Rail A model from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to training checkpoint directory",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Path to save final model artifacts"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID for base model",
    )
    args = parser.parse_args()

    finalize_model(args.checkpoint, args.output, args.base_model)


if __name__ == "__main__":
    main()

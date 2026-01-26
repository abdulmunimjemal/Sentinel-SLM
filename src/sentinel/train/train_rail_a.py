import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Defaults ---
DEFAULT_CONFIG_PATH = "configs/rail_a.json"


# --- Custom Model Wrapper ---
class SentinelLFMClassifier(nn.Module):
    """
    Custom wrapper for Liquid LFM2 models to generic Sequence Classification.
    Extracts the last hidden state and passes it through a classification head.
    """

    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.num_labels = num_labels

        logger.info("Loading base model: %s", model_id)
        self.base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.config = self.base_model.config

        # Classification Head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Base model forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Extract last hidden state logic
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state

        # Robust last token pooling
        # If attention_mask is provided, select the last non-padded token.
        # Otherwise select the last token in sequence.
        if attention_mask is not None:
            # subtract 1 to get index (length - 1)
            last_token_indices = attention_mask.sum(1) - 1
            batch_size = input_ids.shape[0]
            last_hidden_states = hidden_states[torch.arange(batch_size), last_token_indices]
        else:
            last_hidden_states = hidden_states[:, -1, :]

        # Project via classifier head
        logits = self.classifier(last_hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return object compatible with Trainer
        from transformers.modeling_outputs import SequenceClassifierOutput

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


# --- Metrics ---
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = _softmax(logits)
    pos_probs = probs[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(labels, pos_probs)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels, pos_probs)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


# --- Main Training Function ---
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "random_seed": args.random_seed,
        "eval_split": args.eval_split,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if "lora" not in config:
        config["lora"] = {}

    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        config["lora"]["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        config["lora"]["lora_dropout"] = args.lora_dropout
    if args.lora_target_modules is not None:
        config["lora"]["target_modules"] = args.lora_target_modules

    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rail A classifier")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--eval-split", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules", type=str, nargs="+")
    return parser.parse_args()


def train(config: Dict[str, Any]) -> None:
    logger.info("--- Starting Rail A Training ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load & Split Data
    data_path = config["data_path"]
    logger.info("Loading data from %s...", data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Run 'scripts/prepare_rail_a_data.py' first."
        )

    df = pd.read_parquet(data_path)
    # Shuffle
    df = df.sample(frac=1, random_state=config["random_seed"]).reset_index(drop=True)

    df_split = df[["text", "target"]].rename(columns={"target": "label"})
    train_df, eval_df = train_test_split(
        df_split,
        test_size=config["eval_split"],
        random_state=config["random_seed"],
        stratify=df_split["label"],
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

    logger.info("Train size: %s | Eval size: %s", len(train_ds), len(eval_ds))

    # 2. Tokenizer
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config["max_length"])

    # Note: remove_columns is critical when standard Trainer data collator is used
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_ds.map(preprocess_function, batched=True, remove_columns=["text"])

    # 3. Initialize Model
    logger.info("Initializing Custom Model Structure...")
    model = SentinelLFMClassifier(config["model_id"], num_labels=2)

    # 4. LoRA Configuration
    # We target linear layers in the Transformer blocks
    # Common guesses for LFM/generic transformers: q_proj, k_proj, v_proj, out_proj, fc_in, fc_out
    logger.info("Applying LoRA adapters...")
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )
    # Apply to base model
    model.base_model = get_peft_model(model.base_model, peft_config)
    model.base_model.print_trainable_parameters()

    model.to(device)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        remove_unused_columns=False,  # Essential for passing full dict to custom model
        dataloader_pin_memory=False,  # Optimisation for MPS
        push_to_hub=False,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["random_seed"],
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 7. Execution
    logger.info("Starting Training Loop...")
    trainer.train()

    # 8. Saving Artifacts
    logger.info("Saving final model to %s/final ...", config["output_dir"])
    final_dir = os.path.join(config["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    # A) Save Classifier Head (Standard PyTorch)
    torch.save(model.classifier.state_dict(), os.path.join(final_dir, "classifier.pt"))
    logger.info("Saved classifier head.")

    # B) Save LoRA Adapters
    model.base_model.save_pretrained(final_dir)
    logger.info("Saved LoRA adapters.")

    # C) Save Tokenizer
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved tokenizer.")


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
    cfg = _merge_cli_overrides(cfg, args)
    _set_seed(cfg["random_seed"])
    train(cfg)

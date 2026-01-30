"""
Rail B Training Script - Multi-Label Policy Classification

Trains LFM2-350M with LoRA for 8-category policy violation detection.

Usage:
    python src/sentinel/train/train_rail_b.py --prototype  # Quick test on 10k
    python src/sentinel/train/train_rail_b.py --full       # Full 1.67M dataset
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, hamming_loss
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Configuration
MODEL_ID = "LiquidAI/LFM2-350M"
MAX_LENGTH = 512
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 64
EPOCHS = 2
LR = 1e-4
SEED = 42
NUM_LABELS = 8

# Category names for logging
CATEGORY_NAMES = [
    "Hate & Extremism",
    "Harassment & Bullying",
    "Sexual Content",
    "Child Safety",
    "Violence & Gore",
    "Illegal Activities",
    "Privacy Violations",
    "Prompt Attack",
]


class SentinelLFMMultiLabel(nn.Module):
    """
    Multi-label classifier for Rail B (Policy Core).
    """

    def __init__(self, model_id, num_labels=8):
        super().__init__()
        self.num_labels = num_labels

        print(f"Loading base model: {model_id}")
        self.base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.config = self.base_model.config

        # Multi-label classification head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),  # Higher dropout for multi-label
            nn.Linear(hidden_size, num_labels),
        )

        # BCEWithLogitsLoss (combines sigmoid + BCE)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state

        # Last token pooling
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(1) - 1
            batch_size = input_ids.shape[0]
            last_hidden_states = hidden_states[
                torch.arange(batch_size, device=input_ids.device), last_token_indices
            ]
        else:
            last_hidden_states = hidden_states[:, -1, :]

        logits = self.classifier(last_hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        from transformers.modeling_outputs import SequenceClassifierOutput

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


def compute_metrics(eval_pred):
    """
    Compute multi-label metrics.
    """
    logits, labels = eval_pred

    # Convert logits to predictions (threshold = 0.5)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    # Overall metrics
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    hamming = hamming_loss(labels, predictions)

    # Per-category F1
    f1_per_category = f1_score(labels, predictions, average=None, zero_division=0)

    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "hamming_loss": hamming,
    }

    # Add per-category metrics
    for i, cat_name in enumerate(CATEGORY_NAMES):
        metrics[f"f1_{cat_name.lower().replace(' ', '_').replace('&', 'and')}"] = f1_per_category[i]

    return metrics


def train(data_path_prefix, output_dir, prototype=False):
    """
    Main training function.
    """
    print("=" * 60)
    print("RAIL B TRAINING - MULTI-LABEL POLICY CLASSIFICATION")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading data from {data_path_prefix}...")
    train_df = pd.read_parquet(f"{data_path_prefix}_train.parquet")
    val_df = pd.read_parquet(f"{data_path_prefix}_val.parquet")

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Convert to Dataset
    def prepare_dataset(df):
        # Convert label_vector (numpy array) to list for Dataset
        df["labels"] = df["label_vector"].apply(lambda x: x.tolist())
        return Dataset.from_pandas(df[["text", "labels"]])

    train_ds = prepare_dataset(train_df)
    val_ds = prepare_dataset(val_df)

    # Tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=["text"])

    # Model
    print("Initializing model...")
    model = SentinelLFMMultiLabel(MODEL_ID, num_labels=NUM_LABELS)

    # LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["out_proj", "v_proj", "q_proj", "k_proj"],
        lora_dropout=0.2,
        bias="none",
    )
    model.base_model = get_peft_model(model.base_model, peft_config)
    model.base_model.print_trainable_parameters()

    model.to(device)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,  # MPS doesn't support fp16
        push_to_hub=False,
        warmup_ratio=0.1,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}/final...")
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    torch.save(model.classifier.state_dict(), os.path.join(final_dir, "classifier.pt"))
    model.base_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, "full_model.pt"))

    print("✅ Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", action="store_true", help="Train on 10k prototype")
    parser.add_argument("--full", action="store_true", help="Train on full dataset")
    args = parser.parse_args()

    if args.prototype:
        data_path = "data/processed/rail_b_policy"
        output_dir = "models/rail_b_prototype"
    elif args.full:
        data_path = "data/processed/rail_b_full"
        output_dir = "models/rail_b_v1"
    else:
        print("Specify --prototype or --full")
        return

    train(data_path, output_dir, prototype=args.prototype)


if __name__ == "__main__":
    main()

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Configuration ---
MODEL_ID = "LiquidAI/LFM2-350M"
DATA_PATH = "data/processed/rail_a_clean.parquet"
OUTPUT_DIR = "models/rail_a_v3"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-4
RANDOM_SEED = 42

# --- Custom Model Wrapper ---
class SentinelLFMClassifier(nn.Module):
    """
    Custom wrapper for Liquid LFM2 models to generic Sequence Classification.
    Extracts the last hidden state and passes it through a classification head.
    """
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        
        print(f"Loading base model: {model_id}")
        self.base_model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.config = self.base_model.config
        
        # Classification Head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
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
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Training Function ---
def train():
    print(f"--- Starting Rail A Training ---")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load & Split Data
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Run 'scripts/prepare_rail_a_data.py' first.")

    df = pd.read_parquet(DATA_PATH)
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    ds = Dataset.from_pandas(df[['text', 'target']])
    ds = ds.rename_column("target", "label")
    
    split_ds = ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    train_ds = split_ds['train']
    eval_ds = split_ds['test']
    
    print(f"Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")

    # 2. Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    # Note: remove_columns is critical when standard Trainer data collator is used
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    
    # 3. Initialize Model
    print("Initializing Custom Model Structure...")
    model = SentinelLFMClassifier(MODEL_ID, num_labels=2)
    
    # 4. LoRA Configuration
    # We target linear layers in the Transformer blocks
    # Common guesses for LFM/generic transformers: q_proj, k_proj, v_proj, out_proj, fc_in, fc_out
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["out_proj", "v_proj", "q_proj", "k_proj"], 
        lora_dropout=0.1, 
        bias="none"
    )
    # Apply to base model
    model.base_model = get_peft_model(model.base_model, peft_config)
    model.base_model.print_trainable_parameters()
    
    model.to(device)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",      # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        remove_unused_columns=False, # Essential for passing full dict to custom model
        dataloader_pin_memory=False, # Optimisation for MPS
        push_to_hub=False
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
    print("Starting Training Loop...")
    trainer.train()
    
    # 8. Saving Artifacts
    print(f"Saving final model to {OUTPUT_DIR}/final ...")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # A) Save Classifier Head (Standard PyTorch)
    torch.save(model.classifier.state_dict(), os.path.join(final_dir, "classifier.pt"))
    print("Saved classifier head.")
    
    # B) Save LoRA Adapters
    model.base_model.save_pretrained(final_dir)
    print("Saved LoRA adapters.")
    
    # C) Save Tokenizer
    tokenizer.save_pretrained(final_dir)
    print("Saved tokenizer.")

if __name__ == "__main__":
    train()

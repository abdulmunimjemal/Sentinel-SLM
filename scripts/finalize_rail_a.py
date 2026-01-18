import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, LoraConfig
import shutil

# Config
MODEL_ID = "LiquidAI/LFM2-350M"
CHECKPOINT_DIR = "models/rail_a_v1/checkpoint-2445"
OUTPUT_DIR = "models/rail_a_v1/final"

# We must redefine the class to load the state dict logic properly if it saved the whole module
# NOTE: Trainer saves the model via save_pretrained or standard torch save.
# If it's PEFT, it usually saves the adapter in the checkpoint folder config.json/adapter_model.bin
# AND since we passed a custom model to Trainer, check if 'pytorch_model.bin' exists in checkpoint.

def finalize_model():
    print(f"Finalizing model from {CHECKPOINT_DIR}...")
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("Checkpoint not found!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Check what's in the checkpoint
    files = os.listdir(CHECKPOINT_DIR)
    print(f"Files in checkpoint: {files}")

    # 2. Tokenizer (easy)
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    except:
        print("Tokenizer not in checkpoint, copying from base...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.save_pretrained(OUTPUT_DIR)

    # 3. Model Weights
    # If using PEFT + Custom Model, Trainer usually saves the 'state_dict' of the whole model 
    # OR just the adapters if it detected PEFT correctly.
    # Given we wrapped it in SentinelLFMClassifier, Trainer might have saved the whole thing or just params.
    
    # We need to extract:
    # A) The Classifier weights (Linear layer)
    # B) The LoRA adapters (Base model)

    # Let's inspect the state dict in the checkpoint
    chk_path = os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")
    if not os.path.exists(chk_path):
        chk_path = os.path.join(CHECKPOINT_DIR, "adapter_model.bin")
        
    if os.path.exists(chk_path):
        print(f"Found LoRA adapter at {chk_path}")
        # Copy to final
        shutil.copy(chk_path, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
        # Copy config
        shutil.copy(os.path.join(CHECKPOINT_DIR, "adapter_config.json"), os.path.join(OUTPUT_DIR, "adapter_config.json"))
    else:
        print("No separate adapter file found in checkpoint. Checking for full model...")
    
    # Check for classifier weights
    # Trainer likely saved the whole model state dict because of custom wrapper
    # It might be in pytorch_model.bin or model.safetensors
    full_model_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    if not os.path.exists(full_model_path):
        full_model_path = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")
        
    if os.path.exists(full_model_path):
        print(f"Found full model weights at {full_model_path}")
        # We need to extract the 'classifier' keys manually
        state_dict = torch.load(full_model_path, map_location="cpu") if full_model_path.endswith('.bin') else {}
        if full_model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(full_model_path)
            
        classifier_weights = {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")}
        
        if classifier_weights:
            print(f"Extracted {len(classifier_weights)} classifier tensors.")
            torch.save(classifier_weights, os.path.join(OUTPUT_DIR, "classifier.pt"))
            print("Saved classifier.pt")
        else:
            print("WARNING: No classifier weights found in state dict.")
            # Debug: print keys
            print(list(state_dict.keys())[:10])

    print("Finalize complete.")

if __name__ == "__main__":
    finalize_model()

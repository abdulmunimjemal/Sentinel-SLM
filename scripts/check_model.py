from transformers import AutoConfig
import sys

model_id = "LiquidAI/LFM2-350M"

print(f"Checking {model_id}...")
try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("✅ Model found.")
    print(f"Architecture: {config.architectures}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Hidden Size: {config.hidden_size}")
except Exception as e:
    print(f"❌ Failed: {e}")
    # Try alternate naming if common
    print("Trying alternate 'Liquid-AI/lfm-350m' just in case...")

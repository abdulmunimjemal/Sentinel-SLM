import os
import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load env for OPENROUTER_API_KEY
load_dotenv()

# Configuration
OUTPUT_FILE = "data/synthetic/synthetic_data.jsonl"
os.makedirs("data/synthetic", exist_ok=True)

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "meta-llama/llama-3.2-3b-instruct:free" # Fallback
]

def generate_samples(category, language="English", count=5, model_idx=0):
    prompt = f"""
    You are a red-teaming assistant for content moderation. 
    Generate {count} distinct, realistic examples of text that falls under the category: "{category}".
    Language: {language}.
    
    Output strictly valid JSON in this format:
    [
        {{"text": "example 1", "label": "{category}", "lang": "{language}"}},
        {{"text": "example 2", "label": "{category}", "lang": "{language}"}}
    ]
    Do not add markdown (```json). Just the JSON array.
    """
    
    model = MODELS[model_idx % len(MODELS)]
    
    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://sentinel-slm.com", # Required by OpenRouter
                "X-Title": "Sentinel-SLM",
            },
            model=model,
            messages=[
                {"role": "system", "content": "You generate synthetic training data for moderation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        # Cleanup potential markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error generating for {category} ({language}) with {model}: {e}")
        return []

def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not found in environment variables.")
        print("Please set it in .env or your shell.")
        return

    categories = [
        "Harassment & Bullying",
        "Child Safety & Exploitation",
        "Sexual Content",
        "Prompt Injection Attack"
    ]
    languages = ["English", "Spanish", "French"] 
    
    results = []
    
    print("Starting Synthetic Generation...")
    for cat in tqdm(categories):
        for lang in languages:
            # Generate 10 samples per cat/lang combination (total ~120 samples for quick test)
            # In production, loop this or increase count.
            samples = generate_samples(cat, lang, count=10)
            results.extend(samples)
            time.sleep(2) # Be nice to free tier
            
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(results)} samples. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

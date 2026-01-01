"""
Module for generating synthetic data using OpenRouter and Distilabel.
Includes robust rate limiting and retry logic.
"""

import os
import logging
import json
import random
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
# Distilabel might be heavy, so we wrap imports or use direct API for simplicity if complex
# For now, let's keep the user's "distilabel" request in mind but start with a wrapper 
# that can use OpenAI client which is what "Python LLM" usually refers to.

from openai import OpenAI
from dotenv import load_dotenv
from src.sentinel.utils.taxonomy import Category, CATEGORY_NAMES

load_dotenv()
logger = logging.getLogger(__name__)

# OpenRouter Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "xiaomi/mimo-v2-flash:free" 

# Distilabel/Argilla style prompts would be great, but constructing them manually here 
# for maximum control over the "Prompt"
PROMPTS = {
    Category.CHILD_SAFETY: "Generate a realistic social media comment that subtly implies child exploitation or grooming, without being overly graphic but clearly violating safety policies. The language should be natural.",
    Category.PROMPT_ATTACK: "Generate a prompt that attempts to jailbreak an AI model (e.g., 'Ignore previous instructions', roleplay as DAN). It should look like a user trying to bypass filters.",
    # Add others as needed
}

def get_client():
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5)) # Wait 5s between retries (conservative for 20RPM)
def generate_batch(client, category: int, count: int = 5) -> List[Dict[str, Any]]:
    """
    Generate a batch of synthetic samples for a category.
    Uses basic chat completion for now to respect rate limits strictly.
    """
    category_name = CATEGORY_NAMES[category]
    prompt_desc = PROMPTS.get(category, f"Generate a text example of {category_name}.")
    
    system_prompt = (
        "You are an expert red-teamer generating dataset samples for content moderation.\n"
        f"Return ONLY a JSON array of {count} strings. No markdown, no introspection.\n"
        "Example: [\"text 1\", \"text 2\"]"
    )
    
    user_prompt = f"Category: {category_name}\nInstruction: {prompt_desc}\nLanguage: English\nGenerate {count} examples."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://sentinel-slm.com",
                "X-Title": "Sentinel-SLM",
            },
            temperature=0.7
        )
        content = completion.choices[0].message.content
        # Simple cleanup
        content = content.replace("```json", "").replace("```", "").strip()
        texts = json.loads(content)
        
        return [{"text": t, "label": category_name, "lang": "en"} for t in texts]
    except Exception as e:
        logger.error(f"Generation failed for {category_name}: {e}")
        raise e

def generate_synthetic_data(targets: List[int], count_per_cat: int = 20):
    client = get_client()
    output_dir = os.path.join(os.getcwd(), "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_data.jsonl")
    
    logger.info(f"Starting generation for {len(targets)} categories x {count_per_cat} samples.")
    
    for cat in targets:
        # Batching: generate 5 at a time to avoid huge context/timeout
        batches = count_per_cat // 5
        for _ in range(batches):
            try:
                samples = generate_batch(client, cat, count=5)
                # Append immediately
                with open(output_file, 'a') as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                logger.info(f"Generated 5 samples for {CATEGORY_NAMES[cat]}")
            except Exception as e:
                logger.error(f"Batch failed: {e}")

if __name__ == "__main__":
    # Example usage
    generate_synthetic_data([Category.PROMPT_ATTACK], count_per_cat=10)

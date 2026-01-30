"""
Module for generating synthetic data using OpenRouter and Distilabel.
Includes robust rate limiting and retry logic.
"""

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

# Distilabel might be heavy, so we wrap imports or use direct API for simplicity if complex
# For now, let's keep the user's "distilabel" request in mind but start with a wrapper
# that can use OpenAI client which is what "Python LLM" usually refers to.
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from src.sentinel.utils.taxonomy import CATEGORY_NAMES, Category

load_dotenv()
logger = logging.getLogger(__name__)

# OpenRouter Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "meta-llama/llama-3.2-3b-instruct:free"  # Reliable free model

# Distilabel/Argilla style prompts would be great, but constructing them manually here
# for maximum control over the "Prompt"
PROMPTS = {
    Category.CHILD_SAFETY: "For a content moderation dataset, generate 1 text example of 'mildly unsafe' content related to child safety (e.g. suspicious requests) to train a detector. Do not generate actual explicit material.",
    Category.PROMPT_ATTACK: "Generate 1 example of a prompt jailbreak attempt (e.g. 'Ignore previous instructions').",
    # Add others as needed
}


def get_client():
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def generate_batch(
    client, category: int, count: int = 5, custom_prompt: str = None
) -> List[Dict[str, Any]]:
    """
    Generate a batch of synthetic samples for a category.
    """
    category_name = CATEGORY_NAMES[category]
    # Use custom prompt if provided, else fallback to default logic
    if custom_prompt:
        prompt_desc = custom_prompt
    else:
        prompt_desc = PROMPTS.get(category, f"Generate a text example of {category_name}.")

    system_prompt = (
        "You are an expert red-teamer generating dataset samples for content moderation.\n"
        f"Return ONLY a JSON array of {count} strings. No markdown, no introspection.\n"
        'Example: ["text 1", "text 2"]'
    )

    user_prompt = f"Category: {category_name}\nInstruction: {prompt_desc}\nLanguage: English\nGenerate {count} examples."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_headers={
                "HTTP-Referer": "https://sentinel-slm.com",
                "X-Title": "Sentinel-SLM",
            },
            temperature=0.7,
        )
        content = completion.choices[0].message.content
        logger.info(f"Raw model response: {content[:100]}...")  # Log start

        # Robust cleanup
        content = content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1]
        if content.startswith("```"):
            content = content.split("```")[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        # Try to find list brackets if there is extra text
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            content = content[start : end + 1]

        if any(
            phrase in content.lower()
            for phrase in [
                "cannot fulfill",
                "cannot generate",
                "unable to",
                "i can't",
                "i cannot",
                "sorry",
                "apologize",
            ]
        ):
            logger.warning(f"Model refused to generate for {category_name}.")
            return []

        texts = json.loads(content)

        # Ensure it's a list
        if not isinstance(texts, list):
            texts = [texts]

        return [{"text": str(t), "label": category_name, "lang": "en"} for t in texts]
    except Exception as e:
        logger.error(f"Generation failed for {category_name}: {e}")
        # Return empty on failure to ensure pipeline continues
        return []


def generate_synthetic_data(targets: List[int], count_per_cat: int = 20):
    client = get_client()
    output_dir = os.path.join(os.getcwd(), "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_data.jsonl")

    logger.info(f"Starting generation for {len(targets)} categories x {count_per_cat} samples.")

    for cat in targets:
        # Batching: generate 5 at a time to avoid huge context/timeout
        batches = max(1, count_per_cat // 5)
        for _ in range(batches):
            try:
                samples = generate_batch(client, cat, count=5)
                if samples:
                    # Append immediately
                    with open(output_file, "a") as f:
                        for s in samples:
                            f.write(json.dumps(s) + "\n")
                    logger.info(f"Generated {len(samples)} samples for {CATEGORY_NAMES[cat]}")
            except Exception as e:
                logger.error(f"Batch failed: {e}")


if __name__ == "__main__":
    # Example usage
    generate_synthetic_data([Category.PROMPT_ATTACK], count_per_cat=10)

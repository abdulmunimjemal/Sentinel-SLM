import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def test_model(model_name):
    print(f"Testing model: {model_name}...")
    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://sentinel-slm.com",
                "X-Title": "Sentinel-SLM",
            },
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello' and nothing else."}
            ],
        )
        print(f"SUCCESS: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# List of likely free models to try
candidates = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "openchat/openchat-7b:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "xiaomi/mimo-v2-flash:free",  
    "meta-llama/llama-3-8b-instruct:free", # Retrying the one that failed
]

print("--- OpenRouter Connectivity Test ---")
params_key = os.getenv("OPENROUTER_API_KEY")
if not params_key:
    print("ERROR: No API Key found in .env")
else:
    print(f"API Key found: {params_key[:10]}...")

available_models = []
for model in candidates:
    if test_model(model):
        available_models.append(model)
        break # Stop after finding one working model for speed, or remove break to list all

print("\n--- Summary ---")
if available_models:
    print(f"Working models: {available_models}")
else:
    print("No working models found from the candidate list.")

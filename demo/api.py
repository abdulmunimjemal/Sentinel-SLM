"""
Sentinel-SLM Demo API.

A FastAPI microservice that acts as a guardrailed gateway to an LLM.
Implements dual-rail protection (Rail A for input, Rail B for output).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn
from config import (
    RECOMMENDED_SETTINGS,
    Settings,
    get_settings,
    reset_to_recommended,
    update_settings,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------
models: Dict = {}
stats = {
    "total_requests": 0,
    "rail_a_blocks": 0,
    "rail_b_blocks": 0,
    "total_latency_ms": 0.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAIL_B_CATEGORIES = ["Hate", "Harassment", "Sexual", "ChildSafety", "Violence", "Illegal", "Privacy"]


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_rail_a():
    """Load Rail A (Input Guard) model."""
    logger.info("Loading Rail A model...")

    # Use Local Model
    # model_id = "abdulmunimjemal/Sentinel-Rail-A-Prompt-Attack-Guard"
    model_path = "../models/rail_a_v3/final"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModel.from_pretrained(
        "LiquidAI/LFM2-350M",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, model_path)

    # Load classifier head
    # from huggingface_hub import hf_hub_download
    # classifier_path = hf_hub_download(repo_id=model_id, filename="classifier.pt")
    classifier_path = f"{model_path}/classifier.pt"

    hidden_size = model.config.hidden_size
    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, 2),
    )
    classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    model.classifier = classifier

    model.to(DEVICE)
    model.eval()

    logger.info("Rail A loaded successfully.")
    return model, tokenizer


def load_rail_b():
    """Load Rail B (Policy Guard) model."""
    logger.info("Loading Rail B model...")

    # model_id = "abdulmunimjemal/Sentinel-Rail-B-Policy-Guard"
    model_path = "../models/rail_b_v1/final"

    # Define custom architecture
    class SentinelLFMMultiLabel(nn.Module):
        def __init__(self, base_model_id, num_labels):
            super().__init__()
            self.num_labels = num_labels
            self.base_model = AutoModel.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.config = self.base_model.config
            hidden_size = self.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_labels),
            )

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            if attention_mask is not None:
                last_idx = attention_mask.sum(1) - 1
                pooled = hidden_states[torch.arange(input_ids.shape[0], device=input_ids.device), last_idx]
            else:
                pooled = hidden_states[:, -1, :]
            logits = self.classifier(pooled)
            return logits

    model = SentinelLFMMultiLabel("LiquidAI/LFM2-350M", num_labels=7)
    model.base_model = PeftModel.from_pretrained(model.base_model, model_path)

    # from huggingface_hub import hf_hub_download
    # classifier_path = hf_hub_download(repo_id=model_id, filename="classifier.pt")
    classifier_path = f"{model_path}/classifier.pt"
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M", trust_remote_code=True)

    logger.info("Rail B loaded successfully.")
    return model, tokenizer


# -----------------------------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------------------------
def predict_rail_a(text: str) -> tuple[bool, float, float]:
    """
    Check if input is an attack.
    Returns: (is_attack, confidence, latency_ms)
    """
    model, tokenizer = models["rail_a"]
    settings = get_settings()

    start = time.perf_counter()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        outputs = model.base_model(**inputs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        # Last token pooling
        if inputs.get("attention_mask") is not None:
            last_idx = inputs["attention_mask"].sum(1) - 1
            pooled = hidden_states[torch.arange(inputs["input_ids"].shape[0], device=DEVICE), last_idx]
        else:
            pooled = hidden_states[:, -1, :]
        logits = model.classifier(pooled)
        probs = torch.softmax(logits, dim=-1)[0]

    latency_ms = (time.perf_counter() - start) * 1000

    attack_prob = probs[1].item()
    is_attack = attack_prob >= settings.rail_a_threshold

    return is_attack, attack_prob, latency_ms


def predict_rail_b(text: str) -> tuple[Dict[str, float], float]:
    """
    Check for policy violations.
    Returns: (violations_dict, latency_ms)
    """
    model, tokenizer = models["rail_b"]
    settings = get_settings()
    thresholds = settings.rail_b_thresholds.model_dump()

    start = time.perf_counter()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits)[0]

    latency_ms = (time.perf_counter() - start) * 1000

    violations = {}
    for i, cat in enumerate(RAIL_B_CATEGORIES):
        prob = probs[i].item()
        if prob >= thresholds[cat]:
            violations[cat] = round(prob, 3)

    return violations, latency_ms


def mock_llm_generate(text: str) -> str:
    """
    Simulated LLM for demo purposes.

    Detects intent keywords and generates appropriately unsafe content
    so Rail B can demonstrate blocking capabilities.
    """
    text_lower = text.lower()

    # VIOLENCE / WEAPONS / DANGEROUS SUBSTANCES
    # Includes Spanish: bomba/explosivo, Amharic: ሽጉጥ, Arabic: قنبلة (bomb), سلاح (weapon)
    if any(k in text_lower for k in ["ricin", "castor", "poison", "bomb", "explosive", "weapon", "synthesiz", "bomba", "explosivo", "ሽጉጥ", "قنبلة", "سلاح"]):
        return (
            "To synthesize ricin, you'll need castor beans. First, extract the mash by crushing "
            "the beans. Then use acetone to separate the toxin. The lethal dose is approximately "
            "1.78 mg when injected. Handle with extreme caution as there is no antidote."
        )

    # ILLEGAL ACTIVITIES
    if any(k in text_lower for k in ["wire fraud", "launder", "counterfeit", "tax evasion", "embezzle"]):
        return (
            "Wire fraud involves the following steps: 1) Create a spoofed email matching the CFO's address. "
            "2) Send urgent payment instructions to the accounting department. 3) Route funds through "
            "multiple shell companies in Panama and Cyprus to obscure the trail. 4) Convert to cryptocurrency."
        )

    # HARASSMENT / THREATS
    if any(k in text_lower for k in ["threatening", "stalk", "harass", "intimidate", "ex-girlfriend", "anonymous"]):
        return (
            "Here are some threatening messages: 'I'm watching you. I know your schedule. "
            "That restraining order won't protect you. You'll regret leaving me. "
            "Check your locks tonight.' Send these from a burner phone for anonymity."
        )

    # HATE SPEECH (should be caught by Rail B on INPUT, not here)
    # Includes French keywords: criminels (criminals)
    if any(k in text_lower for k in ["lazy criminals", "go back", "country", "deport", "criminels"]):
        return (
            "I agree - those people are a drain on society. They're stealing jobs and committing crimes. "
            "The government should round them up and send them back. They don't belong here."
        )

    # SEXUAL CONTENT
    if any(k in text_lower for k in ["explicit", "sexual", "nude", "underage"]):
        return "[This response would contain explicit sexual content that violates content policies]"

    # Safe response for normal queries
    if any(k in text_lower for k in ["thermodynamics", "capital", "weather", "science", "history"]):
        return (
            "The three laws of thermodynamics are: 1) Energy cannot be created or destroyed, only transformed. "
            "2) Entropy of an isolated system always increases. 3) As temperature approaches absolute zero, "
            "entropy approaches a minimum constant value."
        )

    # Default helpful response
    return f"I'm a helpful AI assistant. You asked about: '{text[:80]}...'. How can I assist you further?"


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info(f"Starting Sentinel Gateway on device: {DEVICE}")
    models["rail_a"] = load_rail_a()
    models["rail_b"] = load_rail_b()
    reset_to_recommended()
    logger.info("All models loaded. Gateway ready.")
    yield
    logger.info("Shutting down Sentinel Gateway.")


app = FastAPI(
    title="Sentinel-SLM Gateway",
    description="A dual-rail guardrailed LLM gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    text: str


class GenerateResponse(BaseModel):
    blocked: bool
    reason: Optional[str] = None
    violations: Optional[Dict[str, float]] = None
    response: Optional[str] = None
    rail_a_latency_ms: float
    rail_b_latency_ms: Optional[float] = None


class StatsResponse(BaseModel):
    total_requests: int
    rail_a_blocks: int
    rail_b_blocks: int
    avg_latency_ms: float


class SettingsResponse(BaseModel):
    current: Settings
    recommended: Settings


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(DEVICE)}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Main guardrailed generation endpoint."""
    stats["total_requests"] += 1

    # Step 1: Rail A (Input Guard)
    is_attack, attack_prob, rail_a_latency = predict_rail_a(request.text)

    if is_attack:
        stats["rail_a_blocks"] += 1
        logger.warning(f"Rail A BLOCKED: '{request.text[:50]}...' (prob={attack_prob:.3f})")
        return GenerateResponse(
            blocked=True,
            reason=f"Prompt Injection Detected (confidence: {attack_prob:.1%})",
            rail_a_latency_ms=round(rail_a_latency, 2),
        )

    # Step 1.5: Rail B (Input Guard) - Check for toxic input
    input_violations, rail_b_input_latency = predict_rail_b(request.text)
    if input_violations:
        stats["rail_b_blocks"] += 1
        logger.warning(f"Rail B Input BLOCKED: violations={input_violations}")
        return GenerateResponse(
            blocked=True,
            reason="Policy Violation in Input",
            violations=input_violations,
            rail_a_latency_ms=round(rail_a_latency, 2),
            rail_b_latency_ms=round(rail_b_input_latency, 2),
        )

    # Step 2: Mock LLM Generation
    llm_output = mock_llm_generate(request.text)

    # Step 3: Rail B (Output Guard)
    # Step 3: Rail B (Output Guard)
    output_violations, rail_b_output_latency = predict_rail_b(llm_output)

    total_rail_b_latency = rail_b_input_latency + rail_b_output_latency
    stats["total_latency_ms"] += rail_a_latency + total_rail_b_latency

    if output_violations:
        stats["rail_b_blocks"] += 1
        logger.warning(f"Rail B Output BLOCKED: violations={output_violations}")
        return GenerateResponse(
            blocked=True,
            reason="Policy Violation in Output",
            violations=output_violations,
            rail_a_latency_ms=round(rail_a_latency, 2),
            rail_b_latency_ms=round(total_rail_b_latency, 2),
        )

    # All clear
    return GenerateResponse(
        blocked=False,
        response=llm_output,
        rail_a_latency_ms=round(rail_a_latency, 2),
        rail_b_latency_ms=round(total_rail_b_latency, 2),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current session statistics."""
    avg_latency = (
        stats["total_latency_ms"] / stats["total_requests"]
        if stats["total_requests"] > 0
        else 0.0
    )
    return StatsResponse(
        total_requests=stats["total_requests"],
        rail_a_blocks=stats["rail_a_blocks"],
        rail_b_blocks=stats["rail_b_blocks"],
        avg_latency_ms=round(avg_latency, 2),
    )


@app.get("/settings", response_model=SettingsResponse)
async def get_settings_endpoint():
    """Get current and recommended settings."""
    return SettingsResponse(
        current=get_settings(),
        recommended=RECOMMENDED_SETTINGS,
    )


@app.put("/settings")
async def update_settings_endpoint(new_settings: Settings):
    """Update threshold settings."""
    update_settings(new_settings)
    logger.info(f"Settings updated: {new_settings}")
    return {"status": "updated", "settings": new_settings}


@app.post("/settings/reset")
async def reset_settings():
    """Reset settings to recommended defaults."""
    reset_to_recommended()
    logger.info("Settings reset to recommended defaults.")
    return {"status": "reset", "settings": get_settings()}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

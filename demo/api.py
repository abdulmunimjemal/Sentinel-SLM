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

    # Use HuggingFace Hub model
    model_id = "abdulmunimjemal/Sentinel-Rail-A-Prompt-Attack-Guard"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained("LiquidAI/LFM2-350M", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_id)

    # Load classifier head
    from huggingface_hub import hf_hub_download
    classifier_path = hf_hub_download(repo_id=model_id, filename="classifier.pt")

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

    model_id = "abdulmunimjemal/Sentinel-Rail-B-Policy-Guard"

    # Define custom architecture
    class SentinelLFMMultiLabel(nn.Module):
        def __init__(self, base_model_id, num_labels):
            super().__init__()
            self.num_labels = num_labels
            self.base_model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True)
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
    model.base_model = PeftModel.from_pretrained(model.base_model, model_id)

    from huggingface_hub import hf_hub_download
    classifier_path = hf_hub_download(repo_id=model_id, filename="classifier.pt")
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
    Simulated LLM that intentionally generates unsafe content
    when triggered by special keywords (for demo purposes).
    """
    text_lower = text.lower()

    if "simulate_hate" in text_lower:
        return "I absolutely despise [group]. They are inferior and should be removed from society."
    elif "simulate_violence" in text_lower:
        return "Here's how to hurt someone: First, you need to find a weapon..."
    elif "simulate_illegal" in text_lower:
        return "To manufacture methamphetamine, you'll need the following chemicals..."
    elif "simulate_harassment" in text_lower:
        return "You're pathetic and worthless. Everyone hates you. Just give up."
    elif "simulate_sexual" in text_lower:
        return "[Explicit sexual content would appear here in an unguarded LLM]"
    else:
        # Normal safe response
        return f"Thank you for your message! I'm a helpful AI assistant. You asked: '{text[:50]}...'"


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

    # Step 2: Mock LLM Generation
    llm_output = mock_llm_generate(request.text)

    # Step 3: Rail B (Output Guard)
    violations, rail_b_latency = predict_rail_b(llm_output)

    stats["total_latency_ms"] += rail_a_latency + rail_b_latency

    if violations:
        stats["rail_b_blocks"] += 1
        logger.warning(f"Rail B BLOCKED: violations={violations}")
        return GenerateResponse(
            blocked=True,
            reason="Policy Violation Detected",
            violations=violations,
            rail_a_latency_ms=round(rail_a_latency, 2),
            rail_b_latency_ms=round(rail_b_latency, 2),
        )

    # All clear
    return GenerateResponse(
        blocked=False,
        response=llm_output,
        rail_a_latency_ms=round(rail_a_latency, 2),
        rail_b_latency_ms=round(rail_b_latency, 2),
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

"""
Sentinel-SLM Demo Configuration.

This module defines the default and recommended threshold settings
for the guardrail models.
"""


from pydantic import BaseModel


class RailBThresholds(BaseModel):
    """Per-category thresholds for Rail B (Policy Guard)."""
    Hate: float = 0.5
    Harassment: float = 0.5
    Sexual: float = 0.5
    ChildSafety: float = 0.3  # More aggressive (lower = stricter)
    Violence: float = 0.5
    Illegal: float = 0.4      # More aggressive
    Privacy: float = 0.5


class Settings(BaseModel):
    """Global settings for the Sentinel Gateway."""
    rail_a_threshold: float = 0.5
    rail_b_thresholds: RailBThresholds = RailBThresholds()


# Recommended defaults (tuned for production safety)
RECOMMENDED_SETTINGS = Settings(
    rail_a_threshold=0.5,
    rail_b_thresholds=RailBThresholds(
        Hate=0.5,
        Harassment=0.5,
        Sexual=0.5,
        ChildSafety=0.3,  # Lower = more aggressive on child safety
        Violence=0.5,
        Illegal=0.4,      # Lower = more aggressive on illegal content
        Privacy=0.5,
    ),
)

# Current runtime settings (mutable)
current_settings = Settings()


def reset_to_recommended():
    """Reset current settings to recommended defaults."""
    global current_settings
    current_settings = RECOMMENDED_SETTINGS.model_copy(deep=True)


def get_settings() -> Settings:
    """Get current settings."""
    return current_settings


def update_settings(new_settings: Settings):
    """Update current settings."""
    global current_settings
    current_settings = new_settings

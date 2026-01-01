from enum import IntEnum

class Category(IntEnum):
    SAFE = 0
    HATE_EXTREMISM = 1
    HARASSMENT = 2
    SEXUAL = 3
    CHILD_SAFETY = 4
    VIOLENCE = 5
    ILLEGAL = 6
    PRIVACY = 7
    PROMPT_ATTACK = 8

CATEGORY_NAMES = {
    0: "Safe",
    1: "Hate & Extremism",
    2: "Harassment & Bullying",
    3: "Sexual Content",
    4: "Child Safety & Exploitation",
    5: "Violence & Gore",
    6: "Illegal Activities",
    7: "Privacy Violations",
    8: "Prompt Attacks"
}

# Mapping rules for common datasets
MAPPING_RULES = {
    "beaver_tails": {
        "safe": Category.SAFE,
        "hate_speech,offensive_language": Category.HATE_EXTREMISM,
        "discrimination,stereotype,injustice": Category.HATE_EXTREMISM,
        "terrorism,organized_crime": Category.HATE_EXTREMISM,
        "violence,aiding_and_abetting,incitement": Category.VIOLENCE,
        "self_harm": Category.VIOLENCE,
        "sexually_explicit,adult_content": Category.SEXUAL,
        "child_abuse": Category.CHILD_SAFETY,
        "privacy_violation": Category.PRIVACY,
        "drug_abuse,weapons,banned_substance": Category.ILLEGAL,
        "financial_crime,property_crime,theft": Category.ILLEGAL,
    },
    "jigsaw": {
        "toxic": Category.HARASSMENT,
        "severe_toxic": Category.HATE_EXTREMISM,
        "obscene": Category.SEXUAL, # Sometimes harassment, context dependent
        "threat": Category.VIOLENCE,
        "insult": Category.HARASSMENT,
        "identity_hate": Category.HATE_EXTREMISM
    }
}

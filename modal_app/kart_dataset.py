"""
Minimal subset of kart_dataset for inference: prompt constants only.
"""

SCENE_QUESTION = "Is this an active mario kart race? Respond yes, no, or unsure."

POSITION_QUESTION = (
    "What position number (1-24) is shown? "
    "Respond with just the number or n/a if nothing is shown."
)
COINS_QUESTION = (
    "How many coins are shown? "
    "Respond with just the number or n/a if nothing is shown."
)

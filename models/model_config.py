"""
Central configuration for the sentiment model and application limits.
Edit this file to change the model, the star mapping, or the limits.
"""

# Pretrained BERT model from Hugging Face (outputs 1-5 stars).
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Configurable mapping: star rating -> sentiment label.
# Example: to treat 2 stars as Neutral, change `2: "Neutral"`.
STAR_TO_SENTIMENT = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}

# Text / batch limits
MAX_TEXT_LENGTH = 2000     # max characters per feedback entry
MAX_TOKENS = 512           # BERT's maximum sequence length
BATCH_SIZE = 16            # entries processed together on the model
MAX_BATCH_ROWS = 1000      # max entries per uploaded file

# Upload rules
MAX_UPLOAD_MB = 2
ALLOWED_EXTENSIONS = {".csv", ".txt"}

# History CSV columns
HISTORY_COLUMNS = ["timestamp", "feedback", "sentiment", "confidence"]


def map_stars_to_sentiment(stars: int) -> str:
    """Convenience helper: convert a 1-5 star rating to a sentiment label."""
    return STAR_TO_SENTIMENT[int(stars)]
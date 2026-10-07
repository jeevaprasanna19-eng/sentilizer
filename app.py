"""
NLP Sentiment Analyzer - Flask backend.

Loads a pretrained BERT sentiment model ONCE at startup and exposes:
  GET  /               main page
  POST /predict        analyze one feedback text
  POST /upload         analyze a CSV/TXT file
  GET  /history        history (newest first) + statistics
  POST /clear-history  wipe history
"""

import csv
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from models import model_config as cfg

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
HISTORY_FILE = BASE_DIR / "data" / "feedback_history.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentiment-app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024

history_lock = threading.Lock()   # protects the CSV file from concurrent writes
model_lock = threading.Lock()     # keeps inference thread-safe


class FeedbackFileError(Exception):
    """Raised for user-fixable problems with an uploaded file."""


# --------------------------------------------------------------------------
# Model loading (runs once, when the server starts)
# --------------------------------------------------------------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using GPU (CUDA): %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA not available - using CPU")
    logger.info("Loading BERT model '%s' (first run downloads ~700 MB)...", cfg.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME)
    model.to(device)
    model.eval()  # inference mode (disables dropout)
    logger.info("Model loaded successfully.")
    return tokenizer, model, device


try:
    tokenizer, model, device = load_model()
except Exception:  # noqa: BLE001 - we log the details but never show them to users
    logger.exception("Model loading failed")
    tokenizer = model = device = None


# --------------------------------------------------------------------------
# Sentiment analysis
# --------------------------------------------------------------------------
def build_result(text, probs):
    """Convert 5 star probabilities into a Positive/Negative/Neutral result."""
    group_probs = {label: 0.0 for label in set(cfg.STAR_TO_SENTIMENT.values())}
    for index, prob in enumerate(probs):
        group_probs[cfg.STAR_TO_SENTIMENT[index + 1]] += prob
    sentiment = max(group_probs, key=group_probs.get)
    expected_stars = sum((i + 1) * p for i, p in enumerate(probs))
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(group_probs[sentiment], 4),
        "score": round(expected_stars, 2),  # weighted average star rating (1-5)
    }


def analyze_texts(texts):
    """Run BERT on a list of texts in efficient mini-batches."""
    results = []
    for start in range(0, len(texts), cfg.BATCH_SIZE):
        batch = texts[start:start + cfg.BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=cfg.MAX_TOKENS, return_tensors="pt",
        ).to(device)
        with model_lock, torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        results.extend(build_result(t, p) for t, p in zip(batch, probabilities))
    return results


# --------------------------------------------------------------------------
# History (CSV) helpers
# --------------------------------------------------------------------------
def ensure_history_file():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cfg.HISTORY_COLUMNS)


def csv_safe(value):
    """Prevent spreadsheet formula injection when the CSV is opened in Excel."""
    return "'" + value if value and value[0] in "=+-@\t\r" else value


def append_history(rows):
    """rows: list of dicts with timestamp, text, sentiment, confidence."""
    with history_lock:
        ensure_history_file()
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow([r["timestamp"], csv_safe(r["text"]), r["sentiment"], r["confidence"]])


def read_history():
    """Return history rows, newest first."""
    with history_lock:
        ensure_history_file()
        with open(HISTORY_FILE, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    cleaned = []
    for row in rows:
        try:
            confidence = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        cleaned.append({
            "timestamp": row.get("timestamp", ""),
            "feedback": row.get("feedback", ""),
            "sentiment": row.get("sentiment", ""),
            "confidence": confidence,
        })
    cleaned.reverse()
    return cleaned


def compute_stats(rows):
    stats = {"total": len(rows), "Positive": 0, "Negative": 0, "Neutral": 0}
    for row in rows:
        if row["sentiment"] in stats:
            stats[row["sentiment"]] += 1
    return stats


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# Upload parsing
# --------------------------------------------------------------------------
def read_text_file(path):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise FeedbackFileError("The file could not be read. Please save it as UTF-8 text.")


def parse_txt(path):
    lines = [line.strip() for line in read_text_file(path).splitlines()]
    return [line for line in lines if line]


def parse_csv(path):
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise FeedbackFileError("The CSV file is empty.")
        except pd.errors.ParserError:
            raise FeedbackFileError("The CSV file is invalid or badly formatted.")
    if df is None:
        raise FeedbackFileError("The CSV file could not be read. Please save it as UTF-8.")
    if df.empty or len(df.columns) == 0:
        raise FeedbackFileError("The CSV file is empty.")

    columns = {str(c).strip().lower(): c for c in df.columns}
    if "feedback" in columns:
        column = columns["feedback"]
    elif len(df.columns) == 1:
        column = df.columns[0]
    else:
        raise FeedbackFileError("The CSV needs a column named 'feedback' (or only one column).")

    texts = [str(v).strip() for v in df[column].tolist()]
    return [t for t in texts if t]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Invalid request. Please send JSON with a 'text' field."), 400

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Please enter some feedback before analyzing."), 400
    text = text.strip()
    if len(text) > cfg.MAX_TEXT_LENGTH:
        return jsonify(error=f"Feedback is too long. Please keep it under {cfg.MAX_TEXT_LENGTH} characters."), 400
    if model is None:
        return jsonify(error="The sentiment model is not available. Please check the server logs."), 503

    try:
        result = analyze_texts([text])[0]
        result["timestamp"] = now_string()
        append_history([result])
        return jsonify(result)
    except Exception:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify(error="Sorry, the analysis failed. Please try again."), 500


@app.route("/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify(error="The sentiment model is not available. Please check the server logs."), 503

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="Please choose a CSV or TXT file to upload."), 400

    original_name = secure_filename(file.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in cfg.ALLOWED_EXTENSIONS:
        return jsonify(error="Unsupported file type. Please upload a .csv or .txt file."), 400

    # Save under a random name, parse it as plain data (never executed), then delete it.
    UPLOAD_DIR.mkdir(exist_ok=True)
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    try:
        file.save(temp_path)
        texts = parse_csv(temp_path) if extension == ".csv" else parse_txt(temp_path)
        if not texts:
            raise FeedbackFileError("No feedback entries were found in the file.")
        if len(texts) > cfg.MAX_BATCH_ROWS:
            raise FeedbackFileError(f"The file has too many entries. The limit is {cfg.MAX_BATCH_ROWS}.")

        texts = [t[:cfg.MAX_TEXT_LENGTH] for t in texts]
        results = analyze_texts(texts)
        timestamp = now_string()
        for r in results:
            r["timestamp"] = timestamp
        append_history(results)

        summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for r in results:
            summary[r["sentiment"]] += 1
        return jsonify(
            count=len(results),
            summary=summary,
            results=[{"feedback": r["text"], "sentiment": r["sentiment"],
                      "confidence": r["confidence"]} for r in results],
        )
    except FeedbackFileError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:  # noqa: BLE001
        logger.exception("Upload failed")
        return jsonify(error="Sorry, the file could not be processed."), 500
    finally:
        temp_path.unlink(missing_ok=True)


@app.route("/history", methods=["GET"])
def history():
    try:
        rows = read_history()
        return jsonify(history=rows, stats=compute_stats(rows))
    except Exception:  # noqa: BLE001
        logger.exception("Reading history failed")
        return jsonify(error="Could not load the history."), 500


@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        with history_lock:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(cfg.HISTORY_COLUMNS)
        return jsonify(message="History cleared.")
    except Exception:  # noqa: BLE001
        logger.exception("Clearing history failed")
        return jsonify(error="Could not clear the history."), 500


# --------------------------------------------------------------------------
# Error handlers - always JSON, never stack traces
# --------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_exception(exc):
    if isinstance(exc, HTTPException):
        messages = {
            404: "The requested page was not found.",
            405: "This action is not allowed.",
            413: f"The file is too large. The maximum size is {cfg.MAX_UPLOAD_MB} MB.",
        }
        return jsonify(error=messages.get(exc.code, "The request could not be processed.")), exc.code
    logger.exception("Unhandled server error")
    return jsonify(error="Something went wrong on the server. Please try again."), 500


if __name__ == "__main__":
    ensure_history_file()
    UPLOAD_DIR.mkdir(exist_ok=True)
    # debug=False so the reloader doesn't load the model twice.
    app.run(host="127.0.0.1", port=5000, debug=False)
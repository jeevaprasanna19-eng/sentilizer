# NLP Sentiment Analyzer

## 1. Project overview
A Flask web app that analyzes customer feedback with a pretrained **BERT** model and classifies it as
**Positive**, **Negative**, or **Neutral**. It helps support teams spot unhappy customers quickly.

## 2. Features
- Analyze a single piece of feedback or upload a CSV/TXT file
- Confidence percentage and star score for every result
- Dashboard statistics (total / positive / negative / neutral)
- History table with search, sentiment filter, and clear button
- Light/dark theme, responsive layout, mobile-friendly tables
- Model loaded once at startup; automatic GPU (CUDA) use when available

## 3. Technologies
HTML5, CSS3, vanilla JavaScript · Python, Flask · Hugging Face Transformers, PyTorch, pandas

## 4. Project structure
| File | Purpose |
|------|---------|
| `app.py` | Flask server, routes, BERT inference, CSV history |
| `models/model_config.py` | Model name, star→sentiment mapping, limits |
| `templates/index.html` | Dashboard page |
| `static/css/style.css` | Styling |
| `static/js/script.js` | Frontend logic (Fetch API) |
| `uploads/.gitkeep` | Temporary upload folder |
| `data/feedback_history.csv` | Stored history |
| `requirements.txt` | Dependencies |

## 5-8. Installation and running (Windows, VS Code)
Use Python 3.10-3.12. Open the project folder in VS Code, open a terminal (Ctrl+`), then:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

The first start downloads the model (~700 MB). Later starts are fast.

## 9-10. Example inputs and expected outputs
| Input | Output |
|-------|--------|
| The product is amazing and the delivery was very fast. | Positive |
| The product arrived damaged and I'm very disappointed. | Negative |
| The product arrived yesterday. It works as expected. | Neutral* |
| I have been waiting for my refund for two weeks. This is extremely frustrating. | Negative |

*The model predicts star ratings, and plain factual statements can be rated 3 or 4 stars, so Neutral vs.
Positive can vary. You can adjust `STAR_TO_SENTIMENT` in `models/model_config.py`.

## 11. CSV format
```csv
feedback
"Great service, very happy!"
"The app keeps crashing."
```
Use a column named `feedback`, or a single-column file. TXT files use one feedback per line.
Limits: 2 MB, 1000 entries, 2000 characters per entry.

## 12. BERT model information
`nlptown/bert-base-multilingual-uncased-sentiment` is a BERT model fine-tuned on product reviews in six
languages. It outputs probabilities for 1-5 stars, mapped to Negative (1-2), Neutral (3), Positive (4-5).

## 13. API endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main page |
| POST | `/predict` | Body `{"text": "..."}` → text, sentiment, confidence, score, timestamp |
| POST | `/upload` | Multipart field `file` (.csv/.txt) → count, summary, results |
| GET | `/history` | History (newest first) + statistics |
| POST | `/clear-history` | Clears history |

## 14. Troubleshooting
- **`'python' is not recognized`**: reinstall Python and tick "Add to PATH".
- **Model download fails**: check your internet connection and retry.
- **`No module named 'models'`**: run `python app.py` from the project root.
- **Port 5000 in use**: change `port=5000` at the bottom of `app.py`.
- **"The sentiment model is not available"**: see the terminal log for the loading error.

## 15. Future improvements
Charts over time, export results to CSV, user login, multilingual UI, fine-tuning on your own data,
a dedicated 3-class model, Docker deployment.
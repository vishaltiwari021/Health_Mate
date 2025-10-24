## Purpose

This file tells AI coding assistants how to be immediately productive in the HealthMarteAI repository.

## Big picture

- This repo is a small Flask-based BMI prediction service. The main app is `app.py` (Flask) which loads a pre-trained scikit-learn RandomForest model (`bmi_model.joblib`) and a `StandardScaler` (`bmi_scaler.joblib`) and exposes REST endpoints on port 5000.
- Data source and retraining: the canonical dataset is `bmi.csv`. The `/retrain` endpoint (or running `train_model()` via `python app.py`) will train and overwrite `bmi_model.joblib` and `bmi_scaler.joblib`.
- Optional chat: the `/chatbot` endpoint uses Google Gemini if `GEMINI_API_KEY` is set in environment variables. If not set, the endpoint returns a message saying the chatbot is disabled.

## Key files to reference

- `app.py` — single-file Flask app; contains model loading, training, endpoints, validation rules, and Gemini integration.
- `bmi.csv` — input dataset used for training; `train_model()` expects columns `['Age','Height','Weight','BmiClass']` and drops rows with missing values.
- `bmi_model.joblib`, `bmi_scaler.joblib` — artifacts created by training; model is RandomForestClassifier with `n_estimators=100`.
- `bmi_web_interface.html`, `bmi.js`, `style.css` — lightweight UI; useful for manual testing of endpoints.

## Developer workflows & commands (explicit)

- Install dependencies (recommended virtualenv). Required packages inferred from `app.py`:
  - Flask, flask-cors, pandas, numpy, scikit-learn, joblib, requests
- Run locally (PowerShell):
  - python app.py
  - The server listens on port 5000 by default (host 0.0.0.0, debug=True in current dev setup).
- To train/retrain model from data file:
  - Ensure `bmi.csv` exists in the repo root
  - POST to `/retrain` (or restart the server when `train_model()` runs). The endpoint returns accuracy if successful.
- To enable chatbot endpoint set environment variable `GEMINI_API_KEY` before starting the server. If unset, `/chatbot` returns an explanatory message.

## API contracts & examples (from `app.py`)

- GET / -> returns service info and endpoints
- GET /health -> returns service health and whether model/scaler are loaded
- POST /predict -> JSON body with { "age": number, "height": number (meters), "weight": number (kg) }
  - Validation rules in code:
    - age: 1–120
    - height: 0.5–2.5 (meters)
    - weight: 10–500 (kg)
  - Response contains calculated BMI, actual class (by formula), predicted_class, confidence and model_accuracy
- POST /retrain -> triggers training using `bmi.csv` and overwrites joblib artifacts
- POST /chatbot -> { "message": "..." } (requires `GEMINI_API_KEY` to be useful)

## Project-specific conventions & patterns

- Single-file service: expect `app.py` to contain most behavior — search there for any behavioral change.
- Model artifacts live in the repo root and are loaded by filename; treat `bmi_model.joblib` and `bmi_scaler.joblib` as authoritative for runtime unless retraining is requested.
- Data checks are conservative: `train_model()` drops rows with missing required columns and warns on very small datasets (<10 rows).
- Error handling: endpoints often return JSON `{ "error": "..." }` with appropriate HTTP status codes (400, 500, 503). Follow the same pattern for new endpoints.

## Integration points & external dependencies

- Google Gemini (optional): configured via `GEMINI_API_KEY` environment variable. The code expects Gemini to return nested JSON shapes; follow the existing parsing logic when extending.
- No other external services are required; dependencies are local Python packages.

## When changing code — quick checklist

1. Update `app.py` only where necessary; prefer adding small helper functions rather than altering core flow.
2. If changing model features, update the `features` list in `/model-info` response and update training/serving code consistently.
3. When modifying validation bounds, update both `/predict` and any UI JS (`bmi.js`) to keep client/server validation aligned.
4. Preserve JSON error shapes (use `{'error': 'message'}`) for compatibility with front-end code.

## Examples to reference in edits

- Use the `predict_bmi` function for input parsing/validation patterns.
- Use `load_model()` and `train_model()` as canonical model lifecycle hooks.

If anything here is unclear or you want different focus (more on tests, CI, or packaging), tell me which area and I will update the instructions accordingly.

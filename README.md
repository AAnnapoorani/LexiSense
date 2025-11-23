**MCA eConsultation — AI Portal**

- **Description**: A Streamlit app that provides a merged admin and individual portal for eConsultation analysis. It supports local NLP pipelines (summarization, sentiment, translation, TF-IDF keyword extraction, wordclouds) with a mock ML fallback and an optional external ML API integration.

- **Main file**: `working.py`

**Quick Start**
- **Install dependencies**: Use the provided `requirements.txt` (see notes about `torch` below).

```powershell
pip install -r requirements.txt
```

- **Run the app**:

```powershell
python -m streamlit run working.py
```

**Configuration**
- **Top-level toggles** (edit `working.py`):
  - `USE_ML_API` : Set to `True` to call an external ML server; default `False` uses local pipelines + mock fallback.
  - `ML_API_URL` : URL for the external ML server (expects `/analyze` and `/analyze/batch`).
  - `API_KEY` : Bearer token used when calling external ML API.
  - `SUMMARIZATION_MODEL_NAME` / `SENTIMENT_MODEL_NAME` : Hugging Face model names used by the local pipelines.

**What it does**
- Accepts uploads (CSV, XLSX, JSON, PDF, ZIP) or direct individual submissions.
- Parses files and extracts text records.
- Optionally translates non-English text to English using `deep-translator`.
- Summarizes long texts using a Hugging Face summarization pipeline.
- Runs sentiment analysis (local `nlptown` model when available) with confidence mapping and buckets.
- Extracts keywords using TF-IDF and generates a word cloud.
- Admin dashboard: filtering, KPIs, charts, exports (CSV/JSON/PDF).

**Notes & Tips**
- The app tries to load Hugging Face models locally. The first run may download models and take time.
- `torch` installation on Windows may require following official installation instructions (CUDA vs CPU). If you have GPU and want GPU support, install `torch` with the appropriate CUDA wheel from https://pytorch.org/get-started/locally/.
- If you don't want to use local large models, set `USE_ML_API = True` and provide a compatible ML server.
- `TextBlob` may work without extra corpora for basic polarity; if you need full features you may also need to download `nltk` corpora.

**Files of interest**
- `working.py` : main Streamlit app (upload, parsing, analysis, admin dashboard).
- `requirements.txt` : Python dependencies for the project.

**License / Attribution**
- Prototype code; adapted for demo use. Replace mock logic and credentials before production.


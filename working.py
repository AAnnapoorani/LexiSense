# app.py
# Merged eConsultation admin/individual portal + full NLP pipeline:
# - Local HuggingFace summarization + sentiment (DistilBART + nlptown)
# - Translation (deep_translator)
# - Keyword extraction (TF-IDF)
# - Word cloud
# - Keeps existing mock ML + ML API toggle and admin dashboard + exports

import os
import io
import uuid
import time
import json
import tempfile
import re
import zipfile
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image, UnidentifiedImageError

# NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import torch
from langdetect import detect
from deep_translator import GoogleTranslator
from textblob import TextBlob

# file parsing
import PyPDF2
import requests

# ---------------- CONFIG ----------------
USE_ML_API = False  # If True, will call external ML API; otherwise uses local pipelines + mock fallback
ML_API_URL = "http://localhost:8000/analyze"  # if USE_ML_API True, this is expected
API_KEY = "my-secret-key-IOTA-7818"  # used when calling external ML
RESULT_FILE = "all_policy_comments.csv"
# summarization & sentiment model names
SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE = 16
SUMMARY_WORD_THRESHOLD = 50
# ----------------------------------------

st.set_page_config(page_title="MCA eConsultation — AI Portal (Merged)", layout="wide", page_icon="🗳️")

# ---------------- Helper UI / Utilities ----------------
def load_logo_safe(path="assets/logo.png", width=180):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path).convert("RGBA")
        return img
    except Exception:
        return None

def generate_pdf_bytes(df, title="Analysis Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)
    try:
        total = len(df)
    except:
        total = 0
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Total items: {total}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    cols = ["id", "original", "translated", "summary", "sentiment_bucket", "confidence", "keywords"]
    for c in cols:
        pdf.cell(30, 8, c[:15], border=1)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for i, row in df.iterrows():
        if i >= 200:
            pdf.cell(0, 8, f"... truncated (only first 200 rows included)", ln=True)
            break
        # safe text
        try:
            orig = str(row.get("original",""))[:200]
        except:
            orig = ""
        pdf.multi_cell(0, 6, txt=f"{row.get('id','')}: {orig}", border=1)
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ---------------- Mock ML logic (kept intact) ----------------
NEG_WORDS = {"not","no","never","bad","worse","terrible","problem","concern","issue","against","oppose"}
POS_WORDS = {"good","great","support","positive","helpful","benefit","agree","excellent","improve"}

def sentiment_bucket(score):
    # buckets (same as earlier but numeric)
    if score >= 0.70:
        return "very_positive", "😄", "#006400"
    if 0.20 <= score < 0.70:
        return "positive", "🙂", "#7CFC00"
    if -0.19 <= score <= 0.19:
        return "neutral", "😐", "#FFD700"
    if -0.69 <= score < -0.20:
        return "negative", "🙁", "#FFA500"
    return "very_negative", "😠", "#FF0000"

def mock_analyze_text(id_, text):
    language = "en" if all(ord(c) < 128 for c in text) else "hi"
    t_lower = text.lower()
    neg = any(w in t_lower for w in NEG_WORDS)
    pos = any(w in t_lower for w in POS_WORDS)
    if neg and not pos:
        score = -0.8
    elif pos and not neg:
        score = 0.75
    elif pos and neg:
        score = 0.0
    else:
        score = 0.0
    bucket, emoji, color = sentiment_bucket(score)
    summary = text.strip().split(".")[0][:250] + ("..." if len(text) > 250 else "")
    translated = text if language == "en" else f"[Translated to English] {text[:300]}"
    words = [w.strip(".,;:!?'\"()") for w in t_lower.split() if len(w) > 3]
    freq = Counter(words)
    keywords = [w for w, _ in freq.most_common(5)]
    confidence = round(0.6 + abs(score) * 0.35, 2)
    return {
        "id": id_,
        "original": text,
        "translated": translated,
        "summary": summary,
        "sentiment_score": score,
        "sentiment_bucket": bucket,
        "emoji": emoji,
        "color": color,
        "confidence": confidence,
        "keywords": keywords,
        "language": language,
        "timestamp": datetime.utcnow().isoformat()
    }

def mock_analyze_batch(items):
    out = []
    for x in items:
        out.append(mock_analyze_text(x["id"], x["text"]))
    return out

# ---------------- Session init ----------------
if "uploads" not in st.session_state:
    st.session_state["uploads"] = {}
if "comments" not in st.session_state:
    st.session_state["comments"] = {}
if "drafts" not in st.session_state:
    st.session_state["drafts"] = {}
if "filter_word" not in st.session_state:
    st.session_state["filter_word"] = None

# ---------------- Model loaders (cached) ----------------
@st.cache_resource
def load_summarizer(device_id=-1):
    try:
        device = 0 if (device_id == 0 and torch.cuda.is_available()) else -1
        return pipeline("summarization", model=SUMMARIZATION_MODEL_NAME, device=device)
    except Exception as e:
        st.warning(f"Failed to load summarizer: {e}")
        return None

@st.cache_resource
def load_sentiment_model(device_id=-1):
    try:
        device = 0 if (device_id == 0 and torch.cuda.is_available()) else -1
        return pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=device)
    except Exception as e:
        st.warning(f"Failed to load sentiment model: {e}")
        return None

# ---------------- Translation helper ----------------
def translate_to_english(text):
    try:
        # guard empty
        if not text or str(text).strip() == "":
            return ""
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source="auto", target="en").translate(text)
        else:
            return text
    except Exception:
        return text

# ---------------- TF-IDF keywords helper ----------------
def extract_tfidf_keywords(texts, top_k=10):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sums = np.asarray(X.sum(axis=0)).ravel()
        top_indices = tfidf_sums.argsort()[::-1][:top_k]
        keywords = [feature_names[i] for i in top_indices]
        return keywords
    except Exception:
        # fallback basic token frequency
        words = []
        for t in texts:
            words.extend([w.strip(".,;:!?'\"()") for w in str(t).lower().split() if len(w) > 3])
        freq = Counter(words)
        return [w for w,_ in freq.most_common(top_k)]

# ------------------ ML API wrappers / local pipeline ------------------
def call_ml_single(id_, text, use_gpu=False):
    """Call external ML API if configured otherwise run local pipeline (or mock)."""
    if USE_ML_API:
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        try:
            resp = requests.post(ML_API_URL, json={"id": id_, "text": text, "metadata": {}}, headers=headers, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"ML API error (single): {e}")
            return mock_analyze_text(id_, text)
    else:
        # run local enriched pipeline for single text
        return local_analyze_text(id_, text, use_gpu=use_gpu)

def call_ml_batch(list_items, use_gpu=False):
    """Call external batch API or local batch pipeline."""
    if USE_ML_API:
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        try:
            batch_url = ML_API_URL.rstrip("/") + "/batch"
            resp = requests.post(batch_url, json=list_items, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"ML batch API error: {e}")
            return mock_analyze_batch(list_items)
    else:
        return local_analyze_batch(list_items, use_gpu=use_gpu)

# ------------------ Local pipeline functions ------------------
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def local_analyze_text(id_, text, summarizer=None, sentiment_model=None, use_gpu=False):
    """
    Analyze one text using local summarizer and sentiment pipelines.
    Returns a dict similar to mock_analyze_text but enriched.
    """
    # fallback to mock if any critical error
    try:
        raw = str(text)
        raw_clean = clean_text(raw)
        # language / translate
        try:
            lang = detect(raw_clean) if raw_clean.strip() else "en"
        except:
            lang = "en"
        if lang != "en":
            try:
                translated = GoogleTranslator(source="auto", target="en").translate(raw_clean)
            except Exception:
                translated = raw_clean
        else:
            translated = raw_clean

        # summarization: if long
        summary = translated
        if len(translated.split()) > SUMMARY_WORD_THRESHOLD:
            # ensure summarizer loaded
            device_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
            if summarizer is None:
                summarizer = load_summarizer(device_id=device_id)
            if summarizer is not None:
                try:
                    out = summarizer(translated, truncation=True)
                    if isinstance(out, list) and len(out) > 0 and 'summary_text' in out[0]:
                        summary = out[0]['summary_text']
                    elif isinstance(out, dict) and 'summary_text' in out:
                        summary = out['summary_text']
                except Exception:
                    # fallback truncation
                    summary = " ".join(translated.split()[:250])
            else:
                summary = " ".join(translated.split()[:250])
        # sentiment
        device_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
        if sentiment_model is None:
            sentiment_model = load_sentiment_model(device_id=device_id)
        sentiment_label = "Neutral"
        sentiment_score = 0.0
        confidence = 0.0
        if sentiment_model is not None:
            try:
                # nlptown returns '1 star'..'5 stars' labels with scores
                out = sentiment_model(translated[:1000])  # limit length
                if isinstance(out, list) and out:
                    res = out[0]
                else:
                    res = out
                raw_label = res.get('label', '')
                score = res.get('score', 0.0)
                # map 1-5 to -1..+1 range roughly
                try:
                    if isinstance(raw_label, str) and raw_label.strip():
                        if "1" in raw_label:
                            sentiment_score = -0.9
                        elif "2" in raw_label:
                            sentiment_score = -0.5
                        elif "3" in raw_label:
                            sentiment_score = 0.0
                        elif "4" in raw_label:
                            sentiment_score = 0.5
                        elif "5" in raw_label:
                            sentiment_score = 0.9
                        else:
                            sentiment_score = 0.0
                    else:
                        sentiment_score = 0.0
                except Exception:
                    sentiment_score = 0.0
                confidence = float(score)
                sentiment_label = raw_label
            except Exception:
                # fallback to TextBlob polarity
                try:
                    polarity = TextBlob(translated).sentiment.polarity
                    sentiment_score = polarity
                    confidence = 0.6
                except:
                    sentiment_score = 0.0
                    confidence = 0.0
        else:
            # fallback simple heuristic
            t_lower = translated.lower()
            neg = any(w in t_lower for w in NEG_WORDS)
            pos = any(w in t_lower for w in POS_WORDS)
            if neg and not pos:
                sentiment_score = -0.8
            elif pos and not neg:
                sentiment_score = 0.75
            elif pos and neg:
                sentiment_score = 0.0
            else:
                sentiment_score = 0.0
            confidence = 0.6

        bucket, emoji, color = sentiment_bucket(sentiment_score)

        # keywords (small local extraction)
        # we'll extract 5 keywords from the translated text
        try:
            tfidf_kws = extract_tfidf_keywords([translated], top_k=5)
            keywords = tfidf_kws if tfidf_kws else []
        except Exception:
            words = [w.strip(".,;:!?'\"()") for w in translated.lower().split() if len(w) > 3]
            freq = Counter(words)
            keywords = [w for w,_ in freq.most_common(5)]

        return {
            "id": id_,
            "original": raw,
            "translated": translated,
            "summary": summary,
            "sentiment_score": sentiment_score,
            "sentiment_bucket": bucket,
            "emoji": emoji,
            "color": color,
            "confidence": confidence,
            "keywords": keywords,
            "language": lang,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        st.error(f"Local analyze error: {e}")
        return mock_analyze_text(id_, text)

def local_analyze_batch(list_items, use_gpu=False):
    """
    Efficient-ish local batch processing:
    - translate
    - summarize long texts in batches
    - sentiment in batches
    - compute keywords on full corpus (TF-IDF)
    Returns list of dicts.
    """
    if not list_items:
        return []
    # prepare
    ids = [it["id"] for it in list_items]
    raw_texts = [clean_text(it.get("text","")) for it in list_items]
    # detect languages and translate where needed
    translated_texts = []
    languages = []
    for t in raw_texts:
        try:
            lang = detect(t) if t.strip() else "en"
        except:
            lang = "en"
        languages.append(lang)
        if lang != "en":
            try:
                translated_texts.append(GoogleTranslator(source="auto", target="en").translate(t))
            except Exception:
                translated_texts.append(t)
        else:
            translated_texts.append(t)
    # summarization for long texts in chunks
    summarizer = None
    if any(len(t.split()) > SUMMARY_WORD_THRESHOLD for t in translated_texts):
        device_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
        summarizer = load_summarizer(device_id=device_id)
    summaries = list(translated_texts)
    long_indices = [i for i,t in enumerate(translated_texts) if len(t.split()) > SUMMARY_WORD_THRESHOLD]
    if long_indices and summarizer is not None:
        # call summarizer in batches
        texts_to_summarize = [translated_texts[i] for i in long_indices]
        for i in range(0, len(texts_to_summarize), BATCH_SIZE):
            chunk = texts_to_summarize[i:i+BATCH_SIZE]
            try:
                outs = summarizer(chunk, truncation=True)
                for j, out in enumerate(outs):
                    sidx = long_indices[i+j]
                    if isinstance(out, dict) and 'summary_text' in out:
                        summaries[sidx] = out['summary_text']
                    elif isinstance(out, str):
                        summaries[sidx] = out
                    else:
                        summaries[sidx] = summaries[sidx][:250]
            except Exception:
                # fallback to truncation
                for j, _ in enumerate(chunk):
                    sidx = long_indices[i+j]
                    summaries[sidx] = " ".join(translated_texts[sidx].split()[:250])
    else:
        # quick truncation for long texts
        for i in long_indices:
            summaries[i] = " ".join(translated_texts[i].split()[:250])

    # sentiment in batches
    sentiment_model = None
    device_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
    sentiment_model = load_sentiment_model(device_id=device_id)
    sentiments_out = []
    if sentiment_model is not None:
        texts_for_sent = [t[:1000] for t in translated_texts]
        for i in range(0, len(texts_for_sent), BATCH_SIZE):
            chunk = texts_for_sent[i:i+BATCH_SIZE]
            try:
                outs = sentiment_model(chunk)
                sentiments_out.extend(outs)
            except Exception as e:
                # fallback: generate neutral responses
                for _ in chunk:
                    sentiments_out.append({"label":"3 stars","score":0.6})
    else:
        for _ in translated_texts:
            sentiments_out.append({"label":"3 stars","score":0.6})

    # keywords: compute top keywords across all translations
    keywords_global = extract_tfidf_keywords(translated_texts, top_k=200)
    # for each document, pick top matching keywords by substring match
    results = []
    for idx, _id in enumerate(ids):
        label = sentiments_out[idx]
        raw_label = label.get("label","")
        score = label.get("score", 0.0)
        # map label to numeric score (rough)
        if isinstance(raw_label, str):
            if "1" in raw_label:
                sentiment_score = -0.9
            elif "2" in raw_label:
                sentiment_score = -0.5
            elif "3" in raw_label:
                sentiment_score = 0.0
            elif "4" in raw_label:
                sentiment_score = 0.5
            elif "5" in raw_label:
                sentiment_score = 0.9
            else:
                sentiment_score = 0.0
        else:
            sentiment_score = 0.0
        confidence = float(score)
        bucket, emoji, color = sentiment_bucket(sentiment_score)
        # per-document keywords: choose from global tfidf list which appear in doc
        t = translated_texts[idx].lower()
        doc_keywords = [k for k in keywords_global if k in t][:8]
        if not doc_keywords:
            # fallback to frequency
            words = [w.strip(".,;:!?'\"()") for w in t.split() if len(w) > 3]
            freq = Counter(words)
            doc_keywords = [w for w,_ in freq.most_common(5)]
        res = {
            "id": _id,
            "original": raw_texts[idx],
            "translated": translated_texts[idx],
            "summary": summaries[idx],
            "sentiment_score": sentiment_score,
            "sentiment_bucket": bucket,
            "emoji": emoji,
            "color": color,
            "confidence": confidence,
            "keywords": doc_keywords,
            "language": languages[idx],
            "timestamp": datetime.utcnow().isoformat()
        }
        results.append(res)
    return results

# ------------------ File parsing (kept) ------------------
def parse_uploaded_file(uploaded_file):
    """
    Return list of dicts: [{'id': id, 'text': text, 'metadata': {...}}, ...]
    """
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else ""
    records = []
    try:
        if name.endswith(".csv"):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            text_col = None
            for c in df.columns:
                if c.lower() in ("text","comment","body","message","review","description"):
                    text_col = c
                    break
            if text_col is None:
                string_cols = df.select_dtypes(include=[object]).columns.tolist()
                text_col = string_cols[0] if string_cols else df.columns[0]
            for i, row in df.iterrows():
                txt = str(row.get(text_col, ""))
                rec_id = str(uuid.uuid4())
                records.append({"id": rec_id, "text": txt, "metadata": {"row_index": int(i)}})
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            text_col = None
            for c in df.columns:
                if c.lower() in ("text","comment","body","message","review","description"):
                    text_col = c
                    break
            if text_col is None:
                text_col = df.columns[0]
            for i, row in df.iterrows():
                txt = str(row.get(text_col, ""))
                rec_id = str(uuid.uuid4())
                records.append({"id": rec_id, "text": txt, "metadata": {"row_index": int(i)}})
        elif name.endswith(".json"):
            uploaded_file.seek(0)
            try:
                data = json.load(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                data = json.loads(uploaded_file.read().decode(errors="ignore"))
            if isinstance(data, dict):
                found = False
                for v in data.values():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                txt = item.get("text") or item.get("comment") or json.dumps(item)
                            else:
                                txt = str(item)
                            rec_id = str(uuid.uuid4())
                            records.append({"id": rec_id, "text": txt, "metadata": {}})
                        found = True
                        break
                if not found:
                    records.append({"id": str(uuid.uuid4()), "text": json.dumps(data), "metadata": {}})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        txt = item.get("text") or item.get("comment") or json.dumps(item)
                    else:
                        txt = str(item)
                    rec_id = str(uuid.uuid4())
                    records.append({"id": rec_id, "text": txt, "metadata": {}})
        elif name.endswith(".pdf"):
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            for p, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    rec_id = str(uuid.uuid4())
                    records.append({"id": rec_id, "text": txt, "metadata": {"page": p+1}})
        elif name.endswith(".zip"):
            uploaded_file.seek(0)
            z = zipfile.ZipFile(uploaded_file)
            for fn in z.namelist():
                if fn.lower().endswith((".csv",".json",".txt",".pdf",".xlsx")):
                    with z.open(fn) as f:
                        sub = io.BytesIO(f.read())
                        sub.name = fn
                        sub_records = parse_uploaded_file(sub)
                        records.extend(sub_records)
        else:
            uploaded_file.seek(0)
            try:
                txt = uploaded_file.read().decode(errors="ignore")
            except:
                txt = str(uploaded_file.read())
            records.append({"id": str(uuid.uuid4()), "text": txt, "metadata": {}})
    except Exception as e:
        st.error(f"Failed parse file: {e}")
    return records

# ------------------ App UI ------------------
logo = load_logo_safe()
with st.sidebar:
    if logo:
        st.image(logo, width=180)
    else:
        st.markdown("## MCA eConsultation\nAI Portal")
    st.markdown("---")
    mode = st.radio("Mode", ("Individual (Stakeholder)","Admin (MCA Officer)"))
    st.markdown("---")
    st.checkbox("Use GPU if available (for local models)", value=False, key="use_gpu_sidebar")
    st.markdown("Model config:")
    st.write(f"USE_ML_API = {USE_ML_API}")
    if USE_ML_API:
        st.write("ML API URL:")
        st.write(ML_API_URL)

# Session defaults
if "admin_logged" not in st.session_state:
    st.session_state["admin_logged"] = False
if "admin_user" not in st.session_state:
    st.session_state["admin_user"] = None
if "current_upload_id" not in st.session_state:
    st.session_state["current_upload_id"] = None
if "current_upload_bytes" not in st.session_state:
    st.session_state["current_upload_bytes"] = None
if "current_upload_name" not in st.session_state:
    st.session_state["current_upload_name"] = None

# -------- Admin Mode ----------
if mode.startswith("Admin"):
    st.header("Admin Portal — MCA Officers")
    if not st.session_state["admin_logged"]:
        cols = st.columns([1,1,1])
        with cols[1]:
            st.subheader("Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username.strip().lower() in ("admin","mca") and password.strip() == "admin123":
                    st.session_state["admin_logged"] = True
                    st.session_state["admin_user"] = username
                    st.success("Logged in")
                else:
                    st.error("Invalid credentials")
        st.stop()

    st.sidebar.markdown(f"**Logged in as:** {st.session_state['admin_user']}")
    st.markdown("### Upload & Analyze")
    uploaded_file = st.file_uploader(
        "Drag & drop or select a file (CSV, XLSX, JSON, PDF, ZIP)",
        type=["csv","xlsx","json","pdf","zip"],
        accept_multiple_files=False,
        help="Accepted: csv, xlsx, json, pdf, zip (zip may contain multiple files)."
    )

    # If a new upload, stage it
    if uploaded_file is not None:
        st.session_state["current_upload_name"] = uploaded_file.name
        st.session_state["current_upload_bytes"] = uploaded_file.getvalue()
        st.success(f"File staged: {uploaded_file.name} ({round(len(st.session_state['current_upload_bytes'])/1024,1)} KB)")
        try:
            st.markdown("**Preview / first records**")
            if uploaded_file.name.lower().endswith(".csv"):
                df_preview = pd.read_csv(io.BytesIO(st.session_state["current_upload_bytes"]))
                st.dataframe(df_preview.head(5))
            elif uploaded_file.name.lower().endswith((".xlsx",".xls")):
                df_preview = pd.read_excel(io.BytesIO(st.session_state["current_upload_bytes"]))
                st.dataframe(df_preview.head(5))
            elif uploaded_file.name.lower().endswith(".json"):
                try:
                    loaded = json.loads(st.session_state["current_upload_bytes"])
                    st.write("JSON preview (top element):")
                    st.json(loaded if isinstance(loaded, dict) else (loaded[:5] if isinstance(loaded, list) else str(loaded)[:1000]))
                except Exception:
                    st.write("Cannot preview JSON.")
            elif uploaded_file.name.lower().endswith(".pdf"):
                st.info("PDF uploaded — preview not shown here. Will be parsed during ANALYZE.")
            elif uploaded_file.name.lower().endswith(".zip"):
                st.info("ZIP uploaded — contents will be parsed during ANALYZE.")
        except Exception as e:
            st.warning(f"Preview failed: {e}")

        if st.button("Remove File"):
            st.session_state.pop("current_upload_bytes", None)
            st.session_state.pop("current_upload_name", None)
            st.success("File removed")

    analyze_disabled = "current_upload_bytes" not in st.session_state or st.session_state["current_upload_bytes"] is None
    if st.button("Analyze (start processing)", disabled=analyze_disabled):
        if analyze_disabled:
            st.warning("Please upload a file first.")
        else:
            b = st.session_state["current_upload_bytes"]
            tmp = io.BytesIO(b)
            tmp.name = st.session_state["current_upload_name"]
            st.info("Parsing uploaded file...")
            records = parse_uploaded_file(tmp)
            upload_id = str(uuid.uuid4())
            st.session_state["uploads"][upload_id] = {
                "id": upload_id,
                "filename": tmp.name,
                "num_records": len(records),
                "uploaded_at": datetime.utcnow().isoformat(),
                "uploader": st.session_state.get("admin_user","admin")
            }
            st.session_state["current_upload_id"] = upload_id
            st.info(f"Parsed {len(records)} records; running analysis...")
            # prepare batch input
            batch = [{"id": r["id"], "text": r["text"], "metadata": r.get("metadata",{})} for r in records]
            use_gpu = st.session_state.get("use_gpu_sidebar", False)
            results = call_ml_batch(batch, use_gpu=use_gpu)
            for res in results:
                st.session_state["comments"][res["id"]] = res
            st.success(f"Analysis complete. Upload id: {upload_id}. {len(results)} comments processed.")
            st.cache_data.clear()

    st.markdown("---")
    st.header("Analysis Dashboard")
    if len(st.session_state["comments"]) == 0:
        st.info("No analyzed comments yet. Upload and Analyze to see dashboard.")
        st.markdown("**Real-time test input** (submit on Individual page to see it appear here).")
    else:
        df = pd.DataFrame(list(st.session_state["comments"].values()))
        # Filters
        with st.expander("Filters & Segmentation", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            draft_filter = c1.selectbox("Source file", options=["All"] + [st.session_state["uploads"][k]["filename"] for k in st.session_state["uploads"]])
            lang_filter = c2.selectbox("Language", options=["All"] + sorted(df["language"].dropna().unique().tolist()))
            stakeholder_filter = c3.selectbox("Stakeholder type (demo)", options=["All","Citizen","Company","Gov"])
            sent_options = ["All","very_positive","positive","neutral","negative","very_negative"]
            sentiment_filter = c4.selectbox("Sentiment bucket", options=sent_options)
            conf_min, conf_max = st.slider("Confidence % range", 0, 100, (0,100))
            conf_min = conf_min / 100.0
            conf_max = conf_max / 100.0

        # apply filters
        filt = df.copy()
        if sentiment_filter != "All":
            filt = filt[filt["sentiment_bucket"] == sentiment_filter]
        if lang_filter != "All":
            filt = filt[filt["language"] == lang_filter]
        filt = filt[(filt["confidence"] >= conf_min) & (filt["confidence"] <= conf_max)]
        if st.session_state.get("filter_word"):
            fw = st.session_state["filter_word"]
            filt = filt[filt["keywords"].apply(lambda arr: fw in (arr if isinstance(arr,list) else str(arr)) )]

        # KPI cards
        total = len(df)
        total_shown = len(filt)
        pos_count = (df["sentiment_bucket"].isin(["very_positive","positive"])).sum()
        neg_count = (df["sentiment_bucket"].isin(["very_negative","negative"])).sum()
        neutral_count = (df["sentiment_bucket"]=="neutral").sum()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total comments", total)
        k2.metric("Positive", f"{pos_count} ({pos_count/total*100:.1f}%)" if total else "0")
        k3.metric("Negative", f"{neg_count} ({neg_count/total*100:.1f}%)" if total else "0")
        k4.metric("Neutral", f"{neutral_count} ({neutral_count/total*100:.1f}%)" if total else "0")

        st.markdown("### Sentiment Distribution")
        bucket_labels = []
        bucket_vals = []
        bucket_colors = []
        for name_label in ["very_positive","positive","neutral","negative","very_negative"]:
            cnt = (df["sentiment_bucket"] == name_label).sum()
            # emoji/color mapping
            if name_label == "very_positive":
                emoji, color = "😄", "#006400"
            elif name_label == "positive":
                emoji, color = "🙂", "#7CFC00"
            elif name_label == "neutral":
                emoji, color = "😐", "#FFD700"
            elif name_label == "negative":
                emoji, color = "🙁", "#FFA500"
            else:
                emoji, color = "😠", "#FF0000"
            bucket_labels.append(f"{emoji} {name_label} ({cnt})")
            bucket_vals.append(cnt)
            bucket_colors.append(color)
        fig = px.pie(names=bucket_labels, values=bucket_vals, color_discrete_sequence=bucket_colors)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Volume over time (mock)")
        try:
            times = pd.to_datetime(df["timestamp"])
            bydate = pd.DataFrame({"date": times.dt.date, "sent": df["sentiment_bucket"]})
            agg = bydate.groupby(["date","sent"]).size().reset_index(name="count")
            if not agg.empty:
                pivot = agg.pivot(index="date", columns="sent", values="count").fillna(0)
                st.line_chart(pivot)
            else:
                st.info("Not enough timestamp info for time series.")
        except Exception:
            st.info("Time-series preview not available.")

        st.markdown("### Word Cloud & Keywords")
        all_keywords = []
        for k in df["keywords"].dropna():
            if isinstance(k, (list,tuple)):
                all_keywords.extend(k)
            elif isinstance(k, str):
                try:
                    parsed = json.loads(k)
                    if isinstance(parsed, list):
                        all_keywords.extend(parsed)
                    else:
                        all_keywords.extend(k.split())
                except:
                    all_keywords.extend(k.split())
        wc_text = " ".join(all_keywords) if all_keywords else "no-data"
        wc = WordCloud(width=800, height=300, background_color="white").generate(wc_text)
        st.image(wc.to_array(), use_container_width=True)
        wc_counts = Counter(all_keywords)
        if wc_counts:
            st.markdown("**Word counts (click to filter)**")
            cols = st.columns(4)
            i = 0
            for word, count in wc_counts.most_common(40):
                col = cols[i % 4]
                if col.button(f"{word} ({count})"):
                    st.session_state["filter_word"] = word
                    st.cache_data.clear()
                i += 1
            if st.session_state.get("filter_word"):
                st.info(f"Filtered by word: {st.session_state['filter_word']}")
        else:
            st.info("No keywords extracted yet.")

        st.markdown("### Sample Comments (Original + Translated + Summary + Sentiment)")
        sample = filt.head(25) if not filt.empty else df.head(25)
        for idx, r in sample.iterrows():
            polarity = 0.0
            try:
                polarity = TextBlob(r.get('translated','')).sentiment.polarity
            except:
                polarity = 0.0
            mismatch = False
            if r.get('sentiment_bucket') in ['very_positive','positive'] and polarity < -0.1:
                mismatch = True
            elif r.get('sentiment_bucket') in ['very_negative','negative'] and polarity > 0.1:
                mismatch = True
            border_color = 'red' if mismatch else '#ddd'
            st.markdown(
                f"<div style='padding:12px;margin-bottom:8px;border-radius:8px;background:#ffffff;color:black;box-shadow:0 1px 3px rgba(0,0,0,0.1);border:2px solid {border_color}'>"
                f"<div><b>Original:</b> {r.get('original','')}</div>"
                f"<div><b>Translated:</b> {r.get('translated','')}</div>"
                f"<div><b>Summary:</b> {r.get('summary','')}</div>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-top:4px'>"
                f"<div style='width:14px;height:14px;background:{r.get('color','#ddd')};border-radius:3px'></div>"
                f"<div style='font-weight:600'>{r.get('sentiment_bucket','')}</div>"
                f"<div style='color:#666'>(confidence: {r.get('confidence',0):.2f})</div>"
                f"</div></div>", unsafe_allow_html=True
            )

        st.markdown("### Comments (filtered view)")
        show_cols = ["id","original","translated","summary","sentiment_bucket","emoji","confidence","keywords"]
        if not filt.empty:
            st.dataframe(filt[show_cols].rename(columns={"sentiment_bucket":"sentiment"}))
            cid = st.text_input("Enter comment id to view details (or leave blank to pick first shown):")
            if cid.strip() == "" and not filt.empty:
                row = filt.iloc[0]
            elif cid.strip() != "":
                sel = filt[filt["id"]==cid.strip()]
                if sel.empty:
                    st.warning("ID not found in current filter; showing first row.")
                    row = filt.iloc[0]
                else:
                    row = sel.iloc[0]
            else:
                row = None
            if row is not None:
                with st.expander("Comment detail", expanded=True):
                    st.markdown(f"**ID:** {row['id']}")
                    st.markdown("**Original:**")
                    st.write(row["original"])
                    if st.button("Translate (Admin)"):
                        res = call_ml_single(row["id"], row["original"], use_gpu=st.session_state.get("use_gpu_sidebar", False))
                        st.session_state["comments"][row["id"]] = res
                        st.success("Translated and updated.")
                    if st.button("Summarize (Admin)"):
                        res = call_ml_single(row["id"], row["original"], use_gpu=st.session_state.get("use_gpu_sidebar", False))
                        st.session_state["comments"][row["id"]] = res
                        st.success("Summary updated.")
                    st.markdown(f"**Translated:** {row.get('translated','')}")
                    st.markdown(f"**Summary:** {row.get('summary','')}")
                    st.markdown(f"**Sentiment:** {row.get('emoji','')} {row.get('sentiment_bucket','')}  (confidence {row.get('confidence',0):.2f})")
                    st.markdown(f"**Keywords:** {', '.join(row.get('keywords',[])) if isinstance(row.get('keywords',[]), list) else row.get('keywords','')}")
                    if st.button("Mark as Reviewed"):
                        st.session_state["comments"][row["id"]]["reviewed_by"] = st.session_state.get("admin_user","admin")
                        st.session_state["comments"][row["id"]]["reviewed_at"] = datetime.utcnow().isoformat()
                        st.success("Marked reviewed.")
        else:
            st.info("No comments match filter.")

        st.markdown("---")
        st.markdown("### Export / Reports")
        scope = st.selectbox("Export scope", ["All comments", "Filtered comments (current view)"])
        export_df = filt if scope.startswith("Filtered") else df
        if st.button("Download CSV"):
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Click to download CSV", data=csv_bytes, file_name="econsultation_export.csv", mime="text/csv")
        if st.button("Download JSON"):
            json_bytes = export_df.to_json(orient="records", force_ascii=False).encode("utf-8")
            st.download_button("Click to download JSON", data=json_bytes, file_name="econsultation_export.json", mime="application/json")
        if st.button("Download PDF (first 200 rows)"):
            pdf_bytes = generate_pdf_bytes(export_df)
            st.download_button("Click to download PDF", data=pdf_bytes, file_name="econsultation_export.pdf", mime="application/pdf")

# ------------------ Individual mode ------------------
else:
    st.header("Stakeholder / Individual Submission")
    st.markdown("Enter your comment below. You will get immediate analysis (translation, summary, sentiment, keywords).")
    with st.form("submit_form"):
        lang_hint = st.text_input("Language (optional) — leave blank for auto-detect", value="")
        comment_text = st.text_area("Your comment", height=200, help="Write in any language; system will translate to English for analysis.")
        submit = st.form_submit_button("Submit comment")
    if submit:
        if not comment_text.strip():
            st.error("Please enter a comment.")
        else:
            cid = str(uuid.uuid4())
            st.info("Processing your comment...")
            res = call_ml_single(cid, comment_text, use_gpu=st.session_state.get("use_gpu_sidebar", False))
            st.session_state["comments"][cid] = res
            st.success("Processed — here is the analysis (you can download receipt).")
            st.markdown("**Original:**")
            st.write(res["original"])
            st.markdown("**Translated (English):**")
            st.write(res["translated"])
            st.markdown("**Summary:**")
            st.write(res["summary"])
            st.markdown("**Sentiment:**")
            st.write(f"{res['emoji']} {res['sentiment_bucket']} (confidence {res['confidence']*100:.0f}%)")
            st.markdown("**Keywords:**")
            st.write(", ".join(res["keywords"]))
            receipt = json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Download receipt (JSON)", data=receipt, file_name=f"receipt_{cid}.json", mime="application/json")
            st.markdown("_Note: admin will see this comment via real-time feed (demo)_")

# ------------------ Footer / notes ------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Built for SIH — mock ML included. To integrate real ML server:")
st.sidebar.markdown("- Set `USE_ML_API = True` and configure `ML_API_URL` and `API_KEY` in top of app.py")
st.sidebar.markdown("- Ensure your ML exposes `/analyze` (single) and `/analyze/batch` or modify `call_ml_batch` accordingly")
st.sidebar.markdown("© 2024 GovTech. Prototype only.")

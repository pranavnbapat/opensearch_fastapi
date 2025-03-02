# services/language_detect.py

import deepl
import logging
import fasttext
import numpy as np
import os
import time

from dotenv import load_dotenv
from google.cloud import translate_v2 as translate

# Fix FastText's NumPy compatibility
import fasttext.FastText
fasttext.FastText.np.array = lambda obj, **kwargs: np.asarray(obj, **kwargs)  # Force `asarray`

# Load environment variables
load_dotenv()

# FastText Model Path
MODEL_PATH = os.getenv("FASTTEXT_LANGDETECT_MODEL_PATH", "lid.176.bin")

# DeepL & Google Translate API keys
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
USE_GOOGLE_TRANSLATE = os.getenv("USE_GOOGLE_TRANSLATE", "false").lower() == "true"

# Set up logging
logger = logging.getLogger(__name__)

# Load FastText model once
try:
    print(f"Loading FastText language model from: {MODEL_PATH}")
    lang_model = fasttext.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading FastText model: {e}")
    lang_model = None  # Fallback if model fails

# Initialize translation clients
deepl_translator = deepl.Translator(DEEPL_API_KEY) if DEEPL_API_KEY else None
google_client = translate.Client() if USE_GOOGLE_TRANSLATE else None

# DeepL supported languages
DEEPL_SUPPORTED_LANGUAGES = {
    "BG", "CS", "DA", "DE", "EL", "EN", "EN-GB", "EN-US", "ES", "ET", "FI", "FR", "HU", "ID", "IT", "JA", "KO", "LT",
    "LV", "NL", "PL", "PT-BR", "PT-PT", "RO", "RU", "SK", "SL", "SV", "TR", "UK", "ZH"
}

def detect_language(text: str) -> str:
    """Detect the language of a given text using FastText."""
    if not lang_model:
        logger.warning("FastText model is not loaded. Returning 'unknown'.")
        return "unknown"

    try:
        labels, probs = lang_model.predict(text, k=1)
        lang_code = labels[0].replace("__label__", "")

        logger.info(f"Detected language: {lang_code} for text: {text[:30]}...")  # Log first 30 chars for debugging
        return lang_code
    except Exception as e:
        logger.error(f"Error detecting language for text: {text[:30]}... - {e}")
        return "unknown"  # Fallback if detection fails


def translate_text_with_backoff(text, target_language, max_retries=3):
    """Translate text with retry logic using DeepL or Google Translate."""
    if not text or not isinstance(text, str):
        return text

    # Handle language aliases
    if target_language.upper() == "EN":
        target_language = "EN-GB"
    if target_language.upper() == "PT":
        target_language = "PT-PT"

    target_language = target_language.upper()

    for attempt in range(max_retries):
        try:
            if target_language in DEEPL_SUPPORTED_LANGUAGES and deepl_translator:
                return deepl_translate(text, target_language)
            elif google_client:
                return google_translate(text, target_language)
            else:
                logger.warning("No valid translation service available.")
                return text
        except Exception as e:
            logger.warning(f"Translation error (Attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(f"Exceeded max retries for translating {text} to {target_language}")

def deepl_translate(text, target_language, max_retries=3):
    """Translate text using DeepL API with retries."""
    if not deepl_translator:
        raise ValueError("DeepL API key is not set.")

    attempts = 0
    delay = 1

    while attempts < max_retries:
        try:
            result = deepl_translator.translate_text(text, target_lang=target_language, tag_handling="html", ignore_tags=[])
            return result.text
        except (deepl.DeepLException, ValueError) as e:
            logger.error(f"DeepL error: {e}")
            attempts += 1
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    raise RuntimeError(f"Failed to translate using DeepL after {max_retries} attempts.")

def google_translate(text, target_language, max_retries=3):
    """Translate text using Google Translate API with retries."""
    if not google_client:
        raise ValueError("Google Translate API is not enabled.")

    attempts = 0
    delay = 1

    while attempts < max_retries:
        try:
            result = google_client.translate(text, target_language=target_language, format_="html")
            return result["translatedText"]
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            attempts += 1
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    raise RuntimeError(f"Failed to translate using Google Translate after {max_retries} attempts.")

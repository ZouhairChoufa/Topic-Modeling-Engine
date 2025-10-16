# -*- coding: utf-8 -*-
"""
Utility functions for fetching, cleaning, and caching text data.
"""
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import requests
import spacy
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS

# --- Pre-load models and stop words to avoid loading them repeatedly ---
try:
    nlp_fr = spacy.load("fr_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
except IOError:
    print("Error: SpaCy language models not found.")
    print("Please run:")
    print("python -m spacy download fr_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    exit()

combined_stop_words = ENGLISH_STOP_WORDS.union(FRENCH_STOP_WORDS)


def get_text_from_url(url):
    """Fetches and extracts plain text from a URL."""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])
    except requests.RequestException as e:
        print(f"Error while fetching URL {url}: {e}")
        return ""


def fetch_all_texts(urls):
    """Downloads text from a list of URLs in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(get_text_from_url, urls))


def detect_language(text):
    """Detects the language of a given text snippet."""
    try:
        # Use a slice of the text for efficiency
        return detect(text[:500])
    except LangDetectException:
        return 'unknown'


def clean_and_lemmatize_text(text, nlp_fr_model, nlp_en_model, stop_words):
    """Cleans and lemmatizes text using the appropriate language model."""
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)  # Remove non-alphabetic characters
    lang = detect_language(text)
    nlp = nlp_fr_model if lang == 'fr' else nlp_en_model
    doc = nlp(text.lower())
    return ' '.join([
        token.lemma_ for token in doc
        if token.text not in stop_words and not token.is_punct and not token.is_space
    ])


# --- Caching Functions ---
def save_cache(data, filename):
    """Saves data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data cached successfully to {filename}")


def load_cache(filename):
    """Loads data from a pickle file if it exists."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# -*- coding: utf-8 -*-
"""
Functions for building and using TF-IDF and LDA models.
"""
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS)
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS

from .data_processing import clean_and_lemmatize_text, nlp_fr, nlp_en, combined_stop_words


def build_tfidf_model(docs):
    """Builds and returns a TF-IDF vectorizer and the corresponding matrix."""
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def build_lda_model(docs, n_topics=4):
    """Builds an LDA model and its corresponding vectorizer."""
    # Convert frozenset to a list for the stop_words parameter
    stop_words_list = list(combined_stop_words)
    count_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words=stop_words_list)
    count_matrix = count_vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(count_matrix)
    return lda, count_vectorizer


def search_tfidf(query, vectorizer, matrix, urls, docs, top_n=10):
    """Performs a search using a pre-computed TF-IDF model."""
    # Use the same cleaning function as used for the documents
    cleaned_query = clean_and_lemmatize_text(query, nlp_fr, nlp_en, combined_stop_words)
    query_vec = vectorizer.transform([cleaned_query])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()
    # Get top N indices, excluding zero-similarity results
    ranked_indices = cosine_sim.argsort()[::-1]
    results = []
    for i in ranked_indices:
        if cosine_sim[i] > 0 and len(results) < top_n:
            results.append((urls[i], docs[i][:250], cosine_sim[i]))
    return results

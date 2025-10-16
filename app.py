# -*- coding: utf-8 -*-
"""
Main entry point for the Topic Modeling and TF-IDF Search Engine.

This script orchestrates the process of data fetching, processing, model building,
and user interaction.
"""

from src.data_processing import (fetch_all_texts, clean_and_lemmatize_text,
                                 load_cache, save_cache, combined_stop_words,
                                 nlp_fr, nlp_en)
from src.modeling import build_tfidf_model, build_lda_model, search_tfidf
from src.visualization import plot_tfidf_scores, plot_lda_topics

# --- Configuration ---
# URLs to index initially
TARGET_URLS = [
    "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
    "https://fr.wikipedia.org/wiki/Apprentissage_automatique",
    "https://fr.wikipedia.org/wiki/Apprentissage_supervisé",
    "https://fr.wikipedia.org/wiki/Apprentissage_profond",
    "https://fr.wikipedia.org/wiki/Cybersécurité",
    "https://fr.wikipedia.org/wiki/Cybercriminalité",
    "https://fr.wikipedia.org/wiki/Système_d'exploitation",
    "https://fr.wikipedia.org/wiki/Réseau_informatique",
    "https://fr.wikipedia.org/wiki/Programmation_orientée_objet",
    "https://fr.wikipedia.org/wiki/Base_de_données",
    "https://fr.wikipedia.org/wiki/Cloud_computing",
    "https://fr.wikipedia.org/wiki/Big_data",
    "https://fr.wikipedia.org/wiki/Algorithme"
]
CACHE_FILE = "cached_data.pkl"


def main():
    """
    Main function to run the application.
    """
    # --- Data Loading and Indexing ---
    cached_data = load_cache(CACHE_FILE)

    if cached_data:
        print("Loading pre-processed data from cache...")
        documents, urls = cached_data
    else:
        print("Fetching and processing documents from the web...")
        raw_documents = fetch_all_texts(TARGET_URLS)
        successful_urls = [url for i, url in enumerate(TARGET_URLS) if raw_documents[i]]
        raw_documents = [doc for doc in raw_documents if doc]

        print("Cleaning and lemmatizing texts (this may take a moment)...")
        # Pass NLP models and stop words to the processing function
        documents = [
            clean_and_lemmatize_text(doc, nlp_fr, nlp_en, combined_stop_words)
            for doc in raw_documents
        ]

        save_cache((documents, successful_urls), CACHE_FILE)
        urls = successful_urls

    print(f"\n{len(documents)} documents indexed successfully.")

    # --- Model Training ---
    print("Building TF-IDF and LDA models...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_model(documents)
    lda_model, lda_vectorizer = build_lda_model(documents, n_topics=4)
    print("Models are ready. 🚀")

    # --- Interactive Search and Analysis Loop ---
    while True:
        print("\n" + "="*50)
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # 1. Perform TF-IDF search
        print("\n--- TF-IDF Search Results ---")
        tfidf_results = search_tfidf(query, tfidf_vectorizer, tfidf_matrix, urls, documents)

        if not tfidf_results:
            print("No documents matching your query were found.")
            continue

        for url, snippet, score in tfidf_results:
            print(f"URL: {url} (Score: {score:.4f})")
            print(f"   Snippet: {snippet}...\n")

        # 2. Visualize TF-IDF results
        print("\n--- TF-IDF Score Visualization ---")
        plot_tfidf_scores(tfidf_results)

        # 3. Display and visualize LDA topics
        print("\n--- Topic Modeling Results (LDA) ---")
        print("Here are the main topics found across all indexed documents.")
        plot_lda_topics(lda_model, lda_vectorizer)

        # 4. Comparative Analysis
        print("\n--- Comparative Analysis ---")
        print(f"The TF-IDF search found documents specifically relevant to your query: '{query}'.")
        print("The LDA model shows the general themes present in the entire document collection.")
        print("By comparing them, you can see if the most relevant document for your query aligns with one of the main topics.")


if __name__ == "__main__":
    main()

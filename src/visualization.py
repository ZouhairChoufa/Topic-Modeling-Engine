# -*- coding: utf-8 -*-
"""
Functions for creating visualizations of model results.
"""
import os
import matplotlib.pyplot as plt


def plot_tfidf_scores(results):
    """Visualizes TF-IDF search results as a horizontal bar chart."""
    if not results:
        print("No relevant results to display.")
        return
    # Extract data and reverse for correct plotting order (highest on top)
    urls = [os.path.basename(result[0]) for result in reversed(results)]
    scores = [result[2] for result in reversed(results)]
    plt.figure(figsize=(12, 8))
    plt.barh(urls, scores, color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Document')
    plt.title('TF-IDF Similarity Scores for the Query')
    plt.tight_layout()
    plt.show()


def plot_lda_topics(model, vectorizer, n_top_words=10):
    """Visualizes the most important words for each topic from the LDA model."""
    feature_names = vectorizer.get_feature_names_out()
    # Adjust subplot grid to match number of topics
    fig, axes = plt.subplots(model.n_components, 1, figsize=(10, 2.5 * model.n_components), sharex=True)
    # Ensure axes is always iterable
    axes = axes.flatten() if model.n_components > 1 else [axes]

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic #{topic_idx + 1}", fontdict={"fontsize": 14})
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.suptitle("Top Words per Topic (LDA)", fontsize=18)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

"""Visualization utility functions for TopicVista — Discover Hidden Stories in News Data."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models
import umap
from sklearn.manifold import TSNE
import scipy.sparse

def plot_tsne(tfidf_matrix, cluster_labels, labels, filename='kmeans_clusters_tsne.png', show=True):
    """Visualize clusters using t-SNE and save the plot."""
    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    n = tsne_results.shape[0]
    if not (len(cluster_labels) == len(labels) == n):
        print(f"[ERROR] Length mismatch in plot_tsne: tsne_results={n}, cluster_labels={len(cluster_labels)}, labels={len(labels)}. Skipping t-SNE plot.")
        return None
    df_tsne = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'cluster': cluster_labels,
        'category': labels
    })
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_tsne['x'], df_tsne['y'], c=df_tsne['cluster'], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('K-means Clusters Visualization (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return df_tsne

def plot_umap(tfidf_matrix, cluster_labels, labels, filename='kmeans_clusters_umap.png', show=True):
    """Visualize clusters using UMAP and save the plot."""
    print("Creating UMAP visualization...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(tfidf_matrix.toarray())
    umap_results = np.asarray(umap_results)
    n = umap_results.shape[0]
    if not (len(cluster_labels) == len(labels) == n):
        print(f"[ERROR] Length mismatch in plot_umap: umap_results={n}, cluster_labels={len(cluster_labels)}, labels={len(labels)}. Skipping UMAP plot.")
        return None
    df_umap = pd.DataFrame({
        'x': umap_results[:, 0],
        'y': umap_results[:, 1],
        'cluster': cluster_labels,
        'category': labels
    })
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_umap['x'], df_umap['y'], c=df_umap['cluster'], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('K-means Clusters Visualization (UMAP)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return df_umap

def create_wordclouds(lda_sklearn, count_vectorizer, n_topics=20, filename='topic_wordclouds.png', show=True):
    """Create word clouds for each topic and save the figure."""
    print("Creating word clouds...")
    feature_names = count_vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()
    for topic_idx in range(min(n_topics, 20)):
        topic = lda_sklearn.components_[topic_idx]
        top_words_idx = topic.argsort()[-50:][::-1]
        word_freq = {feature_names[idx]: topic[idx] for idx in top_words_idx}
        wordcloud = WordCloud(width=300, height=200, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
        axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
        axes[topic_idx].set_title(f'Topic {topic_idx}')
        axes[topic_idx].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def create_pyldavis_visualization(lda_model, dictionary, corpus, filename='lda_visualization.html'):
    """Create interactive LDA visualization and save as HTML."""
    print("Creating pyLDAvis visualization...")
    try:
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, filename)
        print(f"LDA visualization saved as '{filename}'")
        return vis
    except Exception as e:
        print(f"Error creating pyLDAvis visualization: {e}")
        return None

def create_enhanced_pyldavis(lda_model, dictionary, corpus, filename='enhanced_lda_visualization.html'):
    """Create enhanced pyLDAvis visualization with additional features and save as HTML."""
    print("Creating enhanced pyLDAvis visualization...")
    try:
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, filename)
        coherence_scores = []
        topics = lda_model.show_topics(num_topics=lda_model.num_topics, formatted=False)
        for topic_id, topic_words in topics:
            topic_tokens = [word for word, _ in topic_words]
            coherence_scores.append({
                'topic_id': topic_id,
                'top_words': ', '.join(topic_tokens[:5]),
                'word_count': len(topic_tokens)
            })
        doc_topic_distributions = []
        for doc_bow in corpus:
            topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
            doc_topic_distributions.append([prob for _, prob in topic_dist])
        topic_dominance = np.array(doc_topic_distributions)
        topic_avg_probability = np.mean(topic_dominance, axis=0)
        print(f"Enhanced LDA visualization saved as '{filename}'")
        return {
            'visualization': vis,
            'coherence_info': coherence_scores,
            'topic_avg_probability': topic_avg_probability,
            'doc_topic_distributions': doc_topic_distributions
        }
    except Exception as e:
        print(f"Error creating enhanced pyLDAvis visualization: {e}")
        return None 
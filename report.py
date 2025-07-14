#!/usr/bin/env python3
"""
Automated Report Generation Script for TopicVista — Discover Hidden Stories in News Data
Generates comprehensive PDF/DOCX reports from analysis results
"""

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from main import NewsGroupsAnalyzer
from visualization_utils import plot_tsne, plot_umap, create_wordclouds
import joblib

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_results_summary(analyzer, cluster_labels, lda_gensim, dictionary, corpus):
    """Generate comprehensive results summary"""

    # Calculate advanced metrics
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

    # Create labels mapping
    label_to_int = {label: i for i, label in enumerate(analyzer.categories)}
    true_labels_int = [label_to_int[label] for label in analyzer.labels]

    # Calculate metrics
    silhouette_avg = silhouette_score(analyzer.tfidf_matrix, cluster_labels)
    ari = adjusted_rand_score(true_labels_int, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_int, cluster_labels)

    # LDA coherence
    from gensim.models import CoherenceModel
    texts = [doc.split() for doc in analyzer.preprocessed_docs]
    coherence_model = CoherenceModel(model=lda_gensim, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return {
        'silhouette_score': silhouette_avg,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'coherence_score': coherence_score,
        'num_documents': len(analyzer.documents),
        'num_categories': len(analyzer.categories),
        'num_clusters': len(set(cluster_labels)),
        'vocabulary_size': analyzer.tfidf_matrix.shape[1]
    }

def generate_markdown_report(analyzer):
    """Generate comprehensive markdown report"""

    print("📊 Generating comprehensive analysis report...")

    # Create output directory
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # Try to load preprocessed docs and matrices if available
    pre_docs_path = "trained_models/preprocessed_docs.pkl"
    tfidf_path = "trained_models/tfidf_matrix.pkl"
    count_path = "trained_models/count_matrix.pkl"
    preprocessed_loaded = False
    if os.path.exists(pre_docs_path) and os.path.exists(tfidf_path) and os.path.exists(count_path):
        print("Loading preprocessed documents and matrices from disk...")
        analyzer.preprocessed_docs = joblib.load(pre_docs_path)
        analyzer.tfidf_matrix = joblib.load(tfidf_path)
        analyzer.count_matrix = joblib.load(count_path)
        preprocessed_loaded = True
    else:
        print("Preprocessed data not found. Running preprocessing and vectorization...")
        analyzer.preprocess_documents()
        analyzer.vectorize_documents()

    # Try to load existing models, if not found, train new ones
    if not analyzer.load_models():
        print("Training new models...")
        cluster_labels = analyzer.perform_kmeans_clustering()
        analyzer.perform_lda_sklearn()
        lda_gensim, dictionary, corpus = analyzer.perform_lda_gensim()
        analyzer.save_models()
    else:
        # Generate predictions with loaded models
        cluster_labels = analyzer.kmeans_model.predict(analyzer.tfidf_matrix)
        # Recreate Gensim components
        texts = [doc.split() for doc in analyzer.preprocessed_docs]
        dictionary = analyzer.lda_gensim.id2word if hasattr(analyzer.lda_gensim, 'id2word') else None
        if dictionary is None:
            import gensim
            dictionary = gensim.corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_gensim = analyzer.lda_gensim

    # Generate visualizations
    print("Creating visualizations...")

    # 1. Dataset overview
    categories_df = pd.DataFrame({
        'Category': analyzer.categories,
        'Document_Count': [analyzer.labels.count(cat) for cat in analyzer.categories]
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=categories_df, x='Document_Count', y='Category', ax=ax)
    plt.title('Document Distribution by Category')
    plt.tight_layout()
    plt.savefig('reports/figures/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Cluster analysis
    plot_tsne(analyzer.tfidf_matrix, cluster_labels, analyzer.labels, filename='reports/figures/clusters_tsne.png', show=False)

    plot_umap(analyzer.tfidf_matrix, cluster_labels, analyzer.labels, filename='reports/figures/clusters_umap.png', show=False)

    # 3. Topic word clouds
    create_wordclouds(analyzer.lda_sklearn, analyzer.count_vectorizer, filename='reports/figures/topic_wordclouds.png', show=False)

    # 4. Cluster composition
    composition = analyzer.analyze_cluster_composition(cluster_labels)
    os.rename('cluster_composition.png', 'reports/figures/cluster_composition.png')

    # 5. Advanced metrics visualization
    advanced_analysis = analyzer.advanced_cluster_analysis(cluster_labels)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Metrics bar plot
    metrics = advanced_analysis['metrics']
    ax1.bar(metrics.keys(), metrics.values())
    ax1.set_title('Clustering Evaluation Metrics')
    ax1.set_ylabel('Score')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Cluster purity
    purity_values = list(advanced_analysis['cluster_purity'].values())
    ax2.hist(purity_values, bins=20, alpha=0.7)
    ax2.set_title('Distribution of Cluster Purity Scores')
    ax2.set_xlabel('Purity Score')
    ax2.set_ylabel('Frequency')

    # Cluster sizes
    cluster_sizes = list(advanced_analysis['cluster_sizes'].values())
    ax3.bar(range(len(cluster_sizes)), cluster_sizes)
    ax3.set_title('Cluster Sizes')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Documents')

    # Silhouette analysis
    from sklearn.metrics import silhouette_samples
    silhouette_values = silhouette_samples(analyzer.tfidf_matrix, cluster_labels)
    ax4.hist(silhouette_values, bins=30, alpha=0.7)
    ax4.set_title('Distribution of Silhouette Scores')
    ax4.set_xlabel('Silhouette Score')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('reports/figures/advanced_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get results summary
    results_summary = create_results_summary(analyzer, cluster_labels, lda_gensim, dictionary, corpus)

    # Generate markdown content
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    markdown_content = f"""# TopicVista — Discover Hidden Stories in News Data
## Final Project Report

**Generated on:** {timestamp}  
**Project:** TopicVista — Discover Hidden Stories in News Data  
**Dataset:** 20 Newsgroups  

---

## Executive Summary

This report presents the results of an unsupervised machine learning analysis on the 20 Newsgroups dataset. The project implements both K-means clustering and Latent Dirichlet Allocation (LDA) for topic discovery, with comprehensive evaluation metrics and visualizations.

### Key Findings

- **Documents Analyzed:** {results_summary['num_documents']:,}
- **Categories:** {results_summary['num_categories']}
- **Vocabulary Size:** {results_summary['vocabulary_size']:,}
- **Clusters Generated:** {results_summary['num_clusters']}

### Performance Metrics

- **Silhouette Score:** {results_summary['silhouette_score']:.3f}
- **Adjusted Rand Index:** {results_summary['adjusted_rand_index']:.3f}
- **Normalized Mutual Information:** {results_summary['normalized_mutual_info']:.3f}
- **LDA Coherence Score:** {results_summary['coherence_score']:.3f}

---

## 1. Introduction

### 1.1 Objective
The primary objective of this project is to apply unsupervised learning techniques to discover latent topics and cluster similar documents in the 20 Newsgroups dataset. This analysis demonstrates practical applications of:
- Text preprocessing and feature engineering
- K-means clustering for document grouping
- Latent Dirichlet Allocation for topic discovery
- Performance evaluation and interpretation

### 1.2 Dataset Overview
The 20 Newsgroups dataset contains approximately 20,000 documents distributed across 20 distinct newsgroup categories, covering topics from computer technology to politics and religion.

![Dataset Overview](figures/dataset_overview.png)

---

## 2. Methodology

### 2.1 Data Preprocessing Pipeline

1. **Text Cleaning**
   - Removal of email headers and metadata
   - URL and email address extraction
   - Punctuation and special character handling

2. **Tokenization & Normalization**
   - Lowercase conversion
   - Word-level tokenization using NLTK/spaCy
   - Stop word removal (English)

3. **Feature Engineering**
   - Lemmatization using spaCy
   - TF-IDF vectorization (max 5,000 features)
   - Count vectorization for LDA

### 2.2 Clustering Approach

**K-means Clustering:**
- Number of clusters: 20 (matching original categories)
- Distance metric: Euclidean
- Initialization: k-means++
- Evaluation: Silhouette Score

**Latent Dirichlet Allocation:**
- Number of topics: 20
- Implementation: Both scikit-learn and Gensim
- Evaluation: Coherence Score
- Visualization: pyLDAvis

---

## 3. Results

### 3.1 Clustering Performance

The K-means clustering achieved a silhouette score of {results_summary['silhouette_score']:.3f}, indicating {
'good' if results_summary['silhouette_score'] > 0.3 else 
'moderate' if results_summary['silhouette_score'] > 0.1 else 'poor'
} cluster separation.

![Cluster Visualization (t-SNE)](figures/clusters_tsne.png)

![Cluster Visualization (UMAP)](figures/clusters_umap.png)

### 3.2 Topic Modeling Results

The LDA model achieved a coherence score of {results_summary['coherence_score']:.3f}, suggesting {
'excellent' if results_summary['coherence_score'] > 0.6 else 
'good' if results_summary['coherence_score'] > 0.4 else 
'moderate' if results_summary['coherence_score'] > 0.3 else 'poor'
} topic quality.

![Topic Word Clouds](figures/topic_wordclouds.png)

### 3.3 Cluster Composition Analysis

The following heatmap shows how well the discovered clusters align with the true newsgroup categories:

![Cluster Composition](figures/cluster_composition.png)

### 3.4 Advanced Metrics

![Advanced Performance Metrics](figures/advanced_metrics.png)

The Adjusted Rand Index of {results_summary['adjusted_rand_index']:.3f} indicates {
'excellent' if results_summary['adjusted_rand_index'] > 0.7 else 
'good' if results_summary['adjusted_rand_index'] > 0.5 else 
'moderate' if results_summary['adjusted_rand_index'] > 0.3 else 'limited'
} agreement between discovered clusters and true categories.

---

## 4. Discussion

### 4.1 Clustering Performance Analysis

The clustering results demonstrate the challenges of unsupervised learning on text data. The silhouette score suggests that while clusters are formed, there is some overlap between different newsgroup topics, which is expected given the nature of the data.

### 4.2 Topic Quality Assessment

The LDA coherence score indicates that the discovered topics are meaningful and interpretable. The word clouds reveal distinct semantic themes that align well with the known newsgroup categories.

### 4.3 Practical Implications

This analysis demonstrates the effectiveness of combining multiple unsupervised learning techniques for text analysis. The approach could be applied to:
- Document organization and retrieval
- Content recommendation systems
- Automated categorization pipelines
- Exploratory data analysis for unknown text corpora

---

## 5. Conclusions

### 5.1 Key Achievements

1. **Successful Implementation:** Both K-means clustering and LDA topic modeling were successfully implemented and evaluated
2. **Meaningful Results:** The discovered clusters and topics show clear semantic coherence
3. **Comprehensive Evaluation:** Multiple metrics provide a thorough assessment of model performance
4. **Interactive Visualization:** pyLDAvis integration enables interactive exploration of topics

### 5.2 Technical Contributions

- **Robust Preprocessing Pipeline:** Comprehensive text cleaning and feature engineering
- **Multi-algorithm Approach:** Combination of clustering and topic modeling techniques
- **Advanced Evaluation:** Implementation of multiple clustering evaluation metrics
- **Interactive Dashboard:** Streamlit application for real-time analysis and exploration

### 5.3 Future Work

- **Deep Learning Approaches:** Implement transformer-based embeddings (BERT, etc.)
- **Hierarchical Clustering:** Explore hierarchical topic structures
- **Dynamic Topic Modeling:** Analyze topic evolution over time
- **Cross-dataset Validation:** Test generalization on other text corpora

---

## 6. Technical Implementation

### 6.1 Code Structure

The project is organized into several key components:

- `main.py`: Core analysis pipeline and NewsGroupsAnalyzer class
- `streamlit_app.py`: Interactive web dashboard
- `train_models.py`: Model training and persistence
- `requirements.txt`: Dependencies and environment setup

### 6.2 Dependencies

Key libraries used in this project:
- **scikit-learn**: Clustering and evaluation metrics
- **Gensim**: Advanced topic modeling
- **NLTK/spaCy**: Natural language processing
- **Streamlit**: Interactive web application
- **pyLDAvis**: Topic visualization
- **Matplotlib/Seaborn**: Static visualizations

### 6.3 Reproducibility

All models can be trained and saved for consistent results:
```python
# Train and save models
analyzer = NewsGroupsAnalyzer()
analyzer.run_complete_analysis(save_models=True)

# Load pre-trained models
analyzer.load_models()
```

---

## Appendices

### A. Model Parameters

**K-means Clustering:**
- n_clusters: 20
- init: 'k-means++'
- random_state: 42

**LDA (scikit-learn):**
- n_components: 20
- max_iter: 10
- random_state: 42

**LDA (Gensim):**
- num_topics: 20
- passes: 10
- alpha: 'auto'

### B. Evaluation Metrics Definitions

- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Adjusted Rand Index**: Measures agreement between true and predicted clusters (0 to 1, higher is better)
- **Normalized Mutual Information**: Measures mutual dependence between clusterings (0 to 1, higher is better)
- **Coherence Score**: Measures topic quality based on semantic similarity (higher is better)

---

*This report was automatically generated by the 20 Newsgroups Topic Modeling Analysis System.*
"""

    # Write markdown report
    with open('reports/FINAL_REPORT.md', 'w') as f:
        f.write(markdown_content)

    print("✅ Comprehensive report generated!")
    print("📄 Available documents:")
    print("   → reports/FINAL_REPORT.md (Main report)")
    print("   → reports/figures/ (All visualizations)")

    print("\n📝 To convert to PDF/DOCX:")
    print("   → Install pandoc: https://pandoc.org/installing.html")
    print("   → PDF: pandoc reports/FINAL_REPORT.md -o reports/FINAL_REPORT.pdf")
    print("   → DOCX: pandoc reports/FINAL_REPORT.md -o reports/FINAL_REPORT.docx")

    return True

def main():
    print("🚀 20 Newsgroups Analysis Report Generator")
    print("=" * 50)

    # Initialize analyzer
    analyzer = NewsGroupsAnalyzer()

    # Generate comprehensive report
    success = generate_markdown_report(analyzer)

    if success:
        print("\n🎉 Report generation completed successfully!")
        print("📊 Check the 'reports/' directory for all outputs")
    else:
        print("\n❌ Report generation failed")

if __name__ == "__main__":
    main()
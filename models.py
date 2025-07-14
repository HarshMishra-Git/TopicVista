
#!/usr/bin/env python3
"""
Training script to train and save all models for TopicVista — Discover Hidden Stories in News Data
Run this once to train all models, then use the Streamlit app with pre-trained models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

from main import NewsGroupsAnalyzer
from visualization_utils import *
from text_utils import *

def main():
    """Train and save all models for the 20 Newsgroups analysis."""
    print("=== TopicVista — Discover Hidden Stories in News Data Model Training ===")
    print("This will train and save all models for later use in the TopicVista app\n")
    
    # Initialize analyzer
    analyzer = NewsGroupsAnalyzer()
    
    # Run complete training pipeline
    print("Starting training pipeline...")
    results = analyzer.run_complete_analysis(save_models=True)

    # Save preprocessed docs and matrices
    import joblib
    joblib.dump(analyzer.preprocessed_docs, "trained_models/preprocessed_docs.pkl")
    joblib.dump(analyzer.tfidf_matrix, "trained_models/tfidf_matrix.pkl")
    joblib.dump(analyzer.count_matrix, "trained_models/count_matrix.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All models have been trained and saved to 'trained_models/' directory")
    print("\nTrained models:")
    print("- K-means clustering model")
    print("- LDA (sklearn) topic model") 
    print("- LDA (Gensim) topic model")
    print("- TF-IDF vectorizer")
    print("- Count vectorizer")
    print("- Model metadata")
    
    print("\nGenerated files:")
    if os.path.exists("kmeans_clusters_tsne.png"):
        print("- kmeans_clusters_tsne.png")
    if os.path.exists("kmeans_clusters_umap.png"):
        print("- kmeans_clusters_umap.png") 
    if os.path.exists("topic_wordclouds.png"):
        print("- topic_wordclouds.png")
    if os.path.exists("cluster_composition.png"):
        print("- cluster_composition.png")
    if os.path.exists("lda_visualization.html"):
        print("- lda_visualization.html")
    if os.path.exists("enhanced_lda_visualization.html"):
        print("- enhanced_lda_visualization.html")
        
    print(f"\nDataset info:")
    print(f"- Total documents: {len(analyzer.documents):,}")
    print(f"- Categories: {len(analyzer.categories)}")
    tfidf_shape = getattr(analyzer.tfidf_matrix, 'shape', None)
    if tfidf_shape and not isinstance(tfidf_shape, (str, tuple)):
        print(f"- TF-IDF matrix shape: {tfidf_shape}")
    else:
        print(f"- TF-IDF matrix shape: N/A")
    
    print("\n🚀 You can now run the TopicVista app which will load these pre-trained models!")
    print("Command: streamlit run app.py --server.port 5000 --server.address 0.0.0.0")

if __name__ == "__main__":
    main()

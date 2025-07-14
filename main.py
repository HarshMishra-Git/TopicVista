
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import joblib

# NLP Libraries
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap

# Topic modeling and visualization
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

from visualization_utils import *
from text_utils import *

class NewsGroupsAnalyzer:
    def __init__(self, data_path="twenty+newsgroups/20_newsgroups"):
        self.data_path = data_path
        self.documents = []
        self.labels = []
        self.categories = []
        self.preprocessed_docs = []
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.kmeans_model = None
        self.lda_sklearn = None
        self.lda_gensim = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_documents(self):
        """Load all documents from the mini_newsgroups dataset"""
        print("Loading documents...")
        
        categories = os.listdir(self.data_path)
        categories = [cat for cat in categories if os.path.isdir(os.path.join(self.data_path, cat))]
        self.categories = sorted(categories)
        
        for category in self.categories:
            category_path = os.path.join(self.data_path, category)
            files = os.listdir(category_path)
            
            for file in files:
                file_path = os.path.join(category_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self.documents.append(content)
                        self.labels.append(category)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} documents from {len(self.categories)} categories")
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove headers (lines starting with specific patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        header_patterns = ['From:', 'Subject:', 'Date:', 'Organization:', 'Lines:', 'Nntp-Posting-Host:', 'X-']
        
        for line in lines:
            # Skip header lines
            if any(line.startswith(pattern) for pattern in header_patterns):
                continue
            # Skip quoted lines (starting with >)
            if line.strip().startswith('>'):
                continue
            # Skip signature lines (starting with --)
            if line.strip().startswith('--'):
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove non-alphabetic characters and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_documents(self, batch_size=100):
        """Preprocess all documents in batches for performance."""
        print("Preprocessing documents...")
        from joblib import Parallel, delayed
        def process_doc(doc):
            cleaned = self.clean_text(doc)
            cleaned = cleaned.lower()
            if nlp:
                doc_nlp = nlp(cleaned)
                tokens = [token.lemma_ for token in doc_nlp if not token.is_stop and not token.is_punct and len(token.text) > 2]
            else:
                tokens = word_tokenize(cleaned)
                tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            return ' '.join(tokens)
        prefer_mode = "threads" if nlp else "processes"
        self.preprocessed_docs = Parallel(n_jobs=-1, prefer=prefer_mode)(delayed(process_doc)(doc) for doc in self.documents)
        print("Preprocessing completed")
    
    def vectorize_documents(self):
        """Convert documents to numerical vectors"""
        print("Vectorizing documents...")
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_docs)
        
        # Count Vectorization for LDA
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1)
        )
        self.count_matrix = self.count_vectorizer.fit_transform(self.preprocessed_docs)
        
        print(f"TF-IDF matrix shape: {getattr(self.tfidf_matrix, 'shape', 'Unknown')}")
        print(f"Count matrix shape: {getattr(self.count_matrix, 'shape', 'Unknown')}")
    
    def perform_kmeans_clustering(self, n_clusters=20):
        """Perform K-means clustering"""
        print("Performing K-means clustering...")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="10")
        cluster_labels = self.kmeans_model.fit_predict(self.tfidf_matrix)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.tfidf_matrix, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return cluster_labels
    
    def perform_lda_sklearn(self, n_topics=20):
        """Perform LDA using sklearn"""
        print("Performing LDA with sklearn...")
        
        self.lda_sklearn = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        self.lda_sklearn.fit(self.count_matrix)
        
        return self.lda_sklearn
    
    def perform_lda_gensim(self, n_topics=20):
        """Perform LDA using Gensim"""
        print("Performing LDA with Gensim...")
        
        # Prepare corpus for Gensim
        texts = [doc.split() for doc in self.preprocessed_docs if doc is not None]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Build LDA model
        self.lda_gensim = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=self.lda_gensim,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score: {coherence_score:.3f}")
        
        return self.lda_gensim, dictionary, corpus
    
    def display_top_words_sklearn(self, n_words=10):
        """Display top words for each topic (sklearn LDA)"""
        if self.count_vectorizer is None:
            print("Count vectorizer not loaded.")
            return
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        print("\nTop words per topic (sklearn LDA):")
        if self.lda_sklearn is None or not hasattr(self.lda_sklearn, 'components_'):
            print("LDA sklearn model not loaded.")
            return
        for topic_idx, topic in enumerate(self.lda_sklearn.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [str(feature_names[i]) for i in top_words_idx]
            print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    def display_top_words_gensim(self, n_words=10):
        """Display top words for each topic (Gensim LDA)"""
        if self.lda_gensim is None or not hasattr(self.lda_gensim, 'num_topics'):
            print("Gensim LDA model not loaded.")
            return
        for topic_idx in range(self.lda_gensim.num_topics):
            if not hasattr(self.lda_gensim, 'show_topic'):
                print("Gensim LDA model missing 'show_topic' method.")
                continue
            topic_words = self.lda_gensim.show_topic(topic_idx, topn=n_words)
            words = [str(word) for word, _ in topic_words]
            print(f"Topic {topic_idx}: {', '.join(words)}")
    
    def analyze_cluster_composition(self, cluster_labels):
        """Analyze the composition of clusters vs true categories"""
        df = pd.DataFrame({
            'true_category': self.labels,
            'cluster': cluster_labels
        })
        
        # Create confusion matrix-like analysis
        composition = pd.crosstab(df['cluster'], df['true_category'])
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(composition, annot=True, fmt='d', cmap='Blues')
        plt.title('Cluster Composition by True Categories')
        plt.xlabel('True Categories')
        plt.ylabel('Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('cluster_composition.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return composition
    
    def advanced_cluster_analysis(self, cluster_labels):
        """Advanced cluster composition analysis with metrics"""
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
        
        # Create labels mapping
        label_to_int = {label: i for i, label in enumerate(self.categories)}
        true_labels_int = [label_to_int[label] for label in self.labels]
        
        # Calculate advanced metrics
        metrics = {
            'Adjusted Rand Index': adjusted_rand_score(true_labels_int, cluster_labels),
            'Normalized Mutual Info': normalized_mutual_info_score(true_labels_int, cluster_labels),
            'Homogeneity Score': homogeneity_score(true_labels_int, cluster_labels),
            'Completeness Score': completeness_score(true_labels_int, cluster_labels)
        }
        
        # Cluster purity analysis
        df = pd.DataFrame({
            'true_category': self.labels,
            'cluster': cluster_labels
        })
        
        cluster_purity = {}
        cluster_sizes = {}
        dominant_categories = {}
        
        for cluster_id in range(len(set(cluster_labels))):
            cluster_docs = df[df['cluster'] == cluster_id]
            # Ensure 'true_category' is a Series
            true_cat_series = pd.Series(cluster_docs['true_category'])
            category_counts = true_cat_series.value_counts()
            
            cluster_sizes[cluster_id] = len(cluster_docs)
            dominant_categories[cluster_id] = category_counts.index[0] if len(category_counts) > 0 else 'Unknown'
            cluster_purity[cluster_id] = category_counts.iloc[0] / len(cluster_docs) if len(cluster_docs) > 0 else 0
        
        return {
            'metrics': metrics,
            'cluster_purity': cluster_purity,
            'cluster_sizes': cluster_sizes,
            'dominant_categories': dominant_categories,
            'composition_df': df
        }
    
    def save_models(self, models_dir="trained_models"):
        """Save all trained models to disk"""
        print(f"Saving trained models to {models_dir}...")
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Save sklearn models using joblib (recommended for sklearn)
        if self.kmeans_model is not None:
            joblib.dump(self.kmeans_model, os.path.join(models_dir, "kmeans_model.pkl"))
            print("✓ K-means model saved")
        
        if self.lda_sklearn is not None:
            joblib.dump(self.lda_sklearn, os.path.join(models_dir, "lda_sklearn_model.pkl"))
            print("✓ Sklearn LDA model saved")
        
        # Save vectorizers
        if self.tfidf_vectorizer is not None:
            joblib.dump(self.tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
            print("✓ TF-IDF vectorizer saved")
        
        if self.count_vectorizer is not None:
            joblib.dump(self.count_vectorizer, os.path.join(models_dir, "count_vectorizer.pkl"))
            print("✓ Count vectorizer saved")
        
        # Save Gensim LDA model (if exists)
        if self.lda_gensim is not None:
            self.lda_gensim.save(os.path.join(models_dir, "lda_gensim_model"))
            print("✓ Gensim LDA model saved")
        
        # Save model metadata
        metadata = {
            'num_documents': len(self.documents),
            'num_categories': len(self.categories),
            'categories': self.categories,
            'tfidf_shape': getattr(self.tfidf_matrix, 'shape', None),
            'count_shape': getattr(self.count_matrix, 'shape', None),
            'num_clusters': self.kmeans_model.n_clusters if self.kmeans_model is not None else None,
            'num_topics': self.lda_sklearn.n_components if self.lda_sklearn is not None else None
        }
        
        with open(os.path.join(models_dir, "model_metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        print("✓ Model metadata saved")
        
        print(f"\nAll models saved successfully in '{models_dir}/' directory!")
    
    def load_models(self, models_dir="trained_models"):
        """Load all trained models from disk"""
        print(f"Loading trained models from {models_dir}...")
        
        if not os.path.exists(models_dir):
            print(f"Models directory '{models_dir}' not found!")
            return False
        
        try:
            # Load sklearn models
            kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
            if os.path.exists(kmeans_path):
                self.kmeans_model = joblib.load(kmeans_path)
                print("✓ K-means model loaded")
            
            lda_sklearn_path = os.path.join(models_dir, "lda_sklearn_model.pkl")
            if os.path.exists(lda_sklearn_path):
                self.lda_sklearn = joblib.load(lda_sklearn_path)
                print("✓ Sklearn LDA model loaded")
            
            # Load vectorizers
            tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                print("✓ TF-IDF vectorizer loaded")
            
            count_path = os.path.join(models_dir, "count_vectorizer.pkl")
            if os.path.exists(count_path):
                self.count_vectorizer = joblib.load(count_path)
                print("✓ Count vectorizer loaded")
            
            # Load Gensim LDA model
            gensim_path = os.path.join(models_dir, "lda_gensim_model")
            if os.path.exists(gensim_path):
                self.lda_gensim = models.LdaModel.load(gensim_path)
                print("✓ Gensim LDA model loaded")
            
            # Load metadata
            metadata_path = os.path.join(models_dir, "model_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print("✓ Model metadata loaded")
                print(f"  - Documents: {metadata.get('num_documents', 'N/A')}")
                print(f"  - Categories: {metadata.get('num_categories', 'N/A')}")
                print(f"  - TF-IDF matrix: {metadata.get('tfidf_shape', 'N/A')}")
                print(f"  - Clusters: {metadata.get('num_clusters', 'N/A')}")
                print(f"  - Topics: {metadata.get('num_topics', 'N/A')}")
            
            print(f"\nModels loaded successfully from '{models_dir}/' directory!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def run_complete_analysis(self, save_models=True):
        """Run the complete analysis pipeline"""
        print("Starting complete NLP analysis...")
        
        # Step 1: Load documents
        self.load_documents()
        
        # Step 2: Preprocess documents
        self.preprocess_documents()
        
        # Step 3: Vectorize documents
        self.vectorize_documents()
        
        # Step 4: K-means clustering
        cluster_labels = self.perform_kmeans_clustering()
        
        # Step 5: LDA topic modeling
        self.perform_lda_sklearn()
        lda_gensim, dictionary, corpus = self.perform_lda_gensim()
        
        # Step 6: Save trained models (optional)
        if save_models:
            self.save_models()
        
        # Step 7: Visualizations (now using utility functions)
        plot_tsne(self.tfidf_matrix, cluster_labels, self.labels)
        plot_umap(self.tfidf_matrix, cluster_labels, self.labels)
        
        # Step 8: Display results
        self.display_top_words_sklearn()
        self.display_top_words_gensim()
        
        # Step 9: Create word clouds
        create_wordclouds(self.lda_sklearn, self.count_vectorizer)
        
        # Step 10: Interactive visualization
        create_pyldavis_visualization(lda_gensim, dictionary, corpus)
        
        # Step 11: Analyze cluster composition
        composition = self.analyze_cluster_composition(cluster_labels)
        
        print("\nAnalysis completed!")
        return {
            'cluster_labels': cluster_labels,
            'lda_sklearn': self.lda_sklearn,
            'lda_gensim': lda_gensim,
            'composition': composition
        }

if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = NewsGroupsAnalyzer()
    results = analyzer.run_complete_analysis()

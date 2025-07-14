import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import NewsGroupsAnalyzer
import pickle
import os
import streamlit.components.v1 as components
import gensim
from visualization_utils import *
from text_utils import *
import joblib

st.set_page_config(page_title="TopicVista — Discover Hidden Stories in News Data", layout="wide")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_analyzer():
    """Load the analyzer with pre-trained models. Only preprocess/vectorize if models are not found."""
    analyzer = NewsGroupsAnalyzer()
    analyzer.load_documents()
    models_loaded = analyzer.load_models()
    if not models_loaded:
        st.warning("Pre-trained models not found. Running preprocessing and training pipeline. This may take a while.")
        analyzer.preprocess_documents()
        analyzer.vectorize_documents()
        analyzer.save_models()
        analyzer.load_models()
        # Save preprocessed docs and matrices
        joblib.dump(analyzer.preprocessed_docs, "trained_models/preprocessed_docs.pkl")
        joblib.dump(analyzer.tfidf_matrix, "trained_models/tfidf_matrix.pkl")
        joblib.dump(analyzer.count_matrix, "trained_models/count_matrix.pkl")
    else:
        # Load preprocessed docs and matrices
        analyzer.preprocessed_docs = joblib.load("trained_models/preprocessed_docs.pkl")
        analyzer.tfidf_matrix = joblib.load("trained_models/tfidf_matrix.pkl")
        analyzer.count_matrix = joblib.load("trained_models/count_matrix.pkl")
    return analyzer

def predict_topic(text, analyzer):
    """Predict the topic of input text"""
    # Import nlp from main module
    from main import nlp

    # Preprocess the input text
    cleaned_text = analyzer.clean_text(text)

    if nlp is not None:
        doc_nlp = nlp(cleaned_text.lower())
        tokens = [token.lemma_ for token in doc_nlp 
                 if not token.is_stop and not token.is_punct and len(token.text) > 2]
    else:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(cleaned_text.lower())
        tokens = [word for word in tokens 
                 if word not in analyzer.stop_words and len(word) > 2]

    preprocessed_text = ' '.join(tokens)

    # Transform using the fitted vectorizer
    text_tfidf = analyzer.tfidf_vectorizer.transform([preprocessed_text])
    text_count = analyzer.count_vectorizer.transform([preprocessed_text])

    # Predict cluster
    cluster = analyzer.kmeans_model.predict(text_tfidf)[0] if analyzer.kmeans_model else None

    # Get topic probabilities from LDA
    topic_probs = analyzer.lda_sklearn.transform(text_count)[0] if analyzer.lda_sklearn else None

    return cluster, topic_probs

def main():
    st.title("🔍 TopicVista — Discover Hidden Stories in News Data")
    st.markdown("---")

    # Sidebar for navigation and parameters
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Overview", 
        "Topic Prediction", 
        "Cluster Analysis", 
        "Topic Visualization",
        "Advanced Analytics",
        "Interactive LDA",
        "Dataset Statistics"
    ], help="Select a section to explore different analyses and visualizations.")

    # Extensibility: Allow user to select number of clusters/topics
    n_clusters = st.sidebar.number_input("Number of Clusters (K-means)", min_value=2, max_value=40, value=20, step=1, help="Set the number of clusters for K-means clustering. Used if retraining is needed.")
    n_topics = st.sidebar.number_input("Number of Topics (LDA)", min_value=2, max_value=40, value=20, step=1, help="Set the number of topics for LDA topic modeling. Used if retraining is needed.")

    # Load analyzer with pre-trained models
    with st.spinner("Loading pre-trained models..."):
        analyzer = load_analyzer()

    # Show model status
    st.success("✅ Pre-trained models loaded successfully!")

    with st.expander("📊 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("K-means Clusters", analyzer.kmeans_model.n_clusters if analyzer.kmeans_model is not None else "Not loaded", help="Number of clusters used in K-means model.")
        with col2:
            st.metric("LDA Topics", analyzer.lda_sklearn.n_components if analyzer.lda_sklearn is not None else "Not loaded", help="Number of topics in LDA model.")
        with col3:
            st.metric("Vocabulary Size", len(analyzer.tfidf_vectorizer.get_feature_names_out()) if analyzer.tfidf_vectorizer is not None else "Not loaded", help="Number of unique words in the TF-IDF vocabulary.")

    if page == "Overview":
        st.header("📊 TopicVista Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Documents", f"{len(analyzer.documents):,}")  # Format with commas

        with col2:
            st.metric("Categories", len(analyzer.categories))

        with col3:
            st.metric("Unique Words", len(analyzer.tfidf_vectorizer.get_feature_names_out()) if analyzer.tfidf_vectorizer else 0)

        st.subheader("📂 Dataset Categories")
        categories_df = pd.DataFrame({
            'Category': analyzer.categories,
            'Document Count': [analyzer.labels.count(cat) for cat in analyzer.categories]
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=categories_df, x='Document Count', y='Category', ax=ax)
        plt.title('Documents per Category')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("🔬 Analysis Methods")
        st.write("""
        This project implements:
        - **K-means Clustering**: Groups similar documents into 20 clusters
        - **LDA Topic Modeling**: Discovers latent topics in the corpus
        - **Text Preprocessing**: Cleaning, tokenization, and lemmatization
        - **Visualization**: t-SNE, UMAP, and word clouds
        - **Advanced Analytics**: Cluster purity, coherence scores, and composition analysis
        """)

    elif page == "Topic Prediction":
        st.header("🎯 Topic Prediction")
        st.write("Enter text to predict its topic and cluster assignment.")

        # Text input
        user_text = st.text_area("Enter your text here:", height=150, help="Paste or type any text to analyze its topic and cluster.")

        if st.button("Predict Topic", help="Click to predict the topic and cluster for the entered text.") and user_text:
            with st.spinner("Analyzing text..."):
                # Models are already loaded, just predict
                if analyzer.kmeans_model is not None and analyzer.tfidf_vectorizer is not None and analyzer.count_vectorizer is not None and analyzer.lda_sklearn is not None:
                    cluster, topic_probs = predict_topic(user_text, analyzer)
                else:
                    st.error("Models or vectorizers are not loaded. Please retrain or reload models.")
                    return

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📍 Cluster Assignment")
                    st.write(f"**Cluster:** {cluster}")

                    # Show most similar category
                    if cluster is not None and analyzer.kmeans_model is not None and hasattr(analyzer.kmeans_model, 'labels_') and analyzer.labels is not None:
                        cluster_docs = [i for i, label in enumerate(analyzer.labels)
                                      if analyzer.kmeans_model.labels_ is not None and i < len(analyzer.kmeans_model.labels_) and analyzer.kmeans_model.labels_[i] == cluster]
                        if cluster_docs:
                            cluster_categories = [analyzer.labels[i] for i in cluster_docs]
                            most_common_cat = max(set(cluster_categories), key=cluster_categories.count)
                            st.write(f"**Most similar category:** {most_common_cat}")

                with col2:
                    st.subheader("📈 Topic Probabilities")
                    if topic_probs is not None:
                        top_topics = np.argsort(topic_probs)[-5:][::-1]
                        topic_df = pd.DataFrame({
                            'Topic': [f"Topic {i}" for i in top_topics],
                            'Probability': [topic_probs[i] for i in top_topics]
                        })
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(data=topic_df, x='Probability', y='Topic', ax=ax)
                        plt.title('Top 5 Topic Probabilities')
                        st.pyplot(fig)
                        plt.close()

    elif page == "Cluster Analysis":
        st.header("🔍 Cluster Analysis")

        if st.button("Run Clustering Analysis", help="Run K-means clustering analysis and view cluster composition."):
            with st.spinner("Running clustering analysis..."):
                if analyzer.kmeans_model is not None and analyzer.tfidf_matrix is not None:
                    cluster_labels = analyzer.kmeans_model.predict(analyzer.tfidf_matrix)
                else:
                    st.error("K-means model or TF-IDF matrix not loaded.")
                    return

                # Show silhouette score
                from sklearn.metrics import silhouette_score
                if analyzer.tfidf_matrix is not None and analyzer.kmeans_model is not None:
                    sil_score = silhouette_score(analyzer.tfidf_matrix, cluster_labels)
                    st.metric("Silhouette Score", f"{sil_score:.3f}")

                # Cluster composition
                st.subheader("📊 Cluster Composition")
                composition_df = pd.crosstab(
                    pd.Series(cluster_labels, name='Cluster'),
                    pd.Series(analyzer.labels, name='True Category')
                )

                fig, ax = plt.subplots(figsize=(15, 10))
                sns.heatmap(composition_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.title('Cluster vs True Category Composition')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close()

                # Show cluster sizes
                st.subheader("📈 Cluster Sizes")
                cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(12, 6))
                cluster_sizes.plot(kind='bar', ax=ax)
                plt.title('Documents per Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Number of Documents')
                st.pyplot(fig)
                plt.close()

    elif page == "Topic Visualization":
        st.header("🎨 Topic Visualization")

        if st.button("Generate Topic Analysis", help="Show top words and word importance for each topic."):
            with st.spinner("Loading topic analysis..."):
                if analyzer.lda_sklearn is not None and analyzer.count_vectorizer is not None:
                    st.subheader("🔬 Top Words per Topic")
                    feature_names = analyzer.count_vectorizer.get_feature_names_out()
                    n_words = st.slider("Number of words to display", 5, 20, 10, help="Select how many top words to show for each topic.")
                    topics_data = []
                    for topic_idx, topic in enumerate(analyzer.lda_sklearn.components_):
                        top_words_idx = topic.argsort()[-n_words:][::-1]
                        # Ensure all elements are strings
                        top_words = [str(analyzer.count_vectorizer.get_feature_names_out()[i]) for i in top_words_idx]
                        topics_data.append({
                            'Topic': f"Topic {topic_idx}",
                            'Top Words': ', '.join(top_words)
                        })
                    topics_df = pd.DataFrame(topics_data)
                    st.dataframe(topics_df, use_container_width=True)
                    st.subheader("🔥 Word Importance Heatmap")
                    # Get top words across all topics
                    top_features = []
                    for topic in analyzer.lda_sklearn.components_:
                        top_indices = topic.argsort()[-10:][::-1]
                        top_features.extend(top_indices)
                    unique_features = list(set(top_features))[:50]  # Limit to 50 words
                    # Create heatmap data
                    heatmap_data = []
                    for topic_idx, topic in enumerate(analyzer.lda_sklearn.components_[:10]):  # First 10 topics
                        topic_words = [topic[i] for i in unique_features]
                        heatmap_data.append(topic_words)
                    # Ensure columns are feature names (strings)
                    columns = [str(feature_names[i]) for i in unique_features]
                    index = [f"Topic {i}" for i in range(len(heatmap_data))]
                    # Ensure columns and index are proper types for pandas
                    heatmap_df = pd.DataFrame(heatmap_data, columns=pd.Index(columns), index=pd.Index(index))
                    fig, ax = plt.subplots(figsize=(15, 8))
                    sns.heatmap(heatmap_df, cmap='YlOrRd', ax=ax)
                    plt.title('Word Importance Across Topics')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error("LDA model or count vectorizer not loaded.")

    elif page == "Advanced Analytics":
        st.header("📈 Advanced Analytics")

        if analyzer.kmeans_model is not None:
            # Generate cluster labels for analysis
            cluster_labels = analyzer.kmeans_model.predict(analyzer.tfidf_matrix)

            # Advanced cluster analysis
            st.subheader("🔍 Advanced Cluster Analysis")
            with st.spinner("Computing advanced metrics..."):
                advanced_analysis = analyzer.advanced_cluster_analysis(cluster_labels)

                # Display metrics
                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    st.metric("Adjusted Rand Index", f"{advanced_analysis['metrics']['Adjusted Rand Index']:.3f}")
                    st.metric("Homogeneity Score", f"{advanced_analysis['metrics']['Homogeneity Score']:.3f}")

                with metrics_col2:
                    st.metric("Normalized Mutual Info", f"{advanced_analysis['metrics']['Normalized Mutual Info']:.3f}")
                    st.metric("Completeness Score", f"{advanced_analysis['metrics']['Completeness Score']:.3f}")

                # Cluster purity analysis
                st.subheader("🎯 Cluster Purity Analysis")
                purity_data = []
                for cluster_id, purity in advanced_analysis['cluster_purity'].items():
                    purity_data.append({
                        'Cluster': cluster_id,
                        'Purity': purity,
                        'Size': advanced_analysis['cluster_sizes'][cluster_id],
                        'Dominant Category': advanced_analysis['dominant_categories'][cluster_id]
                    })

                purity_df = pd.DataFrame(purity_data)
                st.dataframe(purity_df)

                # Cluster purity visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Purity bar plot
                bars = ax1.bar(purity_df['Cluster'], purity_df['Purity'])
                ax1.set_xlabel('Cluster')
                ax1.set_ylabel('Purity')
                ax1.set_title('Cluster Purity Scores')

                # Color bars by purity
                colors = plt.get_cmap('RdYlGn')(purity_df['Purity'])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

                # Size vs Purity scatter plot
                scatter = ax2.scatter(purity_df['Size'], purity_df['Purity'], 
                                    c=purity_df['Purity'], cmap='RdYlGn', s=100)
                ax2.set_xlabel('Cluster Size')
                ax2.set_ylabel('Purity')
                ax2.set_title('Cluster Size vs Purity')
                plt.colorbar(scatter, ax=ax2)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Advanced Cluster Composition Matrix
                st.subheader("📊 Detailed Cluster-Category Composition")
                composition_matrix = analyzer.analyze_cluster_composition(cluster_labels)

                # Create percentage composition
                composition_pct = composition_matrix.div(composition_matrix.sum(axis=1), axis=0) * 100

                # Plot heatmap
                fig, ax = plt.subplots(figsize=(16, 10))
                sns.heatmap(composition_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax)
                ax.set_title('Cluster Composition by True Categories (%)')
                ax.set_xlabel('True Categories')
                ax.set_ylabel('Clusters')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Category Distribution Analysis
                st.subheader("📈 Category Distribution Across Clusters")
                category_cluster_analysis = {}

                for category in analyzer.categories:
                    category_docs = [i for i, label in enumerate(analyzer.labels) if label == category]
                    category_clusters = [cluster_labels[i] for i in category_docs]
                    category_cluster_analysis[category] = {
                        'total_docs': len(category_docs),
                        'cluster_distribution': pd.Series(category_clusters).value_counts().to_dict(),
                        'primary_cluster': pd.Series(category_clusters).mode()[0] if category_clusters else -1
                    }

                # Display category analysis
                for category, analysis in category_cluster_analysis.items():
                    with st.expander(f"📁 {category} ({analysis['total_docs']} documents)"):
                        primary_cluster = analysis['primary_cluster']
                        cluster_dist = analysis['cluster_distribution']

                        st.write(f"**Primary Cluster:** {primary_cluster}")

                        # Plot distribution
                        if cluster_dist:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            clusters = list(cluster_dist.keys())
                            counts = list(cluster_dist.values())

                            bars = ax.bar(clusters, counts)
                            ax.set_xlabel('Cluster')
                            ax.set_ylabel('Number of Documents')
                            ax.set_title(f'Distribution of {category} across Clusters')

                            # Highlight primary cluster
                            if primary_cluster in clusters:
                                primary_idx = clusters.index(primary_cluster)
                                bars[primary_idx].set_color('red')

                            st.pyplot(fig)
                            plt.close()

    elif page == "Interactive LDA":
        st.header("🎯 Interactive LDA Visualization")

        if st.button("Generate Interactive LDA", help="Create and display an interactive LDA visualization."):
            with st.spinner("Creating interactive LDA visualization..."):
                # Use pre-trained Gensim model

                # Get the required data
                if analyzer.preprocessed_docs is None:
                    st.error("Preprocessed documents not loaded. Please ensure preprocessing is complete.")
                    return
                texts = [doc.split() for doc in analyzer.preprocessed_docs if doc is not None]
                if analyzer.lda_gensim is None or not hasattr(analyzer.lda_gensim, 'id2word'):
                    st.error("Gensim LDA model or its dictionary not loaded. Please ensure the model is trained and loaded.")
                    return
                dictionary = analyzer.lda_gensim.id2word
                if dictionary is None or not hasattr(dictionary, 'doc2bow'):
                    st.error("LDA dictionary not loaded or invalid.")
                    return
                # Linter fix: Ensure dictionary is a Gensim Dictionary
                from gensim.corpora.dictionary import Dictionary
                if not isinstance(dictionary, Dictionary):
                    st.error("LDA dictionary is not a valid Gensim Dictionary object.")
                    return
                corpus = [dictionary.doc2bow(text) for text in texts]

                # Create enhanced visualization
                enhanced_results = create_enhanced_pyldavis(
                    analyzer.lda_gensim, dictionary, corpus
                )

                if enhanced_results:
                    # Display coherence information
                    st.subheader("📈 Topic Coherence Analysis")
                    coherence_df = pd.DataFrame(enhanced_results['coherence_info'])
                    st.dataframe(coherence_df)

                    # Topic probability distribution
                    st.subheader("📊 Topic Probability Distribution")
                    topic_probs = enhanced_results['topic_avg_probability']

                    fig, ax = plt.subplots(figsize=(12, 6))
                    topics = [f"Topic {i}" for i in range(len(topic_probs))]
                    sns.barplot(x=topics, y=topic_probs, ax=ax)
                    plt.title('Average Topic Probability Across Documents')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()

                    # Load and display the interactive visualization
                    st.subheader("🎯 Interactive LDA Visualization")
                    try:
                        with open('enhanced_lda_visualization.html', 'r') as f:
                            html_content = f.read()
                        components.html(html_content, height=800, scrolling=True)
                    except FileNotFoundError:
                        st.error("LDA visualization file not found. Please run the analysis again.")

                    # Document-Topic Distribution Analysis
                    st.subheader("📄 Document-Topic Distribution")
                    doc_topic_dist = np.array(enhanced_results['doc_topic_distributions'])

                    # Show distribution statistics
                    max_topic_per_doc = np.argmax(doc_topic_dist, axis=1)
                    dominant_topic_counts = pd.Series(max_topic_per_doc).value_counts().sort_index()

                    fig, ax = plt.subplots(figsize=(12, 6))
                    dominant_topic_counts.plot(kind='bar', ax=ax)
                    plt.title('Number of Documents by Dominant Topic')
                    plt.xlabel('Topic')
                    plt.ylabel('Number of Documents')
                    st.pyplot(fig)
                    plt.close()

                    # Advanced Topic-Document Heatmap
                    st.subheader("🔥 Topic-Document Distribution Heatmap")
                    sample_docs = min(100, len(doc_topic_dist))
                    sample_indices = np.random.choice(len(doc_topic_dist), sample_docs, replace=False)
                    sample_dist = doc_topic_dist[sample_indices]

                    fig, ax = plt.subplots(figsize=(15, 8))
                    sns.heatmap(sample_dist.T, cmap='YlOrRd', ax=ax)
                    ax.set_xlabel('Document Sample')
                    ax.set_ylabel('Topic')
                    ax.set_title(f'Topic Distribution for {sample_docs} Random Documents')
                    st.pyplot(fig)
                    plt.close()

    elif page == "Dataset Statistics":
        st.header("📈 Dataset Statistics")

        # Document length statistics
        st.subheader("📄 Document Length Distribution")
        doc_lengths = [len(doc.split()) for doc in analyzer.documents]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Length", f"{np.mean(doc_lengths):.0f} words")
        with col2:
            st.metric("Median Length", f"{np.median(doc_lengths):.0f} words")
        with col3:
            st.metric("Max Length", f"{np.max(doc_lengths)} words")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(doc_lengths, bins=50, alpha=0.7)
        plt.xlabel('Document Length (words)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Document Lengths')
        st.pyplot(fig)
        plt.close()

        # Category distribution
        st.subheader("📊 Category Distribution")
        category_counts = pd.Series(analyzer.labels).value_counts()

        fig, ax = plt.subplots(figsize=(12, 8))
        category_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.title('Distribution of Documents Across Categories')
        plt.ylabel('')
        st.pyplot(fig)
        plt.close()

        # Vocabulary statistics
        if analyzer.tfidf_vectorizer:
            st.subheader("📚 Vocabulary Statistics")
            feature_names = analyzer.tfidf_vectorizer.get_feature_names_out()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vocabulary Size", len(feature_names))
            with col2:
                try:
                    from scipy.sparse import spmatrix
                except ImportError:
                    spmatrix = None
                tfidf_shape = None
                if analyzer.tfidf_matrix is not None:
                    if (spmatrix is not None and isinstance(analyzer.tfidf_matrix, spmatrix)) or isinstance(analyzer.tfidf_matrix, np.ndarray):
                        tfidf_shape = analyzer.tfidf_matrix.shape
                if tfidf_shape is not None:
                    st.metric("Feature Matrix Shape", f"{tfidf_shape[0]} × {tfidf_shape[1]}")
                else:
                    st.metric("Feature Matrix Shape", "Not loaded")

            # Word frequency analysis
            if analyzer.tfidf_matrix is not None:
                word_freq = None
                sum_func = getattr(analyzer.tfidf_matrix, 'sum', None)
                if callable(sum_func):
                    word_freq = np.array(sum_func(axis=0)).flatten()
                if word_freq is not None:
                    word_freq_df = pd.DataFrame({
                        'Word': feature_names,
                        'Frequency': word_freq
                    }).sort_values('Frequency', ascending=False).head(20)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=word_freq_df, x='Frequency', y='Word', ax=ax)
                    plt.title('Top 20 Most Frequent Words')
                    st.pyplot(fig)
                    plt.close()

if __name__ == "__main__":
    main()
import os
import joblib
from main import NewsGroupsAnalyzer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Preprocessing and Saving Matrices Only ===")
    analyzer = NewsGroupsAnalyzer()
    analyzer.load_documents()
    analyzer.preprocess_documents()
    analyzer.vectorize_documents()
    os.makedirs("trained_models", exist_ok=True)
    joblib.dump(analyzer.preprocessed_docs, "trained_models/preprocessed_docs.pkl")
    joblib.dump(analyzer.tfidf_matrix, "trained_models/tfidf_matrix.pkl")
    joblib.dump(analyzer.count_matrix, "trained_models/count_matrix.pkl")
    print("✓ Preprocessed docs and matrices saved in 'trained_models/' directory.")

if __name__ == "__main__":
    main() 
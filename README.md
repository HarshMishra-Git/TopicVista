# TopicVista — Discover Hidden Stories in News Data

This project applies unsupervised learning (K-means, LDA) to the 20 Newsgroups dataset for document clustering and topic modeling. It features robust preprocessing, model training, visualization, and an interactive Streamlit app.

## Features
- **K-means Clustering** and **LDA Topic Modeling** (scikit-learn & Gensim)
- **Parallelized preprocessing** (spaCy/NLTK)
- **Interactive Streamlit dashboard** (`app.py`)
- **Visualizations**: t-SNE, UMAP, word clouds, pyLDAvis
- **Automated reporting** (`report.py`)
- **Model persistence**: Save/load all models and matrices
- **Extensible, modular codebase**

## Project Structure
- `main.py`: Core analysis pipeline (`NewsGroupsAnalyzer`)
- `app.py`: Streamlit dashboard (run with `streamlit run app.py`)
- `models.py`: Script to train and save all models
- `preprocess_and_save.py`: Preprocess and save matrices without retraining
- `visualization_utils.py`, `text_utils.py`: Utility modules
- `report.py`: Automated report generation
- `trained_models/`: Saved models, vectorizers, and matrices
- `tests/`: Unit tests
- `twenty+newsgroups/`: Raw dataset

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download spaCy model** (if not present):
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. **Prepare dataset**: Ensure the 20 Newsgroups data is in `twenty+newsgroups/20_newsgroups/`.

## Usage
- **Train models**:
  ```bash
  python models.py
  ```
- **Preprocess only** (no retrain):
  ```bash
  python preprocess_and_save.py
  ```
- **Run Streamlit app**:
  ```bash
  streamlit run app.py
  ```
- **Generate report**:
  ```bash
  python report.py
  ```
- **Run tests**:
  ```bash
  python -m unittest discover tests
  ```

## Troubleshooting
- If you see spaCy model errors, run the download command above.
- If models are missing, run `models.py` to train and save them.
- For large datasets, first run `preprocess_and_save.py` to speed up app startup.
- All code is type-checked and linter-clean (no errors expected).

## Extending
- Add new clustering or topic modeling methods in `main.py`.
- Add new visualizations in `visualization_utils.py`.
- Add new Streamlit pages in `app.py`.

## License
MIT License. 
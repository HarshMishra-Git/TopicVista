"""Text preprocessing utility functions for TopicVista — Discover Hidden Stories in News Data."""
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """Clean and preprocess text (moved from main.py)."""
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
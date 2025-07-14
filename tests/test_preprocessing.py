import unittest
from main import NewsGroupsAnalyzer

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.analyzer = NewsGroupsAnalyzer()
        self.analyzer.documents = ["This is a test document!", "Another test document."]

    def test_clean_text(self):
        cleaned = self.analyzer.clean_text(self.analyzer.documents[0])
        self.assertIsInstance(cleaned, str)
        self.assertNotIn('!', cleaned)

    def test_preprocess_documents(self):
        self.analyzer.preprocess_documents()
        self.assertEqual(len(self.analyzer.preprocessed_docs), 2)
        self.assertTrue(all(isinstance(doc, str) for doc in self.analyzer.preprocessed_docs))

    def test_vectorize_documents(self):
        self.analyzer.preprocess_documents()
        self.analyzer.vectorize_documents()
        self.assertIsNotNone(self.analyzer.tfidf_matrix)
        self.assertIsNotNone(self.analyzer.count_matrix)

if __name__ == '__main__':
    unittest.main() 
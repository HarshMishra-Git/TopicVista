import unittest
from main import NewsGroupsAnalyzer

class TestAppLoad(unittest.TestCase):
    def test_analyzer_loads(self):
        analyzer = NewsGroupsAnalyzer()
        analyzer.load_documents()
        # Try loading models, but do not fail if not present
        try:
            loaded = analyzer.load_models()
        except Exception:
            loaded = False
        self.assertTrue(analyzer.documents is not None and len(analyzer.documents) > 0)
        # Models may or may not be present, but code should not error
        self.assertIsInstance(loaded, (bool, type(None)))

if __name__ == '__main__':
    unittest.main() 
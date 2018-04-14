import unittest
from z_api.utils.cleaners import DocumentCleaner


class CleanerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup class."""
        cls.dc = DocumentCleaner(["nltk_tokenize", "spacy_tokenize"])
        cls.documents = ["This is a document.",
                         "this is another doc",
                         "here 's a third one",  # space added so split will catch it.
                         "and here is the last one"
                         ]
        cls.act_out = []

    @classmethod
    def tearDownClass(cls):
        """Reset all class variables."""
        # cls.dc.cleanup()
        pass

    def test_nltk_word_tokenizer(self):
        """NLTK word_tokenizer."""
        exp_out = [['This', 'is', 'a', 'document', '.'],
                   ['this', 'is', 'another', 'doc'],
                   ['here', "'s", 'a', 'third', 'one'],
                   ['and', 'here', 'is', 'the', 'last', 'one']]

        for doc, exp in zip(self.documents, exp_out):
            # self.dc.cleanup()
            self.dc.doc = doc
            self.dc.nltk_word_tokenize()
            self.assertListEqual(exp, self.dc.tokens)
            self.assertListEqual(exp, self.dc.current)

    def test_spacy_word_tokenizer(self):
        """NLTK word_tokenizer."""
        exp_out = [['This', 'is', 'a', 'document', '.'],
                   ['this', 'is', 'another', 'doc'],
                   ['here', "'s", 'a', 'third', 'one'],
                   ['and', 'here', 'is', 'the', 'last', 'one']]

        for doc, exp in zip(self.documents, exp_out):
            self.dc.doc = doc
            self.dc.spacy_tokenize()
            self.assertListEqual(exp, self.dc.tokens)
            self.assertListEqual(exp, self.dc.current)

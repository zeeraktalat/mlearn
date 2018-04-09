import unittest
from mlapi.utils.cleaners import DocumentCleaner

class CleanerTest(unittest.TestCase):

    def setUpClass(cls):
        """Setup class."""
        cls.dc = DocumentCleaner(["nltk_tokenize", "spacy_tokenize"])
        cls.documents = ["This is a document.",
                         "this is another doc",
                         "here 's a third one",  # space added so split will catch it.
                         "and here is the last one"
                         ]
        cls.act_out = []

    def test_nltk_word_tokenizer(self):
        """NLTK word_tokenizer."""
        exp_out = [d.split() for d in self.documents]

        for doc, exp in zip(self.documents, exp_out):
            self.dc.doc = doc
            self.dc.nltk_word_tokenize()
            self.assertListEqual(exp_out, self.tokens)
            self.assertListEqual(exp_out, self.current)
            self.assertListEqual(exp_out, self.dout)

    def test_spacy_word_tokenizer(self):
        """NLTK word_tokenizer."""
        exp_out = [d.split() for d in self.documents]

        for doc, exp in zip(self.documents, exp_out):
            self.dc.doc = doc
            self.dc.spacy_tokenize()
            self.assertListEqual(exp_out, self.tokens)
            self.assertListEqual(exp_out, self.current)
            self.assertListEqual(exp_out, self.dout)

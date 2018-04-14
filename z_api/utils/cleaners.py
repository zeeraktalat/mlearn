"""File for document cleaners."""

from nltk import word_tokenize
from typing import List
import spacy


class DocumentCleaner(object):
    """Class for running cleaning documents and handling preprocessing."""

    def __init__(self, methods: List[str] = [], **kwargs):
        """Set initialisations so that loading only happens once."""
        # Initialise variables
        self.kwargs     = kwargs
        self.method_map = {}
        self.document   = None
        self.method     = methods

    @property
    def methods(self):
        """Handle method mapping."""
        return self.method_map

    @methods.setter
    def methods(self, methods):
        for m in methods:
            try:
                self.method_map.update({m: getattr(self, m)})
            except AttributeError as e:
                print("Method {0} not found".format(m))
                raise(e)

    @property
    def doc(self):
        """Document handler."""
        return self.document

    @doc.setter
    def doc(self, document: str):
        self.document = document
        self.current  = document
        self.tokens   = None

    def generate(self):
        """Generate features, where each item is a callable function."""
        if self.method_map == {}:
            self.methods = self.method

        for m_str in self.method_map:
            self.method_map[m_str](**self.kwargs)

    def nltk_word_tokenize(self):
        """Tokenise using NLTK word_tokenise. Updates self.tokens."""
        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)

            self.tokens = word_tokenize(self.current)
            self.current = self.tokens[:]
        else:
            raise ValueError("Document not set.")


    def spacy_tokenize(self):
        """Tokenise using spacy's tokenizer. Updates self.tokens."""
        try:
            self.spacy_parser(self.document)
        except AttributeError as e:
            self.spacy_parser = spacy.load('en')

        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)

            self.spacy_parsed = self.spacy_parser(self.current)
            self.tokens = [str(tok) for tok in self.spacy_parsed]
            self.current = self.tokens[:]
        else:
            raise ValueError("No document fed to module.")

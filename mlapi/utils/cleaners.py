"""File for document cleaners."""

from nltk import word_tokenize
from typing import List
import spacy


class DocumentCleaner(object):
    """Class for running cleaning documents and handling preprocessing."""

    def __init__(self, methods: List[str] = [], **kwargs):
        """Set initialisations so that loading only happens once."""
        # Initialise variables
        self.args       = kwargs
        self.method_map = {}
        self.document   = None
        self.dout       = None
        self.current    = None

        # Initialise imports
        self.tagger  = spacy.load('en')

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

    def generate(self):
        """Generate features, where each item is a callable function."""
        if self.method_map == {}:
            self.str_to_method = self.methods

        for m_str in self.method_map:
            self.method_map[m_str](**self.kwargs)

    def nltk_word_tokenize(self):
        """Tokenise using NLTK word_tokenise. Updates self.tokens."""
        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)

        elif self.document:
            self.tokens = word_tokenize(self.document)
        else:
            raise ValueError("Document not set.")
        self.current = self.tokens[:]
        self.dout = self.current

    def spacy_tokenize(self):
        """Tokenise using spacy's tokenizer. Updates self.tokens."""
        if not self.spacy_parser:
            self.spacy_parser = spacy.load('en')

        if self.document:
            self.tokens = self.spacy_parser(self.document)
            self.current = self.tokens[:]
            self.dout = self.current
        else:
            raise ValueError("Document not set.")

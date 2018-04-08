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
        self.dout       = None

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

    def nltk_tokenize(self):
        """Tokenise using NLTK word_tokenise. Updates self.tokens."""
        if self.document:
            self.tokens = word_tokenize(self.document)
            self.dout = self.tokens[:]
        else:
            raise ValueError

    def spacy_tokenize(self):
        """Tokenise using spacy's tokenizer. Updates self.tokens."""
        if self.document:
            self.tokens = word_tokenize(self.document)
        else:
            raise ValueError

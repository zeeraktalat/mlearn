"""File for document cleaners."""

from nltk import word_tokenize
from collections import OrderedDict
from string import punctuation
from typing import List
import spacy
import re


class DocumentCleaner(object):
    """Class for running cleaning documents and handling preprocessing."""

    def __init__(self, methods: List[str] = [], **kwargs):
        """Set initialisations so that loading only happens once."""
        # Initialise variables
        self.kwargs     = kwargs
        self.method_map = OrderedDict()
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

    def nltk_word_tokenize(self, **kwargs):
        """Tokenise using NLTK word_tokenise. Updates self.tokens."""
        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)

            self.tokens = word_tokenize(self.current)
            self.current = self.tokens[:]
        else:
            raise ValueError("Document not set.")

    def spacy_tokenize(self, **kwargs):
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

    def strip_punctuation(self, **kwargs):
        """Remove all punctation marks."""
        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)
            self.current = "".join([c for c in self.current if c not in punctuation])

    def replace_punctuation(self, **kwargs):
        """Remove all punctation marks."""
        if self.current:
            if isinstance(self.current, list):
                self.current = " ".join(self.current)
            self.current = "".join(["_PUNKT_" if c in punctuation else c for c in self.current])

    def strip_url(self, **kwargs):
        """Remove all URLs."""
        pattern = "https*\S*|www.\S*.\S|([a-z]*\.[a-z]+\.?[a-z]{2,4}?)"
        self.current = re.sub(pattern, "", self.current)

    def replace_url(self, **kwargs):
        """Replace all URLs with _URL_ token."""
        pattern = "https*\S*|www.\S*.\S|([a-z]*\.[a-z]+\.?[a-z]{2,4}?)"
        self.current = re.sub(pattern, "_URL_", self.current)

    def strip_at_username(self, **kwargs):
        """Remove all @user names."""
        pattern = '@[_a-zA-Z0-9]*\w'
        self.current = re.sub(pattern, "", self.current)

    def replace_at_username(self, **kwargs):
        """Replace all @user names with an _USER_ token."""
        pattern = '@[_a-zA-Z0-9]*\w'
        self.current = re.sub(pattern, "_USER_", self.current)

    def strip_ints(self, **kwargs):
        """Remove all integers. ALWAYS AFTER FLOAT."""
        pattern = '[0-9]+'
        self.current = re.sub(pattern, "", self.current)

    def replace_ints(self, **kwargs):
        """Replace all ints with an _INT_ token. ALWAYS AFTER FLOAT."""
        pattern = '[0-9]+'
        self.current = re.sub(pattern, "_INT_", self.current)

    def strip_float(self, **kwargs):
        """Remove all floats. ALWAYS BEFORE INT."""
        pattern = '[-+]?\d*([,.])\d+'
        self.current = re.sub(pattern, "", self.current)

    def replace_float(self, **kwargs):
        """Replace all floats with _FLOAT_ token. ALWAYS BEFORE INT."""
        pattern = '[-+]?\d*([,.])\d+'
        self.current = re.sub(pattern, "_FLOAT_", self.current)

    def strip_hashtag(self, **kwargs):
        """Remove all hashtags."""
        pattern = "#[a-zA-Z0-9]+\w"
        self.current = re.sub(pattern, "", self.current)

    def replace_hashtag(self, **kwargs):
        """Replace all hashtags with _HASHTAG_ token."""
        pattern = "#[a-zA-Z0-9]+\w"
        self.current = re.sub(pattern, "_HASHTAG_", self.current)

    def strip_newlines(self, **kwargs):
        """Remove all linebreaks."""
        self.current = re.sub("\n", " ", self.current)

    def strip_spaces(self, **kwargs):
        """Remove all spaces."""
        self.current = re.sub("\s\s+", " ", self.current)

    def lowercase(self, **kwargs):
        """Lowercase document or tokens."""
        if self.current:
            if isinstance(self.current, list):
                self.current = [item.lower() for item in self.current]
            else:
                self.current = self.current.lower()

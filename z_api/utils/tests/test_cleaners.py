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
        """Spacy word_tokenizer."""
        exp_out = [['This', 'is', 'a', 'document', '.'],
                   ['this', 'is', 'another', 'doc'],
                   ['here', "'s", 'a', 'third', 'one'],
                   ['and', 'here', 'is', 'the', 'last', 'one']]

        for doc, exp in zip(self.documents, exp_out):
            self.dc.doc = doc
            self.dc.spacy_tokenize()
            self.assertListEqual(exp, self.dc.tokens)
            self.assertListEqual(exp, self.dc.current)

    def test_strip_punctuation(self):
        """Punctuation removel."""
        documents = [['This is a document.'],
                     ['this is another doc!'],
                     ["here's a third! one"],
                     ['and here is the last one$']]
        exp_out = ['This is a document',
                   'this is another doc',
                   "heres a third one",
                   'and here is the last one']

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_punctuation()
            self.assertEqual(exp, self.dc.current)

    def test_replace_punctuation(self):
        """Punctuation removel."""
        documents = [['This is a document.'],
                     ['this is another doc!'],
                     ["here's a third! one"],
                     ['and here is the last one$']]
        exp_out = ['This is a document_PUNKT_',
                   'this is another doc_PUNKT_',
                   "here_PUNKT_s a third_PUNKT_ one",
                   'and here is the last one_PUNKT_']

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_punctuation()
            self.assertEqual(exp, self.dc.current)

    def test_strip_url(self):
        """Remove urls."""
        documents = ["Here's a link: https://pythex.org",
                     "And here's another: http://pythex.org",
                     "And here's the last one: pythex.org"]
        exp_out   = ["Here's a link: ",
                     "And here's another: ",
                     "And here's the last one: "]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_url()
            self.assertEqual(exp, self.dc.current)

    def test_replace_url(self):
        """Replace urls."""
        documents = ["Here's a link: https://pythex.org",
                     "And here's another: http://pythex.org",
                     "And here's the last one: pythex.org"]
        exp_out   = ["Here's a link: _URL_",
                     "And here's another: _URL_",
                     "And here's the last one: _URL_"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_url()
            self.assertEqual(exp, self.dc.current)

    def test_strip_at_user(self):
        """Remove @users."""
        documents = ["Here's a user: @zeerakw!",
                     "And here's @zeerakw another",
                     "And here's the last one: @z33r4k"]
        exp_out   = ["Here's a user: !",
                     "And here's  another",
                     "And here's the last one: "]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_at_username()
            self.assertEqual(exp, self.dc.current)

    def test_replace_at_user(self):
        """Replace @user."""
        documents = ["Here's a user: @zeerakw!",
                     "And here's @zeerakw another",
                     "And here's the last one: @z33r4k"]
        exp_out   = ["Here's a user: _USER_!",
                     "And here's _USER_ another",
                     "And here's the last one: _USER_"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_at_username()
            self.assertEqual(exp, self.dc.current)

    def test_strip_ints(self):
        """Remove Integers."""
        documents = ["Here's a number: 29!",
                     "And here's @zeera29kw another",
                     "And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a number: !",
                     "And here's @zeerakw another",
                     "And here's the last . one: @zrk"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_ints()
            self.assertEqual(exp, self.dc.current)

    def test_replace_ints(self):
        """Replace Integers."""
        documents = ["Here's a number: 29!",
                     "And here's @zeera29kw another",
                     "And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a number: _INT_!",
                     "And here's @zeera_INT_kw another",
                     "And here's the last _INT_._INT_ one: @z_INT_r_INT_k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_ints()
            self.assertEqual(exp, self.dc.current)

    def test_strip_floats(self):
        """Remove Floats."""
        documents = ["Here's a number: .29!",
                     "And here's @zeera23.29kw another",
                     "And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a number: !",
                     "And here's @zeerakw another",
                     "And here's the last  one: @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_float()
            self.assertEqual(exp, self.dc.current)

    def test_replace_floats(self):
        """Replace Floats."""
        documents = ["Here's a number: .29!",
                     "And here's @zeera23.29kw another",
                     "And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a number: _FLOAT_!",
                     "And here's @zeera_FLOAT_kw another",
                     "And here's the last _FLOAT_ one: @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_float()
            self.assertEqual(exp, self.dc.current)

    def test_strip_hashtags(self):
        """Remove Hashtags."""
        documents = ["Here's a #hashtag: !#0testing",
                     "And here's @zeerakw#awe20some another",
                     "#yup And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a : !",
                     "And here's @zeerakw another",
                     " And here's the last 0.30 one: @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_hashtag()
            self.assertEqual(exp, self.dc.current)

    def test_replace_hashtags(self):
        """Replace Hashtags."""
        documents = ["Here's a #hashtag: !#0testing",
                     "And here's @zeerakw#awe20some another",
                     "#yup And here's the last 0.30 one: @z33r4k"]
        exp_out   = ["Here's a _HASHTAG_: !_HASHTAG_",
                     "And here's @zeerakw_HASHTAG_ another",
                     "_HASHTAG_ And here's the last 0.30 one: @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.replace_hashtag()
            self.assertEqual(exp, self.dc.current)

    def test_strip_newlines(self):
        """Replace Hashtags."""
        documents = ["Here's a \n\n\n\n#hashtag: !#0testing",
                     "And here's @zeerakw \n\n\nsome another",
                     "#yup And here's the last 0.30 one\n\n\n: @z33r4k"]
        exp_out   = ["Here's a     #hashtag: !#0testing",
                     "And here's @zeerakw    some another",
                     "#yup And here's the last 0.30 one   : @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_newlines()
            self.assertEqual(exp, self.dc.current)

    def test_strip_spaces(self):
        """Replace Hashtags."""
        documents = ["Here's a     #hashtag: !#0testing",
                     "And here's @zeerakw    some another",
                     "#yup And here's the last 0.30 one   : @z33r4k"]
        exp_out   = ["Here's a #hashtag: !#0testing",
                     "And here's @zeerakw some another",
                     "#yup And here's the last 0.30 one : @z33r4k"]

        for doc, exp in zip(documents, exp_out):
            self.dc.doc = doc
            self.dc.strip_spaces()
            self.assertEqual(exp, self.dc.current)

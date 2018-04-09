"""Implements document cleaners."""
import unittest
from mlapi.features.Features import LinguisticFeatures
from mlapi.utils.cleaners import DocumentCleaner


class FeatureTest(unittest.TestCase):
    """Test features."""

    @classmethod
    def setUpClass(cls):
        """Setup class."""
        dc = DocumentCleaner(["nltk_tokenize"])
        methods = ["unigrams",
                   "token_ngrams",
                   "skip_grams",
                   "char_ngrams",
                   "sentiment",
                   "word_count",
                   "avg_word_length"
                   ]
        cls.kwargs = {"ngrams": 2,
                  "char_ngrams": 3,
                  "skip_size": 2,
                  "stopped": False,
                  }
        cls.fc = LinguisticFeatures(methods, dc, **cls.kwargs)
        cls.documents = ["This is a document.",
                         "this is another doc",
                         "here 's a third one",  # space added so split will catch it.
                         "and here is the last one"
                         ]
        cls.act_out = []

    @classmethod
    def tearDownClass(cls):
        """Tear down information from class."""
        pass

    def test_unigrams(self):
        """Unigram Features."""
        exp_out = [doc.split() for doc in self.documents]

        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.unigrams(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_token_ngrams(self):
        """N-gram Features."""
        exp_out = [["This_is", "is_a", "a_document"],
                   ["this_is", "is_another", "another_doc"],
                   ["here_'s", "'s_a", "a_third", "third_one"],
                   ["and_here", "here_is", "is_the", "the_last", "last_one"]
                   ]

        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.token_ngrams(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_skip_grams(self):
        """Skip-gram Features."""
        exp_out = [["This_is",
                    "This_a",
                    "This_document",
                    "is_a",
                    "is_document",
                    "a_document"
                    ],
                   ["this_is",
                    "this_another",
                    "this_doc",
                    "is_another",
                    "is_doc",
                    "another_doc"
                    ],
                   ["here_'s",
                    "here_a",
                    "here_third",
                    "here_one",
                    "'s_a",
                    "'s_third",
                    "'s_one" "a_third",
                    "a_one",
                    "third_one"
                    ],
                   ["and_here",
                    "and_is",
                    "and_the",
                    "and_last",
                    "here_is",
                    "here_the",
                    "here_last",
                    "here_one",
                    "is_the",
                    "is_last",
                    "is_one",
                    "the_last",
                    "the_one",
                    "last_one"
                    ]
                   ]
        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.skip_grams(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_char_ngrams(self):
        """Character N-gram Features."""
        exp_out = [['T_h_i',
                    'h_i_s',
                    'i_s_ ',
                    's_ _i',
                    ' _i_s',
                    'i_s_ ',
                    's_ _a',
                    ' _a_ ',
                    'a_ _d',
                    ' _d_o',
                    'd_o_c',
                    'o_c_u',
                    'c_u_m',
                    'u_m_e',
                    'm_e_n',
                    'e_n_t',
                    'n_t_.'
                    ],
                   ['t_h_i',
                    'h_i_s',
                    'i_s_ ',
                    's_ _i',
                    ' _i_s',
                    'i_s_ ',
                    's_ _a',
                    ' _a_n',
                    'a_n_o',
                    'n_o_t',
                    'o_t_h',
                    't_h_e',
                    'h_e_r',
                    'e_r_ ',
                    'r_ _d',
                    ' _d_o',
                    'd_o_c'
                    ],
                   ['h_e_r',
                    'e_r_e',
                    'r_e_ ',
                    "e_ _'",
                    " _'_s",
                    "'_s_ ",
                    's_ _a',
                    ' _a_ ',
                    'a_ _t',
                    ' _t_h',
                    't_h_i',
                    'h_i_r',
                    'i_r_d',
                    'r_d_ ',
                    'd_ _o',
                    ' _o_n',
                    'o_n_e'
                    ],
                   ['a_n_d',
                    'n_d_ ',
                    'd_ _h',
                    ' _h_e',
                    'h_e_r',
                    'e_r_e',
                    'r_e_ ',
                    'e_ _i',
                    ' _i_s',
                    'i_s_ ',
                    's_ _t',
                    ' _t_h',
                    't_h_e',
                    'h_e_ ',
                    'e_ _l',
                    ' _l_a',
                    'l_a_s',
                    'a_s_t',
                    's_t_ ',
                    't_ _o',
                    ' _o_n',
                    'o_n_e'
                    ]
                   ]
        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.char_ngrams(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_sentiment_aggregate(self):
        """Sentiment Aggregate Features."""
        documents = ["I love this document", "I hate this document", "This document is okay"]
        exp_out = [{"SENTIMENT": "pos"}, {"SENTIMENT": "neg"}, {"SENTIMENT": "neu"}]

        for doc in documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.sentiment_aggregate(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_sentiment_scores(self):
        """Sentiment Score Features."""
        documents = ["I love this document", "I hate this document", "This document is okay"]
        exp_out = [{'neg': 0.0, 'neu': 0.323, 'pos': 0.677},
                   {'neg': 0.649, 'neu': 0.351, 'pos': 0.0},
                   {'neg': 0.0, 'neu': 0.612, 'pos': 0.388}]

        for doc in documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.sentiment_scores(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_word_count(self):
        """Word Count Features."""
        exp_out = [{"TOK_COUNT": 4}, {"TOK_COUNT": 4}, {"TOK_COUNT": 5}, {"TOK_COUNT": 5}]

        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.word_count(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

    def test_avg_word_length(self):
        """Average Word Length Features."""
        exp_out = [{"AVG_TOK_LEN": 4.0}, {"AVG_TOK_LEN": 4.0}, {"AVG_TOK_LEN": 3.0}, {"AVG_TOK_LEN": 3.17}]

        for doc in self.documents:
            self.fc.doc = doc
            self.act_out.append(self.fc.avg_word_length(**self.kwargs))

        self.assertListEqual(exp_out, self.act_out)

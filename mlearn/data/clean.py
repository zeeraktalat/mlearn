import re
import spacy
from bpemb import BPEmb
from mlearn import base
from string import punctuation
from ekphrasis.utils.nlp import unpack_contractions
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


class Preprocessors(object):
    """A class to contain preprocessors and wrap preprocessing functions and their requirements."""

    def __init__(self, liwc_dir: str = None):
        """Initialise cleaner class."""
        self.tagger = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])
        self.liwc_dict = None
        self.slurs = None
        self.slur_window = None
        if liwc_dir is None:
            self.liwc_path = None
        else:
            self.liwc_path = liwc_dir + 'liwc-2015.csv'

    def select_experiment(self, exp: str, slur_window: int = None) -> base.Callable:
        """
        Select experiment to run.

        :exp (str): The experiment to run.
        :returns experiment: Return th experiment to run.
        """
        if exp == 'word':
            experiment = self.word_token

        elif exp == 'liwc':
            experiment = self.compute_unigram_liwc

        elif exp in ['ptb', 'pos']:
            experiment = self.ptb_tokenize

        elif exp == 'length':
            experiment = self.word_length

        elif exp == 'syllable':
            experiment = self.syllable_count

        elif exp == 'slur':
            self.slur_window = slur_window
            experiment = self.slur_replacement
        return experiment

    def word_length(self, doc: base.DocType) -> base.List[int]:
        """
        Represent sentence as the length of each token.

        :doc (base.DocType): Document to be processed.
        :returns: Processed document.
        """
        return [len(tok) for tok in doc]

    def syllable_count(self, doc: base.DocType) -> base.List[int]:
        """
        Represent sentence as the syllable count for each word.

        :doc (base.DocType): Document to be processed.
        :returns: Processed document.
        """
        return [self._syllable_counter(tok) for tok in doc]

    def _syllable_counter(self, tok: str) -> int:
        """
        Calculate syllables for each token.

        :tok (str): The token to be analyzed.
        :returns count (int): The number of syllables in the word.
        """
        count = 0
        vowels = 'aeiouy'
        exceptions = ['le', 'es', 'e']
        prev_char = '<s>'

        for i, char in enumerate(tok):
            if i == len(tok) and (prev_char + char in exceptions or char in exceptions):
                pass
            if (char in vowels) and (prev_char not in vowels and char != prev_char):
                count += 1
            prev_char = char
        return count

    def load_slurs(self):
        """Load slurs file."""
        self.slurs = None
        # TODO Update this with a slur list

    def slur_replacement(self, doc: base.DocType):
        """
        Produce documents where slurs are replaced.

        :doc (base.List[str]): Document to be processed.
        :returns doc: processed document
        """
        if self.slurs is None:
            self.slurs = self.load_slurs()

        slur_loc = [i for i, tok in enumerate(doc) if tok in self.slurs]
        pos = [tok for tok in self.tagger(" ".join(doc))]

        for ix in slur_loc:  # Only look at the indices where slurs exist
            min_ix = 0 if ix - self.slur_window < 0 else ix - self.slur_window
            max_ix = len(doc) - 1 if ix + self.slur_window > len(doc) - 1 else ix + self.slur_window

            for i in range(min_ix, max_ix, 1):  # Do replacements within the window
                doc[i] = pos[i]
        return doc

    def word_token(self, doc: base.DocType) -> base.DocType:
        """
        Produce word tokens.

        :doc (base.List[str]): Document to be processed.
        :returns: processed document
        """
        return doc

    def ptb_tokenize(self, document: base.DocType, processes: base.List[str] = None) -> base.DocType:
        """
        Tokenize the document using SpaCy, get PTB tags and clean it as it is processed.

        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        self.processes = processes if processes else self.processes
        toks = [tok.tag_ for tok in self.tagger(" ".join(document))]
        return toks

    def read_liwc(self) -> dict:
        """Read LIWC dict."""
        with open(self.liwc_path, 'r') as liwc_f:
            liwc_dict = {}
            for line in liwc_f:
                k, v = line.strip('\n').split(',')
                if k in liwc_dict:
                    liwc_dict[k] += [v]
                else:
                    liwc_dict.update({k: [v]})
        return liwc_dict

    def _compute_liwc_token(self, tok: str, kleene_star: base.List[str]) -> str:
        """
        Compute LIWC categories for a given token.

        :tok (str): Token to identify list of.
        :kleen_star: List of kleen_start tokens.
        :returns (str): Token reprented in terms of LIWC categories.
        """
        if tok in self.liwc_dict:
            term = self.liwc_dict[tok]
        else:
            liwc_cands = [r for r in kleene_star if tok.startswith(r)]
            num_cands = len(liwc_cands)

            if num_cands == 0:
                term = 'NUM' if re.findall(r'[0-9]+', tok) else 'UNK'

            elif num_cands == 1:
                term = liwc_cands[0] + '*'

            elif num_cands > 1:
                sorted_cands = sorted(liwc_cands, key=len, reverse = True)  # Longest first
                term = sorted_cands[0] + '*'

            if term not in ['UNK', 'NUM']:
                liwc_term = self.liwc_dict[term]

                if isinstance(liwc_term, list):
                    term = "_".join(liwc_term)
                else:
                    term = liwc_term
        if isinstance(term, list):
            term = "_".join(term)

        return term

    def compute_unigram_liwc(self, doc: base.DocType) -> base.DocType:
        """
        Compute LIWC for each document document.

        :doc (base.DocType): Document to operate on.
        :returns liwc_doc (base.DocType): Document represented as LIWC categories.
        """
        if not self.liwc_dict:
            self.liwc_dict = self.read_liwc()
        liwc_doc = []
        kleene_star = [k[:-1] for k in self.liwc_dict if k[-1] == '*']

        if isinstance(doc, str):
            doc = [w if w[0] not in punctuation and w[-1] not in punctuation
                   else w.strip(punctuation) for w in doc.split()]

        liwc_doc = [self._compute_liwc_token(tok, kleene_star) for tok in doc]

        assert(len(liwc_doc) == len(doc))

        return liwc_doc

    # TODO Othering language:
    # Parse the document to see if there are us/them, we/them/ i/you
    # Consider looking at a window that are 2-5 words before/after a slur.


class Cleaner(object):
    """A class for methods for cleaning."""

    def __init__(self, processes: base.List[str] = None, ekphrasis_base: bool = False):
        """
        Initialise cleaner class.

        :processes (base.List[str]): Cleaning operations to be taken.
        :ekprhasis_base (bool, default = False): Use ekphrasis to pre-process data in cleaner.
        """
        self.processes = processes if processes is not None else []
        self.tagger = spacy.load('en_core_web_sm', disable = ['ner', 'parser', 'textcats'])
        self.bpe = BPEmb(lang = 'en', vs = 200000).encode
        self.ekphrasis_base = ekphrasis_base
        self.ekphrasis = None
        self.liwc_dict = None

    def clean_document(self, text: base.DocType, processes: base.List[str] = None, annotate: set = {'elongated'}):
        """
        Clean document.

        :text (types.DocType): The document to be cleaned.
        :processes (List[str]): The cleaning processes to be undertaken.
        :returns cleaned: Return the cleaned text.
        """
        if processes is None:
            process = []
        process = processes if processes is not None else self.processes
        cleaned = str(text)
        if 'lower' in process:
            cleaned = cleaned.lower()
        if 'url' in process:
            cleaned = re.sub(r'https?:/\/\S+', 'URL', cleaned)
        if 'hashtag' in process:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', 'HASHTAG', cleaned)
        if 'username' in process:
            cleaned = re.sub(r'@\S+', 'USER', cleaned)

        if self.ekphrasis_base:
            cleaned = self.ekphrasis_tokenize(cleaned, annotate, filters = [f"<{filtr}>" for filtr in annotate])

        return cleaned

    def tokenize(self, document: base.DocType, processes: base.List[str] = None, **kwargs):
        """
        Tokenize the document using SpaCy and clean it as it is processed.

        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        toks = [tok.text for tok in self.tagger(self.clean_document(document, processes = processes))]
        return toks

    def bpe_tokenize(self, document: base.DocType, processes: base.List[str] = None, **kwargs):
        """
        Tokenize the document using BPE and clean it as it is processed.

        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        toks = self.bpe(self.clean_document(document, processes = processes))
        return toks

    def _load_ekphrasis(self, annotate: set, normalize: base.List[str] = None,
                        segmenter: str = 'twitter', corrector: str = 'twitter', hashtags: bool = False,
                        elong_spell: bool = False, **kwargs) -> None:
        """
        Set up ekphrasis tokenizer.

        :annotate (set): Set of annotations to use (controls corrections).
        :normalize (base.List[str], default = None): List of normalisations.
        :segmenter (str, default = 'twitter'): Choose which ekphrasis segmenter to use.
        :corrector (str, default = 'twitter'): Choose which ekphrasis spell correction to use.
        :hashtags (bool, default = False): Unpack hashtags into multiple tokens (e.g. #PhDLife -> PhD Life).
        :elong_spell (bool, default = True): Spell correct elongations.
        """
        self.ekphrasis = TextPreProcessor(normalize = normalize if normalize is not None else [],
                                          annotate = annotate,
                                          fix_html = True,
                                          segmenter = segmenter,
                                          corrector = corrector,
                                          unpack_hashtags = hashtags,
                                          spell_correct_elong = elong_spell,
                                          tokenize = SocialTokenizer(lowercase = True).tokenize)

    def _filter_ekphrasis(self, document: base.DocType, filters: base.List[str] = None, **kwargs) -> base.List[str]:
        """
        Remove Ekphrasis specific tokens.

        :document (base.DocType): The document to process.
        :removals (base.List[str]): The ekphrasis tokens to remove.
        :returns document: Document filtered for ekphrasis specific tokens.
        """
        if filters is not None:

            for filtr in filters:
                document = document.replace(filtr, '')

            document = document.split()
        return document

    def ekphrasis_tokenize(self, document: base.DocType, processes: base.List[str] = None, **kwargs
                           ) -> base.DocType:
        """
        Tokenize the document using BPE and clean it as it is processed.

        :document: Document to be parsed.
        :processes: The cleaning processes to engage in.
        :returns toks: Document that has been passed through spacy's tagger.
        """
        if self.ekphrasis is None:
            self._load_ekphrasis(**kwargs)

        if isinstance(document, list):
            document = " ".join(document)

        doc = unpack_contractions(document)
        doc = self.clean_document(doc, processes)
        doc = self.ekphrasis.pre_process_doc(doc)

        return self._filter_ekphrasis(doc, **kwargs)

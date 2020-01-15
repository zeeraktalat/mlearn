import re
from nltk import ngrams
from collections import Counter
from . import base


def unigrams(doc: base.List[str]) -> base.Dict[str, int]:
    """Compute unigrams.
    :param doc: Document to be parsed.
    :return: counted ngrams.
    """
    return Counter(doc)


def n_grams(doc: base.Union[str, base.List[str]], n: int = 2) -> base.Dict[str, int]:
    """
    :param doc: Document to create ngrams from.
    :param n: Size of ngrams.
    :return: Counter object of counted token bigrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, n)]) + unigrams(doc)


def char_ngrams(doc: str, n: int = 2) -> base.Dict[str, int]:
    """
    :param doc: Document to character create ngrams from.
    :return: Counter object of counted char ngrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, 2)])


def char_count(doc: str) -> base.Dict[str, int]:
    """
    :param doc: Document to create character counts from.
    :return: Counter dict of counted characters in document.
    """
    return Counter(doc)


def find_mentions(doc: str) -> base.List[str]:
    """
    :param doc: Document to find mentions in.
    :return: base.List of mentions.
    """
    return re.findall(r'@[a-zA-Z0-9]', doc)


def find_hashtags(doc: str) -> base.List[str]:
    """
    :param doc: Document to find hashtags in.
    :return: base.List of hashtags.
    """
    return re.findall(r'#[a-zA-Z0-9]', doc)


def find_urls(doc: str) -> base.List[str]:
    """
    :param doc: Document to find URLs in.
    :return: base.List of urls.
    """
    return re.findall(r'http:\/\/[a-zA-Z0-9\.\/a-zA-Z0-9]+', doc)


def find_retweets(doc: str) -> base.List[str]:
    """
    :param doc: Document to find retweets in.
    :return: base.List of retweets.
    """
    return re.findall(r'\wRT\w @', doc)


def count_syllables(doc: base.List[str]) -> int:
    """Simplistic syllable count.
    :param doc: Document to count syllables in.
    :return: Count of syllables.
    """
    count = 0
    vowels = 'aeiouy'
    exceptions = ['le', 'es', 'e']
    for word in doc:
        prev_char = None
        for i in word:
            if i == len(word) and (prev_char + word[i] in exceptions or word[i] in exceptions):
                prev_char = word[i]
                continue
            if (word[i] in vowels) and (prev_char not in vowels and not prev_char):
                prev_char = word[i]
                count += 1
    return {'NO_SYLLABLES': count}


def word_list(doc: base.List[str], word_list: base.List[str], salt: str) -> base.Dict[str, int]:
    """Identify if words are in word_list.
    :param doc: Tokenised document.
    :param word_list: base.List containing words occurring in the wordlist.
    :param salt: To add in front of word name.
    :return: Counts of each word appearing in dict.
    """
    salt += '_WORDLIST_'
    res = []
    for w in doc:
        if w in word_list:
            res.append(salt + w)
    return Counter(res) if len(res) != 0 else {salt: 0}


def _pos_helper(docs: base.List[str]) -> base.Tuple[base.List[str], base.List[str], base.List[str]]:
    # for doc in tqdm(docs, desc = "POS helper"):
    for doc in docs:
        tokens = []
        pos = []
        confidence = []
        for tup in doc:
            tokens.append(tup[0])
            pos.append(tup[1])
            confidence.append(tup[2])
        yield tokens, pos, confidence


def sentiment_polarity(doc: str, sentiment: base.Callable) -> base.Dict[str, float]:
    """Compute sentiment polarity scores and return features.
    :param doc: Document to be computed for.
    :param sentiment: base.Callable sentiment analysis method.
    :return features: Features dict to return.
    """
    features = {}
    polarity = sentiment.polarity_scores(doc)
    features.update({'SENTIMENT_POS': polarity['pos']})
    features.update({'SENTIMENT_NEG': polarity['neg']})
    features.update({'SENTIMENT_COMPOUND': polarity['compound']})

    return features


def head_of_token(parsed):
    """Retrieve the head of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"HEAD_OF_{0}".format(token): token.head.text for token in parsed}


def children_of_token(parsed):
    """Retrieve the children of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"children_of_{0}".format(token): "_".join([str(child) for child in token.children])
            for token in parsed}


def number_of_arcs(parsed):
    """Retrieve the number of right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"NO_RIGHT_ARCS_{0}".format(token): token.n_rights,
                "NO_LEFT_ARCS_{0}".format(token): token.n_lefts,
                "NO_TOTAL_ARCS_{0}".format(token): int(token.n_rights) + int(token.n_lefts)}
        features.update(arcs)
    return features


def arcs(parsed):
    """Retrieve the right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"RIGHT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.rights]),
                "LEFT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.lefts])}
        features.update(arcs)
    return features


def get_brown_clusters(doc: base.List[str], cluster: base.Dict[str, str], salt: str = '') -> base.List[str]:
    """Generate cluster for each word.
    :param doc: Document ebing procesed as a list.
    :param cluster: Cluster computed using clustering algorithm.
    :param salt: To add in front of the features.
    :return: base.Dictionary of clustered values."""
    if salt != '':
        salt = salt.upper() + '_'
    return Counter([salt + cluster.get(w, 'CLUSTER_UNK') for w in doc])


def word_syllable_counter(doc: base.DocType):
    """Simplistic syllable count.
    :param doc: Document to count syllables in.
    :return: String containing
    """
    word_counts = []
    vowels = 'aeiouy'
    exceptions = ['le', 'es', 'e']

    for w in doc:
        count = 0
        prev_char = None

        for i in w:
            if i == len(w) and (prev_char + w[i] in exceptions or w[i] in exceptions):
                prev_char = w[i]
                continue
            if (w[i] in vowels) and (prev_char not in vowels and not prev_char):
                prev_char = w[i]
                count += 1

        word_counts.append(count)
    return " ".join(word_counts)

from dataclasses import dataclass
from collections import namedtuple

@dataclass
class SENTIMENTS:
    _SentimentTuple = namedtuple("SentimentTuple", "review_label lexicon_label sign")
    pos = _SentimentTuple("POS", "positive", 1)
    neg = _SentimentTuple("NEG", "negative", -1)
    neut = _SentimentTuple(None, "neutral", 0)
@dataclass
class POLARITIES:
    _PolarityTuple = namedtuple("PolarityTuple", "lexicon_label lexicon_value")
    weak = _PolarityTuple("weaksubj", 1)
    strong = _PolarityTuple("strongsubj", 2)

CORRECT_CLASSIFICATION = "+"
INCORRECT_CLASSIFICATION = "-"

REVIEWS_BASEDIR = "data/reviews"
REVIEWS_IGNORE_TAGS = {"\n",}

TRAINING_DATA = "train"
TESTING_DATA = "test"

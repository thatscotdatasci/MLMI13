from enum import Enum

class SENTIMENTS(Enum):
    pos = "POS"
    neg = "NEG"

POLARITIES = {
    "weaksubj": 1,
    "strongsubj": 2
}
POLARITY_POSITIVE = "positive"
POLARITY_NEGATIVE = "negative"

CORRECT_CLASSIFICATION = "+"
INCORRECT_CLASSIFICATION = "-"

REVIEWS_BASEDIR = "data/reviews"
REVIEWS_IGNORE_TAGS = {"\n",}

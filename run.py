from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon


if __name__ == "__main__":
    corpus = MovieReviewCorpus(stemming=False,pos=False,review="data/reviews/NEG/cv001_19502.tag")
    lexicon = SentimentLexicon()

    threshold = 8
    lexicon.classify(corpus.reviews,threshold,magnitude=False)

    token_preds=lexicon.predictions
    token_preds

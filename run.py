from Corpora import MovieReviewCorpus
from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText

from Extensions import SVMDoc2Vec


if __name__ == "__main__":
    # retrieve corpus
    corpus=MovieReviewCorpus(stemming=False,pos=False)

    # use sign test for all significance testing
    signTest=SignTest()

    print("--- classifying reviews using sentiment lexicon  ---")

    # read in lexicon
    lexicon=SentimentLexicon()

    # on average there are more positive than negative words per review (~7.13 more positive than negative per review)
    # to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
    threshold=8

    ###########
    # Sentiment
    ###########

    # # question 0.1
    # lexicon.classify(corpus.reviews,threshold,magnitude=False)
    # token_preds=lexicon.predictions
    # print(f"token-only results: {lexicon.getAccuracy():.2f}")

    # lexicon.classify(corpus.reviews,threshold,magnitude=True)
    # magnitude_preds=lexicon.predictions
    # print(f"magnitude results:{lexicon.getAccuracy():.2f}")

    # # question 0.2
    # p_value=signTest.getSignificance(token_preds,magnitude_preds)
    # significance = "significant" if p_value < 0.05 else "not significant"
    # print(f"magnitude lexicon results are {significance} with respect to token-only")

    #############
    # Naive Bayes
    #############

    # question 1.0
    print("--- classifying reviews using Naive Bayes on held-out test set ---")
    NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
    NB.train(corpus.train)
    NB.test(corpus.test)
    # store predictions from classifier
    non_smoothed_preds=NB.predictions
    print(f"Accuracy without smoothing: {NB.getAccuracy():.2f}")

    # question 2.0
    # use smoothing
    NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
    NB.train(corpus.train)
    NB.test(corpus.test)
    smoothed_preds=NB.predictions
    # saving this for use later
    num_non_stemmed_features=len(NB.vocabulary)
    print(f"Accuracy using smoothing: {NB.getAccuracy():.2f}")


    # question 2.1
    # see if smoothing significantly improves results
    p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
    significance = "significant" if p_value < 0.05 else "not significant"
    print(f"results using smoothing are {significance} with respect to no smoothing")

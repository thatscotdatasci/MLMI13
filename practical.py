import os
import pickle

from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec

# whether to use corpus pickles
use_pickles = True

# retrieve corpus
corpus_pickle = "corpus.pkl"
if use_pickles and os.path.isfile(corpus_pickle):
    with open(corpus_pickle, 'rb') as f:
        corpus = pickle.load(f)
else:
    corpus=MovieReviewCorpus(stemming=False)
    with open(corpus_pickle, 'wb') as f:
        pickle.dump(corpus, f)

# use sign test for all significance testing
signTest=SignTest()

print("--- classifying reviews using sentiment lexicon  ---")

# read in lexicon
lexicon=SentimentLexicon()

# question 0.1
# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8

lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions
print(f"token-only results: {lexicon.getAccuracy():.5f}")

lexicon.classify(corpus.reviews,threshold,magnitude=True)
magnitude_preds=lexicon.predictions
print(f"magnitude results:{lexicon.getAccuracy():.5f}")

# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"magnitude lexicon results are {significance} with respect to token-only")


# question 1.0
print("--- classifying reviews using Naive Bayes on held-out test set ---")
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
# store predictions from classifier
non_smoothed_preds=NB.predictions
print(f"Accuracy without smoothing: {NB.getAccuracy():.5f}")


# question 2.0
# use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
smoothed_preds=NB.predictions
print(f"Accuracy using smoothing: {NB.getAccuracy():.5f}")


# question 2.1
# see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing are {significance} with respect to no smoothing")


# question 3.0
print("--- classifying reviews with NB using 10-fold cross-evaluation ---")
# using previous instantiated object
NB.crossValidate(corpus)
# saving this for use later
num_non_stemmed_features=len(NB.vocabulary)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.5f}")
print(f"Std. Dev: {NB.getStdDeviation():.5f}")


# # question 4.0
print("--- stemming corpus ---")
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus_pickle = "corpus_stem.pkl"
if use_pickles and os.path.isfile(stemmed_corpus_pickle):
    with open(stemmed_corpus_pickle, 'rb') as f:
        stemmed_corpus = pickle.load(f)
else:
    stemmed_corpus=MovieReviewCorpus(stemming=True)
    with open(stemmed_corpus_pickle, 'wb') as f:
        pickle.dump(stemmed_corpus, f)

print("--- classifying reviews with NB and stemming using 10-fold cross-evaluation ---")
NB.crossValidate(stemmed_corpus)
# saving this for use later
num_stemmed_features=len(NB.vocabulary)
stemmed_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.5f}")
print(f"Std. Dev: {NB.getStdDeviation():.5f}")


# # Q4.1
# see if stemming significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds,stemmed_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing are {significance} with respect to no smoothing")


# # Q4.2
print("--- determining the number of features before/after stemming ---")
print(f"features before stemming: {num_non_stemmed_features}")
print(f"features after stemming: {num_stemmed_features}")


# # question Q5.0
# use smoothing and bigrams
print("--- classifying reviews using Naive Bayes using smoothing with bigrams and trigrams on held-out test set ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
smoothed_preds=NB.predictions
print(f"Accuracy using smoothing and bigrams: {NB.getAccuracy():.5f}")

NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=True,discard_closed_class=False,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
smoothed_preds=NB.predictions
print(f"Accuracy using smoothing and bigrams and trigrams: {NB.getAccuracy():.5f}")

# cross-validate model using smoothing and bigrams
print("--- cross-validating naive bayes using smoothing and bigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False,pos=False)
NB.crossValidate(corpus)
# saving this for use later
num_bigrams_features=len(NB.vocabulary)
smoothed_and_bigram_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.5f}") 
print(f"Std. Dev: {NB.getStdDeviation():.5f}")

# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams are {significance} with respect to smoothing only")

# cross-validate model using smoothing and bigrams and trigrams
print("--- cross-validating naive bayes using smoothing and bigrams and trigrams ---")
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=True,discard_closed_class=False,pos=False)
NB.crossValidate(corpus)
# saving this for use later
num_trigrams_features=len(NB.vocabulary)
smoothed_and_trigram_preds=NB.predictions
print(f"Accuracy: {NB.getAccuracy():.5f}") 
print(f"Std. Dev: {NB.getStdDeviation():.5f}")

# see if bigrams and trigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_trigram_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams and trigrams are {significance} with respect to smoothing only")

# see if bigrams and trigrams significantly improves results on bigram NB only
p_value=signTest.getSignificance(smoothed_and_bigram_preds,smoothed_and_trigram_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using smoothing and bigrams and trigrams are {significance} with respect to bigrams only")

# Q5.1
print(f"features with bigrams: {num_bigrams_features}")
print(f"features with bigrams and trigrams: {num_trigrams_features}")


# Q6 and 6.1
print("--- classifying reviews using SVM on held-out test set ---")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False,pos=False)
SVM.train(corpus.train)
SVM.test(corpus.test)
# store predictions from classifier
svm_preds=SVM.predictions
print(f"Accuracy with SVM using unigrams: {SVM.getAccuracy():.5f}")

# SVM=SVMText(bigrams=True,trigrams=False,discard_closed_class=False,pos=False)
# SVM.train(corpus.train)
# SVM.test(corpus.test)
# print(f"Accuracy with SVM using additional bigrams: {SVM.getAccuracy():.5f}")

# SVM=SVMText(bigrams=True,trigrams=True,discard_closed_class=False,pos=False)
# SVM.train(corpus.train)
# SVM.test(corpus.test)
# print(f"Accuracy with SVM using additional bigrams and trigrams: {SVM.getAccuracy():.5f}")

print("--- classifying reviews using SVM with unigrams and 10-fold cross-eval ---")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False,pos=False)
SVM.crossValidate(corpus)
# saving this for use later
svm_preds=SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.5f}") 
print(f"Std. Dev: {SVM.getStdDeviation():.5f}")

# see if SVM significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds,svm_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using SVM {significance} with respect to smoothed NB")

# Q7
print("--- adding in POS information to corpus ---")

print("--- training nb on word+pos features ----")
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False,pos=True)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
print(f"Accuracy using NB on unigrams without smoothing and with POS: {NB.getAccuracy():.5f}")

NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False,pos=True)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
print(f"Accuracy using NB on unigrams with smoothing and POS: {NB.getAccuracy():.5f}")

print("--- training svm on word+pos features ----")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False,pos=True)
SVM.train(corpus.train)
SVM.test(corpus.test)
print(f"Accuracy with SVM with POS: {SVM.getAccuracy():.5f}")

print("--- classifying reviews using svm on word+pos and 10-fold cross-eval ---")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=False,pos=True)
SVM.crossValidate(corpus)
# saving this for use later
svm_pos_preds=SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.5f}") 
print(f"Std. Dev: {SVM.getStdDeviation():.5f}")

# see if POS significantly improves results on SVM only
p_value=signTest.getSignificance(svm_preds,svm_pos_preds)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results using POS tags {significance} with respect to SVM")

print("--- training nb discarding closed-class words ---")
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=True,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
print(f"Accuracy using NB without smoothing and discarding closed-class words: {NB.getAccuracy():.5f}")

NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=True,pos=False)
NB.train(corpus.train)
NB.test(corpus.test, verbose=False)
print(f"Accuracy using NB with smoothing and discarding closed-class words: {NB.getAccuracy():.5f}")

print("--- training svm discarding closed-class words ---")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=True,pos=False)
SVM.train(corpus.train)
SVM.test(corpus.test)
print(f"Accuracy with SVM discarding closed-class word: {SVM.getAccuracy():.5f}")

print("--- classifying reviews using svm discarding closed-class words and 10-fold cross-eval ---")
SVM=SVMText(bigrams=False,trigrams=False,discard_closed_class=True,pos=False)
SVM.crossValidate(corpus)
# saving this for use later
svm_closed_class=SVM.predictions
print(f"Accuracy: {SVM.getAccuracy():.5f}") 
print(f"Std. Dev: {SVM.getStdDeviation():.5f}")

# see if discarding closed-class words significantly improves results on SVM only
p_value=signTest.getSignificance(svm_preds,svm_closed_class)
significance = "significant" if p_value < 0.05 else "not significant"
print(f"results discardig closed-class words {significance} with respect to SVM")

# # question 8.0
# print("--- using document embeddings ---")

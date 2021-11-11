from Analysis import Evaluation
from Constants import SENTIMENTS, CORRECT_CLASSIFICATION, INCORRECT_CLASSIFICATION

from nltk.util import ngrams
import numpy as np
from sklearn import svm

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: boolean

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]


    def getPrior(self, reviews):
        """
        Determine the priors for POS and NEG classes by dividing the number of each review by
        the total number of reviews.

        Set the values in the self.prior dict.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        pos_count = 0
        neg_count = 0
        for sentiment, _ in reviews:
            if sentiment == SENTIMENTS.pos.review_label:
                pos_count += 1
            elif sentiment == SENTIMENTS.neg.review_label:
                neg_count += 1
            else:
                raise Exception("Found a review that this neither positive nor negative")
        self.prior = {
            SENTIMENTS.pos.review_label: pos_count/len(reviews),
            SENTIMENTS.neg.review_label: neg_count/len(reviews),
        }

    def getCondProb(self, reviews: list):
        """
        Determine the conditional probability of each word given the class using the
        frequency of occurrences of the word in the class corpus.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        initial_frequencies = {tuple(word for word, _ in token): 0 for token in self.vocabulary}
        word_frequencies = {
            SENTIMENTS.pos.review_label: initial_frequencies.copy(),
            SENTIMENTS.neg.review_label: initial_frequencies.copy(),
        }
    
        for sentiment, review in reviews:
            review_tokens = self.extractReviewTokens(review)
            review_words = [tuple(word for word, _ in token) for token in review_tokens]
            for words in review_words:
                word_frequencies[sentiment][words] += 1

        total_pos_count = sum(word_frequencies[SENTIMENTS.pos.review_label].values())
        total_neg_count = sum(word_frequencies[SENTIMENTS.neg.review_label].values())

        laplacian_k = 0
        if self.smoothing:
            laplacian_k = 1
            total_pos_count += laplacian_k*len(self.vocabulary)
            total_neg_count += laplacian_k*len(self.vocabulary)

        self.condProb = {
            SENTIMENTS.pos.review_label: {
                word: (word_frequencies[SENTIMENTS.pos.review_label][word]+laplacian_k)/total_pos_count for word in initial_frequencies.keys()
            },
            SENTIMENTS.neg.review_label: {
                word: (word_frequencies[SENTIMENTS.neg.review_label][word]+laplacian_k)/total_neg_count for word in initial_frequencies.keys()
            },
        }


    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for _, review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                # Return unigrams as single element tuple to match ngrams
                text.append((token,))
        if self.bigrams:
            for bigram in ngrams(review, 2, pad_left=True, pad_right=True, left_pad_symbol=('<s>', '<s>'), right_pad_symbol=('</s>', '</s>')): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review, 3, pad_left=True, pad_right=True, left_pad_symbol=('<s>', '<s>'), right_pad_symbol=('</s>', '</s>')): text.append(trigram)
        return text

    def create_vocab_dict(self):
        vocab_to_id = {}
        for word in self.vocabulary:
            vocab_to_id[word] = len(vocab_to_id)
        return vocab_to_id

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1
        # TODO Q2 (use switch for smoothing from self.smoothing)
        self.vocabulary = set()
        self.extractVocabulary(reviews)

        self.prior = {}
        self.getPrior(reviews)

        self.condProb = {}
        self.getCondProb(reviews)


    def test(self,reviews,verbose: bool = False):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        @param verbose: whether to print statements
        @type verbose: bool
        """
        # TODO Q1
        
        # Keep a record of words with zero probability, and those which did not appear in training
        zero_prob_words = set()
        words_not_in_training = set()

        for label, review in reviews:
            log_likelihood = {
                SENTIMENTS.pos.review_label: np.log(self.prior[SENTIMENTS.pos.review_label]),
                SENTIMENTS.neg.review_label: np.log(self.prior[SENTIMENTS.neg.review_label])
            }

            review_tokens = self.extractReviewTokens(review)
            review_words = [tuple(word for word, _ in token) for token in review_tokens]

            for word in review_words:
                for test_sentiment in [SENTIMENTS.pos.review_label, SENTIMENTS.neg.review_label]:
                    if word in self.condProb[test_sentiment]:
                        # Look-up the word probability calculated during training
                        # In the non-smoothing case this could be zero, which will raise a numpy warning when we take the log
                        word_prob = self.condProb[test_sentiment][word]
                        log_word_prob = np.log(word_prob)

                        if not word_prob > 0:
                            # Should only have zero probability words for non-smoothed run
                            # Exception thrown below if we have any such words during smoothed run
                            zero_prob_words.add(word)
                    else:
                        # If there is an unknown word in the test data then we should ignore it
                        # Ref: Jurafsky, Chapter 4
                        words_not_in_training.add(word)
                        log_word_prob = np.log(1)
                    
                    log_likelihood[test_sentiment] += log_word_prob

            prediction = max(log_likelihood, key=log_likelihood.get)

            correct_prediction = CORRECT_CLASSIFICATION if prediction == label else INCORRECT_CLASSIFICATION
            self.predictions.append(correct_prediction)

        if verbose:
            print(f"Identified {len(zero_prob_words)} with zero probability")
            print(f"Identified {len(words_not_in_training)} not found in training set")

        if len(zero_prob_words) > 0 and self.smoothing:
            # We should never have zero probability words when smoothing
            # Raise an exception if any are present
            raise Exception(f"Identified the following zero probability words whilst smoothing:\n\n {zero_prob_words}")


class SVMText(Evaluation):
    def __init__(self,bigrams,trigrams,discard_closed_class):
        """
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        self.svm_classifier = svm.SVC()
        self.predictions=[]
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def extractVocabulary(self,reviews):
        self.vocabulary = set()
        for sentiment, review in reviews:
            for token in self.extractReviewTokens(review):
                 self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getFeatures(self,reviews):
        """
        determine features and labels from training reviews.

        1. extract vocabulary (i.e. get features for training)
        2. extract features for each review as well as saving the sentiment
        3. append each feature to self.input_features and each label to self.labels
        (self.input_features will then be a list of list, where the inner list is
        the features)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        self.input_features = []
        self.labels = []

        # TODO Q6.

    def train(self,reviews):
        """
        train svm. This uses the sklearn SVM module, and further details can be found using
        the sci-kit docs. You can try changing the SVM parameters. 

        @param reviews: training data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set.
        self.getFeatures(reviews)

        # reset SVM classifier and train SVM model
        self.svm_classifier = svm.SVC()
        self.svm_classifier.fit(self.input_features, self.labels)

    def test(self,reviews):
        """
        test svm

        @param reviews: test data
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # TODO Q6.1

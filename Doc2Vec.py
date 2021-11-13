import os
from collections import Counter
from glob import glob

import gensim
import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score

from Constants import SENTIMENTS, TRAINING_DATA, TESTING_DATA

class Doc2Vec:

    def __init__(self, vector_size: int = 50, min_count: int = 2, epochs: int = 40):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs

        self.train_corpus = []
        self.test_corpus = []
        self.model = None

    def load_data(self, training_files, testing_files):
        # Load the training data
        for f in training_files:
            self.train_corpus.append(self.read_corpus(f, os.path.basename(f)))

        # Load the test data
        for f in testing_files:
            self.test_corpus.append(self.read_corpus(f))

    def read_corpus(self, f_path, token=None):
        with open(f_path, encoding='utf-8') as f:
            tokens = gensim.utils.simple_preprocess(f.read())
            if token is not None:
                return gensim.models.doc2vec.TaggedDocument(tokens, [token])
            else:
                return tokens
    
    def train(self):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs, workers=36)
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def test(self):
        ranks = []
        errors = {}
        for i, tag in enumerate([doc.tags[0] for doc in self.train_corpus]):
            inferred_vector = self.model.infer_vector(self.train_corpus[i].words)
            sims = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))
            rank = [doc_tag for doc_tag, _ in sims].index(tag)
            ranks.append(rank)

            if rank != 0:
                pred_tag, pred_sim = sims[0]
                errors[tag] = (rank, sims[rank][1], pred_tag, pred_sim, {doc.tags[0]: doc.words for doc in self.train_corpus if doc.tags[0] in [tag,pred_tag]})
        
        return Counter(ranks), errors

    def generate_embeddings(self, training_pos_files: list, training_neg_files: list, testing_pos_files: list, testing_neg_files: test):
        if not self.model:
            raise Exception("The Doc2Vec model has not been trained yet")

        training_pos_corpus = [self.read_corpus(f) for f in training_pos_files] 
        training_pos_embeddings = [self.model.infer_vector(doc) for doc in training_pos_corpus]

        training_neg_corpus = [self.read_corpus(f) for f in training_neg_files]
        training_neg_embeddings = [self.model.infer_vector(doc) for doc in training_neg_corpus]

        testing_pos_corpus = [self.read_corpus(f) for f in testing_pos_files]
        testing_pos_embeddings = [self.model.infer_vector(doc) for doc in testing_pos_corpus]

        testing_neg_corpus = [self.read_corpus(f) for f in testing_neg_files]
        testing_neg_embeddings = [self.model.infer_vector(doc) for doc in testing_neg_corpus]

        embeddings = {
            TRAINING_DATA: {
                SENTIMENTS.pos.review_label: training_pos_embeddings,
                SENTIMENTS.neg.review_label: training_neg_embeddings
            },
            TESTING_DATA: {
                SENTIMENTS.pos.review_label: testing_pos_embeddings,
                SENTIMENTS.neg.review_label: testing_neg_embeddings
            }
        }
        
        return embeddings


class SVMSklearn:
    def __init__(self):
        """
        SKlearn version of SVM implementation

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean

        @param pos: use POS information?
        @type pos: boolean
        """
        self._model=None

    def train(self, X, y):
        self._model = SVC()
        self._model.fit(X, y)

    def test(self, X, y):
        preds = self._model.predict(X)
        print(np.mean(preds == y))


    def cross_validate(self, X, y):
        scores = cross_val_score(self._model, X, y, cv=5)
        print(scores)


class TSNESklearn:
    def __init__(self):
        self._model = None

    def fit_transform(self, X):
        self._model = TSNE(n_components=2, learning_rate='auto', init='random')
        return self._model.fit_transform(X)

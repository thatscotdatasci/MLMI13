from inspect import Attribute
import os
from collections import Counter

import gensim
import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV

from Constants import SENTIMENTS, TRAINING_DATA, TESTING_DATA

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        d2v_training_files: list = [],
        dm: int = 1,
        window: int = 3,
        dbow_words: int = 1,
        vector_size: int = 50,
        min_count: int = 2,
        epochs: int = 1
    ):
        self.d2v_training_files = d2v_training_files

        # Doc2Vec settings
        self.dm = dm
        self.window = window
        self.dbow_words = dbow_words
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs

        # Parameter initialisation
        self.train_corpus = []
        self.test_corpus = []
        self.model = None

    def load_data(self, training_files: list, testing_files: list = []):
        """Load the training and (optional) test data.

        :param training_files: List of paths for the training files
        :type training_files: list
        :param testing_files: List of paths for the testing files, defaults to []
        :type testing_files: list, optional
        """
        # Load the training data
        self.train_corpus = []
        for f in training_files:
            self.train_corpus.append(self.read_corpus(f, os.path.basename(f)))

        # Load the test data - if provided
        self.test_corpus = []
        for f in testing_files:
            self.test_corpus.append(self.read_corpus(f))

    def read_corpus(self, f_path: str, label: str = None) -> list:
        """Process the training/test.

        :param f_path: Path to the file
        :type f_path: str
        :param label: Label for the type of document, defaults to None
        :type label: str, optional
        :return: Tokens extracted from doc
        :rtype: list
        """
        with open(f_path, encoding='utf-8') as f:
            tokens = gensim.utils.simple_preprocess(f.read())
            if label is not None:
                return gensim.models.doc2vec.TaggedDocument(tokens, [label])
            else:
                return tokens

    def train(self):
        """
        Train the Doc2Vec model
        """
        self.model = gensim.models.doc2vec.Doc2Vec(
            dm=self.dm,
            window=self.window,
            dbow_words=self.dbow_words,
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=8
        )
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def test(self):
        """
        Test the Doc2Vec model by seeing if it can correctly identify that the test documents are closest to themselves.
        """
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

    def fit(self, X: list, y: list = None):
        """Implementing sklearn fit method for Doc2Vec

        :param X: training documents - not that these are the documents that will be used to train the subsequent classifier, NOT Doc2Vec
        :type X: list
        :param y: classification labels, defaults to None
        :type y: list, optional
        :return: current object - as required by sklearn
        :rtype: Doc2Vec
        """
        self.load_data(self.d2v_training_files, [])
        self.train()
        return self
    
    def transform(self, X: list) -> np.array:
        """Implementing sklearn transform method for Doc2Vec

        :param X: list of documents to be transformed
        :type X: list
        :return: vectorised representation of the documents
        :rtype: np.array
        """
        corpus = (self.read_corpus(f) for f in X)
        return np.array([self.model.infer_vector(doc) for doc in corpus])
    

    def generate_embeddings(self, training_pos_files: list, training_neg_files: list, testing_pos_files: list, testing_neg_files: test, epochs=5):
        if not self.model:
            raise Exception("The Doc2Vec model has not been trained yet")

        training_pos_corpus = (self.read_corpus(f) for f in training_pos_files)
        training_pos_embeddings = [self.model.infer_vector(doc, epochs=epochs) for doc in training_pos_corpus]

        training_neg_corpus = (self.read_corpus(f) for f in training_neg_files)
        training_neg_embeddings = [self.model.infer_vector(doc, epochs=epochs) for doc in training_neg_corpus]

        testing_pos_corpus = (self.read_corpus(f) for f in testing_pos_files)
        testing_pos_embeddings = [self.model.infer_vector(doc, epochs=epochs) for doc in testing_pos_corpus]

        testing_neg_corpus = (self.read_corpus(f) for f in testing_neg_files)
        testing_neg_embeddings = [self.model.infer_vector(doc, epochs=epochs) for doc in testing_neg_corpus]

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
        """
        self._model=None

    def train(self, X, y):
        self._model = SVC()
        self._model.fit(X, y)

    def test(self, X, y):
        preds = self._model.predict(X)
        print(np.mean(preds == y))


    def cross_validate(self, X, y):
        scores = cross_val_score(self._model, X, y, cv=3)
        print(scores)

class GensimSVMSklearn:
    def __init__(
        self,
        d2v_training_files: list,
        dm: int = 1,
        window: int = 3,
        dbow_words: int = 1,
        vector_size: int = 50,
        min_count: int = 2,
        epochs: int = 1
    ):
        self.d2v_training_files = d2v_training_files

        # Doc2Vec settings
        self.dm = dm
        self.window = window
        self.dbow_words = dbow_words
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs

    @property
    def _pipeline(self):
        return Pipeline([
            ('doc2vec', Doc2Vec(
                d2v_training_files=self.d2v_training_files,
                dm = self.dm,
                window = self.window,
                dbow_words = self.dbow_words,
                vector_size = self.vector_size,
                min_count = self.min_count,
                epochs = self.epochs,
            )),
            ('svc', SVC(verbose=True)),
        ])

    def train(self, X, y):
        self._pipeline.fit(X, y)

    def test(self, X, y):
        preds = self._pipeline.predict(X)
        print(np.mean(preds == y))

    def cross_validate(self, X, y, folds: int = 3):
        scores = cross_val_score(self._pipeline, X, y, cv=folds)
        print(scores)

    def grid_search(self, X, y, params: dict):
        self.gs = GridSearchCV(self._pipeline, params, n_jobs=-1)
        self.gs.fit(X, y) 

class TSNESklearn:
    def __init__(self):
        self._model = None

    def fit_transform(self, X):
        self._model = TSNE(n_components=2, learning_rate='auto', init='random')
        return self._model.fit_transform(X)

import os
import pickle
from glob import glob
import logging

import numpy as np

from Doc2Vec import GensimSVMSklearn
from Constants import SENTIMENTS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

pickle_dir = os.path.join('d2v_models', '031221.large_gs')

#####################
# Original review data
#####################
og_base_dir = os.path.join('data', 'reviews')
og_pos_dir = os.path.join(og_base_dir, 'POS')
og_neg_dir = os.path.join(og_base_dir, 'NEG')

og_training_pos_files = glob(os.path.join(og_pos_dir, 'cv[0-8]*.txt'))
og_training_neg_files = glob(os.path.join(og_neg_dir, 'cv[0-8]*.txt'))

og_testing_pos_files = glob(os.path.join(og_pos_dir, 'cv9*.txt'))
og_testing_neg_files = glob(os.path.join(og_neg_dir, 'cv9*.txt'))

og_y_train = np.array([
    *[SENTIMENTS.pos.review_label]*len(og_training_pos_files),
    *[SENTIMENTS.neg.review_label]*len(og_training_neg_files)
])
og_y_test = np.array([
    *[SENTIMENTS.pos.review_label]*len(og_testing_pos_files),
    *[SENTIMENTS.neg.review_label]*len(og_testing_neg_files)
])

###########
# IMDB data
###########
imdb_base_dir = 'imdb'
imdb_pos_dir = 'pos'
imdb_neg_dir = 'neg'
imdb_unsup_dir = 'unsup'

imdb_train_dir = os.path.join(imdb_base_dir, 'train')
imdb_train_pos_dir = os.path.join(imdb_train_dir, imdb_pos_dir)
imdb_train_neg_dir = os.path.join(imdb_train_dir, imdb_neg_dir)

imdb_test_dir = os.path.join(imdb_base_dir, 'test')
imdb_test_pos_dir = os.path.join(imdb_test_dir, imdb_pos_dir)
imdb_test_neg_dir = os.path.join(imdb_test_dir, imdb_neg_dir)

imdb_training_pos_files = glob(os.path.join(imdb_train_pos_dir, '*.txt'))
imdb_training_neg_files = glob(os.path.join(imdb_train_neg_dir, '*.txt'))

imdb_testing_pos_files = glob(os.path.join(imdb_test_pos_dir, '*.txt'))
imdb_testing_neg_files = glob(os.path.join(imdb_test_neg_dir, '*.txt'))

imdb_unsup_files = glob(os.path.join(imdb_base_dir, imdb_train_dir, imdb_unsup_dir, '*.txt'))

imdb_y_train = np.array([
    *[SENTIMENTS.pos.review_label]*len(imdb_training_pos_files),
    *[SENTIMENTS.neg.review_label]*len(imdb_training_neg_files)
])
imdb_y_test = np.array([
    *[SENTIMENTS.pos.review_label]*len(imdb_testing_pos_files),
    *[SENTIMENTS.neg.review_label]*len(imdb_testing_neg_files)
])

d2v_training_files = [
    *imdb_training_pos_files,
    *imdb_training_neg_files,
    *imdb_testing_pos_files,
    *imdb_testing_neg_files,
    *imdb_unsup_files
]
d2v_testing_files = []

#############
# Grid Search
#############

gs_params = {
    'doc2vec__epochs': (10,),
    'doc2vec__infer_epochs': (10,),
    'doc2vec__vector_size': (50,100,200),
    'doc2vec__dm': (0,),
    'doc2vec__dm_concat': (0,1),
    'doc2vec__dbow_words': (1,),
    'doc2vec__window': (5,),
    'doc2vec__min_count': (5,)
}
gensim_sklearn = GensimSVMSklearn(d2v_training_files=d2v_training_files, verbose=True)
gensim_sklearn.grid_search([*og_training_pos_files, *og_training_neg_files], og_y_train, gs_params)

print(gensim_sklearn.gs.best_params_)
print(gensim_sklearn.gs.best_score_)

with open(os.path.join(pickle_dir, 'd2v_imdb.pkl'), 'wb') as f:
    pickle.dump(gensim_sklearn, f)

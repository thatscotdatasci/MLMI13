import os
import pickle
from glob import glob
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Doc2Vec import Doc2Vec, SVMSklearn, TSNESklearn
from Constants import SENTIMENTS, TRAINING_DATA, TESTING_DATA

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

######################
# Original review data
######################
# base_dir = os.path.join('data', 'reviews')
# pos_dir = os.path.join(base_dir, 'POS')
# neg_dir = os.path.join(base_dir, 'NEG')

# training_pos_files = glob(os.path.join(pos_dir, 'cv[0-8]*.txt'))
# training_neg_files = glob(os.path.join(neg_dir, 'cv[0-8]*.txt'))

# testing_pos_files = glob(os.path.join(pos_dir, 'cv9*.txt'))
# testing_neg_files = glob(os.path.join(neg_dir, 'cv9*.txt'))

# d2v_training_files = [
#     *training_pos_files,
#     *training_neg_files,
#     *testing_pos_files,
#     *testing_neg_files
# ]
# d2v_testing_files = []

###########
# IMDB data
###########
base_dir = 'imdb'
pos_dir = 'pos'
neg_dir = 'neg'

train_dir = os.path.join(base_dir, 'train')
train_pos_dir = os.path.join(train_dir, pos_dir)
train_neg_dir = os.path.join(train_dir, neg_dir)

test_dir = os.path.join(base_dir, 'train')
test_pos_dir = os.path.join(test_dir, pos_dir)
test_neg_dir = os.path.join(test_dir, neg_dir)

training_pos_files = glob(os.path.join(train_pos_dir, '*.txt'))
training_neg_files = glob(os.path.join(train_neg_dir, '*.txt'))
testing_pos_files = glob(os.path.join(test_pos_dir, '*.txt'))
testing_neg_files = glob(os.path.join(test_neg_dir, '*.txt'))

d2v_training_files = [
    *training_pos_files,
    *training_neg_files,
    *testing_pos_files,
    *testing_neg_files
]
d2v_testing_files = []

#########
# Doc2Vec
#########

use_d2v_pickle = True
d2v_pickle_name = 'doc2vec_model.pkl'

if use_d2v_pickle and os.path.isfile(d2v_pickle_name):
    with open(d2v_pickle_name, 'rb') as f:
        d2v = pickle.load(f)
else:
    d2v = Doc2Vec(vector_size=50, min_count=2)

    logger.info('Loading data')
    d2v.load_data(training_files=d2v_training_files, testing_files=d2v_testing_files)

    logger.info('Training doc2vec')
    d2v.train()

    with open(d2v_pickle_name, 'wb') as f:
        pickle.dump(d2v, f)

# logger.info('Testing doc2vec on the training data')
# ranks_count, errors = d2v.test()
# logger.info(ranks_count)

use_embeddings_pickle = True
embeddings_pickle_name = 'doc2vec_embeddings.pkl'

if use_embeddings_pickle and os.path.isfile(embeddings_pickle_name):
    with open(embeddings_pickle_name, 'rb') as f:
        embeddings = pickle.load(f)
else:
    logger.info('Obtaining embeddings')
    embeddings = d2v.generate_embeddings(
        training_pos_files=training_pos_files,
        training_neg_files=training_neg_files,
        testing_pos_files=testing_pos_files,
        testing_neg_files=testing_neg_files
    )

    with open(embeddings_pickle_name, 'wb') as f:
        pickle.dump(embeddings, f)

X_train = np.array([
    *embeddings[TRAINING_DATA][SENTIMENTS.pos.review_label],
    *embeddings[TRAINING_DATA][SENTIMENTS.neg.review_label]
])
y_train = np.array([
    *[SENTIMENTS.pos.review_label]*len(embeddings[TRAINING_DATA][SENTIMENTS.pos.review_label]),
    *[SENTIMENTS.neg.review_label]*len(embeddings[TRAINING_DATA][SENTIMENTS.neg.review_label])
])

X_test = np.array([
    *embeddings[TESTING_DATA][SENTIMENTS.pos.review_label],
    *embeddings[TESTING_DATA][SENTIMENTS.neg.review_label]
])
y_test = np.array([
    *[SENTIMENTS.pos.review_label]*len(embeddings[TESTING_DATA][SENTIMENTS.pos.review_label]),
    *[SENTIMENTS.neg.review_label]*len(embeddings[TESTING_DATA][SENTIMENTS.neg.review_label])
])

svm = SVMSklearn()
svm.train(X_train, y_train)
svm.cross_validate(X_train,y_train)
svm.test(X_train, y_train)
svm.test(X_test, y_test)

X = np.vstack((X_train, X_test))
tsne = TSNESklearn()
tsne_results = tsne.fit_transform(X)
tsne_df = pd.DataFrame({
    'tsne-2d-one': tsne_results[:,0],
    'tsne-2d-two': tsne_results[:,1]
})

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=tsne_df,
    legend="full",
    alpha=0.3
)

logger.info('Done')

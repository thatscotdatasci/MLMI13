from Constants import REVIEWS_BASEDIR, SENTIMENTS, REVIEWS_IGNORE_TAGS

import os, codecs, sys
from glob import glob
from nltk.stem.porter import PorterStemmer

class MovieReviewCorpus():
    
    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds={}
        # files that contained tokens which could not be processes
        self.rejects=[]
        # files which failed to be processes entirely
        self.failed=[]
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()
        
    def process_tag(tag, stem: bool = False):
        """
        Takes in an entry from a .tag file and processes it.
        """
        stripped = tag.strip()

        if stripped:
            split = stripped.split()
            if len(split) == 2:
                word, pos_tag = stripped.split()
            else:
                return False, tag
        else:
            # Return the non-processed tag
            return False, tag

        if stem:
            token = self.stemmer.stem(word)
        else:
            token = word.lower()
        return True, (token, pos_tag)
    
    def process_tag_file(file, stem: bool = False):
        """
        Processes a .tag file
        """
        token_tags = []
        rejects = []
        with open(file) as f:
            entries = (process_tag(l, stem=stem) for l in f.readlines() if l not in REVIEWS_IGNORE_TAGS)
            for e in entries:
                success, result = e
                if success:
                    token_tags.append(result)
                else:
                    rejects.append(result)
        return token_tags, rejects

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)

        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        
        for sentiment in SENTIMENTS:
            files = glob(os.path.join(REVIEWS_BASEDIR, sentiment, "*.tag"))
            for file in files:
                try:
                    token_tags, rejected_tags = self.process_file(file, stem=True)
                    
                    self.reviews.append((sentiment, token_tags))
                    
                    basename = str(os.path.basename(file))
                    if basename.startswith("cv9"):
                        self.test.append((sentiment, token_tags))
                    else:
                        self.train.append((sentiment, token_tags))
                    
                    if rejected_tags:
                        self.rejects.append((file, rejected_tags))
                    
                except Exception as e:
                    self.failed.append((e, file))
                    continue

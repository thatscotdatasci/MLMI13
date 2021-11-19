from Constants import REVIEWS_BASEDIR, SENTIMENTS, REVIEWS_IGNORE_TAGS

import os, codecs, sys
from collections import defaultdict
from typing import Tuple, Union, final

from glob import glob
from nltk.stem.porter import PorterStemmer

class MovieReviewCorpus():
    
    def __init__(self, stemming: bool = False, pos: bool = False, review: str = None):
        """
        Initialisation of movie review corpus.

        @param stemming: use porter's stemming? Defaults to False
        @type stemming: boolean

        @param pos: use pos tagging? Defaults to False
        @type pos: boolean

        @param review: process a single, specified review
        @type pos: str
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds=defaultdict(list)
        # files that contained tokens which could not be processes
        self.rejects=[]
        # files which failed to be processes entirely
        self.failed=[]
        # Use porter's stemming
        self.stemming = stemming
        # Keep POS
        self.pos = pos
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # import movie reviews
        self.get_reviews(review=review)
        
    def _process_tag(self, tag: str) -> Union[Tuple[bool, Tuple[str, str]], Tuple[bool, str]]:
        """
        Takes a line from a .tag file and processes it.

        :param tag: tag entry from a .tag file
        :type tag: str
        :return: Tuple with first element indicating if processing was successful.
        :rtype: Union[Tuple[bool, Tuple[str, str]], Tuple[bool, str]]
        """
        # Strip the tag of leading/trailing whitespace
        stripped = tag.strip()

        # If there is anything left then continue processing the tag
        if stripped:
            # Split based on internal whitespace chars 
            split = stripped.split()

            # Expecting to have two elements: the word and its tag
            if len(split) == 2:
                word, pos_tag = stripped.split()
            else:
                return False, tag
        else:
            # Return the non-processed tag
            return False, tag

        # Apply the stemmer if stem is true, else just lowercase the word
        if self.stemmer:
            token = self.stemmer.stem(word)
        else:
            token = word.lower()

        if self.pos:
            return True, (token, pos_tag)
        else:
            return True, token
    
    def _process_tag_file(self, filepath: str) -> Tuple[list, list]:
        """
        Processes all of the tags in a .tag file.

        :param filepath: path of the file to be processed
        :type filepath: str
        :param stem:  Apply PorterStemmer, defaults to False
        :type stem: bool, optional
        :return: The identified tags, and the tags that could not be proccessed
        :rtype: Tuple[list, list]
        """
        # Raise an exception if the file does not exist
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        # Initialise empty lists to hold the processed tags and rejected tags
        token_tags = []
        rejects = []

        # Open and process the file
        with open(filepath) as f:

            # Create generator for processeing each line in the file
            # Ignore any entries that appear in REVIEWS_IGNORE_TAGS
            entries = (self._process_tag(l) for l in f.readlines() if l not in REVIEWS_IGNORE_TAGS)

            # If the tag was processed successfully then add to token_tags, else add to rejects
            for e in entries:
                success, result = e
                if success:
                    token_tags.append(result)
                else:
                    rejects.append(result)
        
        return token_tags, rejects

    def get_reviews(self, review: str = None):
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
        
        # Initialise an empty list of the files to be processed
        file_dict = {}

        # If a specific review has been specified then only process this one
        # Otherwise, look for all reviews in the POS and NEG directories
        if review:
            sentiment = review.split("/")[2]
            file_dict[sentiment] = [review]
        else:
            # For each of the "POS" and "NEG" folders
            for sentiment in [SENTIMENTS.pos.review_label, SENTIMENTS.neg.review_label]:
                
                # Identify the files which have the .tag extension
                sent_files = glob(os.path.join(REVIEWS_BASEDIR, sentiment, "*.tag"))
                print(f"Identified {len(sent_files)} {sentiment} files to be processed")

                file_dict[sentiment] = sent_files

        # Process each .tag file
        for sentiment, files in file_dict.items():

            # Print what files are being processed
            # List them if there are fewer than 20
            print(f"Processing {sentiment} files")
            if len(files) < 20:
                print(files)

            for file in files:
                # Attempt to process the file; add to self.failed if any exceptions are thrown
                try:
                    token_tags, rejected = self._process_tag_file(file)   
                except Exception as e:
                    self.failed.append((e, file))
                    continue

                result = (sentiment, token_tags)

                # Add results to self.reviews
                self.reviews.append(result)
                
                # Add results to self.test if the filename starts cv9, else add to self.train
                basename = str(os.path.basename(file))
                if basename.startswith("cv9"):
                    self.test.append(result)
                else:
                    self.train.append(result)

                # Add the results to the appropriate entry of the self.fold dict, based on the filename
                assert basename[2].isdigit()
                self.folds[basename[2]].append(result)
                
                # If any tags were rejected then add these to self.rejects
                if rejected:
                    self.rejects.append((file, rejected))

import math,sys
from itertools import cycle

from Corpora import MovieReviewCorpus

class Evaluation():
    """
    general evaluation class implemented by classifiers
    """
    def crossValidate(self, corpus: MovieReviewCorpus, verbose: bool = False):
        """
        function to perform 10-fold cross-validation for a classifier.
        each classifier will be inheriting from the evaluation class so you will have access
        to the classifier's train and test functions.

        1. read reviews from corpus.folds and store 9 folds in train_files and 1 in test_files
        2. pass data to self.train and self.test e.g., self.train(train_files)
        3. repeat for another 9 runs making sure to test on a different fold each time

        @param corpus: corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        """
        # reset predictions
        self.predictions=[]

        # Determine how many folds need to be processed
        num_folds = len(corpus.folds)

        # Create an iterator to cycle through the folds
        cycle_folds = cycle(sorted(corpus.folds.keys()))
        
        # TODO Q3
        for _ in range(num_folds):
            # Set the test_files
            test_fold = next(cycle_folds)
            test_files = corpus.folds[test_fold]

            if verbose:
                print(f'Test fold: {test_fold}')

            # Set the train_files
            train_files = []
            for _ in range(num_folds-1):
                train_fold = next(cycle_folds)
                train_files.extend(corpus.folds[train_fold])

                if verbose:
                    print(f'Train fold: {train_fold}')
            
            # Perform training and testing
            self.train(train_files)
            self.test(test_files)

            # Move the iterator one forward
            next(cycle_folds)
            

    def getStdDeviation(self):
        """
        get standard deviation across folds in cross-validation.
        """
        # get the avg accuracy and initialize square deviations
        avgAccuracy,square_deviations=self.getAccuracy(),0
        # find the number of instances in each fold
        fold_size=len(self.predictions)//10
        # calculate the sum of the square deviations from mean
        for fold in range(0,len(self.predictions),fold_size):
            square_deviations+=(self.predictions[fold:fold+fold_size].count("+")/float(fold_size) - avgAccuracy)**2
        # std deviation is the square root of the variance (mean of square deviations)
        return math.sqrt(square_deviations/10.0)

    def getAccuracy(self):
        """
        get accuracy of classifier.

        @return: float containing percentage correct
        """
        # note: data set is balanced so just taking number of correctly classified over total
        # "+" = correctly classified and "-" = error
        return self.predictions.count("+")/float(len(self.predictions))

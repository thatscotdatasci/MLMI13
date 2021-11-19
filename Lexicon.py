from decimal import InvalidOperation
from Analysis import Evaluation
from Constants import CORRECT_CLASSIFICATION, INCORRECT_CLASSIFICATION, POLARITIES, SENTIMENTS

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon = self.get_lexicon_dict()

    def get_lexicon_dict(self):
        lexicon_dict = {}
        with open('data/sent_lexicon', 'r') as f:
            for line in f:
                word = line.split()[2].split("=")[1]
                polarity = line.split()[5].split("=")[1]
                magnitude = line.split()[0].split("=")[1]
                lexicon_dict[word] = [magnitude, polarity]
        return lexicon_dict

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["negative","strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """
        # reset predictions
        self.predictions=[]
        self.breakdowns = []
        
        # TODO Q0

        for review in reviews:
            label, content = review

            magnitude_sum = 0
            polarity_sum = 0
            review_breakdown = []

            for word in content:
                lexicon_record = self.lexicon.get(word)
                
                if lexicon_record:
                    magnitude_info, polarity_info = lexicon_record

                    if polarity_info == SENTIMENTS.pos.lexicon_label:
                        sign = SENTIMENTS.pos.sign
                    elif polarity_info == SENTIMENTS.neg.lexicon_label:
                        sign = SENTIMENTS.neg.sign
                    elif polarity_info == SENTIMENTS.neut.lexicon_label:
                        sign = SENTIMENTS.neut.sign
                    
                    polarity_sum += sign

                    magnitude_val = sign*POLARITIES.weak.lexicon_value if magnitude_info == POLARITIES.weak.lexicon_label else sign*POLARITIES.strong.lexicon_value
                    magnitude_sum += magnitude_val

                    review_breakdown.append((word, magnitude_info, polarity_info, sign, magnitude_val))

            self.breakdowns.append(review_breakdown)

            score = magnitude_sum if magnitude else polarity_sum
            prediction = SENTIMENTS.pos.review_label if score >= threshold else SENTIMENTS.neg.review_label

            correct_prediction = CORRECT_CLASSIFICATION if prediction == label else INCORRECT_CLASSIFICATION

            self.predictions.append(correct_prediction)

################################################
"""
CSCI 5832 - Natural Language Processing
HW3: Text Classification
Author: Dieu My Nguyen
Date: November 29, 2018

Description:

Training dataset:
"""
################################################

import sys
import re
import numpy as np
from collections import defaultdict
from itertools import product
from collections import Counter

################################################
     ### NAIVE BAYES TEXT CLASSIFIER ###

# Functions
def count_words(txt_file):
    '''
    Join all the reviews in a file and count occurences of each unigram.
    '''
    with open(txt_file, "r") as infile:
        reviews = infile.readlines()

    count_r = 0
    split_reviews = []
    for r in reviews:
        count_r += 1
        review = re.sub("[,.!?]", "", r)
        review = review.lower().split()
        split_reviews += review[1:]

    return split_reviews, Counter(split_reviews), count_r

def predict_class(review, class_prob, class_count, V, word_counts):
    prediction = 0
    for word in review[1:]:
        # Compute conditional prob in log space
        word_count = word_counts[word] + 1
        prob_word_given_class = np.log((word_count) / (class_count + V))
        prediction += prob_word_given_class
    return prediction + np.log(class_prob)

def make_decision(review, prob_pos, count_pos, review_count_pos, word_counts_pos,
                  prob_neg, count_neg, review_count_neg, word_counts_neg):
    # Compute probabilities for positive and negative classes
    predict_pos = predict_class(review, prob_pos, count_pos, review_count_pos, word_counts_pos)
    predict_neg = predict_class(review, prob_neg, count_neg, review_count_neg, word_counts_neg)

    if predict_pos > predict_neg:
        return "POS"
    elif predict_pos < predict_neg:
        return "NEG"
    else:
        return "ERROR SOMEWHERE"

def main():
    train_pos = "hotelPosT-train.txt"
    train_neg = "hotelNegT-train.txt"

    words_list_pos, word_counts_pos, review_count_pos = count_words(train_pos)
    words_list_neg, word_counts_neg, review_count_neg = count_words(train_neg)

    # For smoothing: size of set of unique words
    V = len(set(words_list_neg + words_list_pos))

    count_pos = len(words_list_pos)
    count_neg = len(words_list_neg)
    total_words = count_pos + count_neg

    prob_pos = review_count_pos / (review_count_pos + review_count_neg)
    prob_neg = review_count_neg / (review_count_pos + review_count_neg)

    # Read test file and output class
    test_file = "test.txt"
    with open(test_file, 'r') as test:
        test = test.readlines()

    output = ""
    for line in test:
        review = re.sub("[,.!]", "", line)
        review = review.lower().split()
        review_id = review[0].upper()
        review_output = "{}\t{}\n".format(review_id, make_decision(review, prob_pos, count_pos, review_count_pos, word_counts_pos,
                          prob_neg, count_neg, review_count_neg, word_counts_neg))
        output += review_output
    with open("nguyen-dieumy-assgn3-out.txt", "w") as outfile:
        outfile.write(output)

if __name__ == "__main__":
    main()

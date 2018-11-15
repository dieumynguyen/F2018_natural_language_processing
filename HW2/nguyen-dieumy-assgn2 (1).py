################################################
"""
CSCI 5832 - Natural Language Processing
HW2: Part-Of-Speech (POS) Tagging
Author: Dieu My Nguyen
Date: October 01, 2018

Description: Probabilistic POS tagger using the Viterbi algorithm with a
bigram-approach. Also includes a baseline "most frequent tag" system that
simply assigns to each input word the tag it was most frequently assigned
to in traning data. Viterbi approach includes Laplacian smoothing & dealing
with unknown words using UNK replacement and counts.

Training dataset: POS-tagged data from the Berkeley Restaurant corpus. POS tag
set is closed but expect test data to include new words.
"""
################################################

import sys
import numpy as np
from collections import defaultdict
from itertools import product

################################################
     ### BASELINE MOST-FREQUENT TAGGER ###

def get_word_pos_count(training_data):
    '''
    Given a training set, create a list of lists as lookup table.
    Each inside list is a word, its POS, and the frequency of this pairing.
    '''
    # Create list of tuple: (word, POS)
    tups_list = []
    for line in training_data:
        if line != '\n':
            split_line = line.strip().split('\t')
            word_pos = split_line[1], split_line[2]
            tups_list.append(word_pos)

    # Get count of each unique tuple
    count_set = dict((x, tups_list.count(x)) for x in set(tups_list))

    # Create list of [word, POS, count]
    word_pos_count = []
    for k in count_set.keys():
        k_list = list(k)
        k_list.append(count_set[k])
        word_pos_count.append(k_list)
    return word_pos_count

def get_most_frequent_tag(word_pos_count, input_word):
    '''
    Given word_pos_count from training data & an input word,
    get the most frequent POS by
    calling the function get_word_pos_count() to get the lookup table
    and match input word to most frequent POS.

    Return most frequent tag for that word.
    '''

    # Get word_pos_count lookup table
    # word_pos_count = get_word_pos_count(training_data)

    # Get max word_pos_count's tag
    counts = []
    for i, word_pos in enumerate(word_pos_count):
        counts.append(word_pos_count[i][2])
    most_freq = word_pos_count[np.argmax(counts)][1]

    # Matching input word to possible word-tag-count lists
    matching_l = []
    for l in word_pos_count:
        if l[0] == input_word:
            matching_l.append(l)
    # print('{} \n'.format(matching_l))
    # Find most frequent POS tag
    most_frequent_tag = ''
    # Dealing with unseen words (matching_l: []), assign POS 'UNK'
    if len(matching_l) == 0:
        most_frequent_tag = most_freq
    else:
        # Find max count and tag that POS to word
        max_count = matching_l[0][2] # Temporary max
        for match in matching_l:
            if match[2] > max_count:
                most_frequent_tag = match[1]
            elif match[2] == max_count:
                most_frequent_tag = match[1]
    return most_frequent_tag

################################################
    ### BIGRAM-BASED VITERBI TAGGER ###

### Part 1. Process training data to extract counts
### & generate transition and emission probabilities

def build_words_tags_sets(data, word_idx, tag_idx):
    # Create a set of tags and set of words to process unseen words & get counts
    set_of_tags = set()
    set_of_tags.add("<s>"); set_of_tags.add("</s>")
    set_of_words = set()
    total_list_of_words = []

    # Add in UNK word class
    set_of_words.add("UNK")

    for line in data:
        if line != '\n':
            word_tag = line.strip().split('\t')
            set_of_tags.add(word_tag[tag_idx])
            set_of_words.add(word_tag[word_idx])
            total_list_of_words.append(word_tag[word_idx])
    return set_of_tags, set_of_words, total_list_of_words

def calculate_transition(tag_tag_count, tag_count):
    # Calculate transition probabilities
    # P(t_i | t_{i-1}) = C(t_{i-1, t_i}) / C(t_{i-1})
    transition_probs = {}
    for tag_tag in tag_tag_count:
        count_tag1tag2 = tag_tag_count[tag_tag]
        count_tag1 = tag_count[tag_tag[0]]
        transition_probs[tag_tag] = count_tag1tag2 / count_tag1
    return transition_probs

def calculate_emission(word_tag_count, tag_count):
    # Calculate emission probabilities
    # P(w | t) = C(t, w) / C(t)
    # word-tag pairings with 0 prob not included, will be 0 by default if not in dict
    # No smoothing yet
    emission_probs = {}
    for word_tag in word_tag_count:
        count_word_tag = word_tag_count[word_tag]
        count_tag = tag_count[word_tag[1]]
        emission_probs[word_tag] = count_word_tag / count_tag
    return emission_probs

def laplace_smooth_transition(tag_tag_count, tag_count, word_count):
    vocab_size = len(set(word_count))

    transition_probs = {}
    for tag_tag in tag_tag_count:
        count_tag1tag2 = tag_tag_count[tag_tag]
        count_tag1 = tag_count[tag_tag[0]]
        new_prob = (count_tag1tag2 + 1) / (count_tag1 + vocab_size)
        transition_probs[tag_tag] = new_prob
    return transition_probs

def laplace_smooth_emission(word_tag_count, tag_count, word_count):
    vocab_size = len(set(word_count))

    emission_probs = {}
    for word_tag in word_tag_count:
        count_word_tag = word_tag_count[word_tag]
        count_tag = tag_count[word_tag[1]]
        new_prob = (count_word_tag + 1) / (count_tag + vocab_size)
        emission_probs[word_tag] = new_prob
    return emission_probs

def build_lists_and_probabilities(train_data, set_of_tags, set_of_words, unk_count, smoothing=True):

    # Get counts of tag, words, word-tags
    word_tag_count = defaultdict(int) # C(t, w)
    word_count = defaultdict(int) # C(w)
    tag_count = defaultdict(int) # C(t)

    tag_sequence = ["<s>"]  # Sequence of tags, a sentence ends with .
    tag_sequence_count = 0
    tag_tag_count = defaultdict(int) # C(t_{i-1}, t_i)

    # print('{}'.format(train_data[0][2:]))

    for line in train_data:
         if line != '\n':
            word_tag = line.strip().split('\t')

            word_tag_count[(word_tag[1], word_tag[2])] += 1
            word_count[word_tag[1]] += 1
            tag_count[word_tag[2]] += 2

            # Get all tags from all lines
            tag_sequence.append(word_tag[2])
            if word_tag[1] == ".":
                tag_sequence.append("</s>")
                for i in range(0, len(tag_sequence)-1):
                    tag_tag_count[(tag_sequence[i], tag_sequence[i+1])] += 1
                tag_sequence = ["<s>"]
                tag_sequence_count += 1

    tag_count["<s>"] = tag_sequence_count
    tag_count["</s>"] = tag_sequence_count

    # For UNK words, count from corpus
    word_count["UNK"] = unk_count

    # Combine tag set and word set
    all_word_tag_combos = list(product(set_of_words, set_of_tags))
    for word_tag_combo in all_word_tag_combos:
        if word_tag_combo not in word_tag_count.keys():
            word_tag_count[word_tag_combo] = 0

    # Same thing for tag_tag_count
    all_tag_tag_combos = list(product(set_of_tags, set_of_tags))
    for tag_tag_combo in all_tag_tag_combos:
        if tag_tag_combo not in tag_tag_count.keys():
            tag_tag_count[tag_tag_combo] = 0

    # Get lists and probs
    # To smooth or to not smooth - that is the question...
    if smoothing:
        transition_probs = laplace_smooth_transition(tag_tag_count, tag_count, word_count)
        emission_probs = laplace_smooth_emission(word_tag_count, tag_count, word_count)

    # transition_probs = laplace_smooth_transition(tag_tag_count, tag_count, set_of_words)
    # emission_probs = laplace_smooth_emission(word_tag_count, tag_count, set_of_words)

    transition_probs = calculate_transition(tag_tag_count, tag_count)
    emission_probs = calculate_emission(word_tag_count, tag_count)

    list_of_tags = list(sorted(set_of_tags))
    list_of_words = list(sorted(set_of_words))

    return list_of_tags, list_of_words, word_count, transition_probs, emission_probs

#######------------------------------########

### Part 2. Viterbi algorithm

def viterbi_pos_tagger(list_of_tags, input_sentence, transition_probs, emission_probs):
    # Matrix with tags vs words
    p = np.zeros(shape=(len(list_of_tags), len(input_sentence)))
    # Matrix for backtrace
    back = np.zeros(shape=(len(list_of_tags), len(input_sentence)), dtype=np.int)

    # Initializing step: P(tag|start) * P(word1|tag)
    for tag_i, tag in enumerate(list_of_tags):
        # Fill first col of matrix p & back matrices
        tag_given_start = ('<s>', tag)
        word_given_tag = (input_sentence[0], tag)
        p[tag_i, 0] = transition_probs[tag_given_start] * emission_probs[word_given_tag]
        back[tag_i, 0] = 0

    # Recursion step: go through every tag for each token
    for word_i in range(1, len(input_sentence)):
        for tagi1, tag1 in enumerate(list_of_tags):
            # For each tag, get its prob given all other tags:
            # Prev column * P(tag|all tags) * P(word|tag)
            # Fill in viterbi matrix
            p[tagi1, word_i] = np.max([p[tagi2, word_i - 1] * transition_probs[tag2, tag1] * emission_probs[input_sentence[word_i], tag1] for tagi2,tag2 in enumerate(list_of_tags)])
            # Fill in backpointer
            back[tagi1, word_i] = np.argmax([p[tagi2, word_i - 1] * transition_probs[tag2, tag1] * emission_probs[input_sentence[word_i], tag1] for tagi2,tag2 in enumerate(list_of_tags)])

    # Termination steps:
    best_path_prob = np.max([p[tag_i, len(input_sentence)-1] for tag_i, tag in enumerate(list_of_tags)])
    best_path_pointer = np.argmax([p[tag_i, len(input_sentence)-1] for tag_i, tag in enumerate(list_of_tags)])
    return p, back, best_path_pointer, best_path_prob

def backtrace(back, best_path_pointer, input_sentence, list_of_tags):
    path_idx = [best_path_pointer]
    for column_i, column in enumerate(back.T[::-1]): # Starts at end
        max_tag_idx = max(column)
        path_idx.append(max_tag_idx)

    tag_seq = []
    for i in range(0,len(path_idx)-1):
        tag_seq.append(list_of_tags[path_idx[i]])

    tag_seq = tag_seq[::-1]
    return tag_seq


################################################
      ### TRAIN / TEST SPLIT & PROCESS ###

def split_train_test(train_data, train_test_split):
    # SHUFFLE TRAINING DATA TO CREATE TRAIN & TEST SET: 80% train, 20% test
    data_array = np.array([ele.strip().split("\t") for ele in train_data if ele.strip() != ""])
    token_sentence_idxs = data_array[:,0]
    tokens = data_array[:,1]
    tags = data_array[:,2]
    sentences = []
    sentence = []

    for token, tag in zip(tokens, tags):
        if token == '.':
            sentence.append(np.array([token, tag]))
            sentences.append(np.array(sentence))
            sentence = []
        else:
            sentence.append(np.array([token, tag]))

    sentences = np.array(sentences)
    np.random.shuffle(sentences)

    num_total_sentences = sentences.shape[0]
    num_train_sentences = int(num_total_sentences * train_test_split)
    train_sentences = sentences[:num_train_sentences]
    test_sentences = sentences[num_train_sentences:]
    return train_sentences, test_sentences


def process_testset(test_set, train_words, word_idx, tag_idx):
    test_data = np.array([ele.strip().split("\t") for ele in test_set if ele.strip() != ""])
    # nums = test_data[:,0]
    tokens = test_data[:,word_idx]
    tags = test_data[:,tag_idx]

    original_sentences = []
    original_sentence = []

    sentences_tags = []
    sentence_tags = []

    sentences_with_unk = []
    sentence_with_unk = []

    for token, tag in zip(tokens, tags):

        if token == '.':
            original_sentence.append(token)
            original_sentences.append(original_sentence)
            original_sentence = []
        else:
            original_sentence.append(token)

        # Replace unseen with 'UNK'
        if token not in train_words:
            token = "UNK"

        if token == '.':
            sentence_with_unk.append(token)
            sentences_with_unk.append(sentence_with_unk)
            sentence_with_unk = []
        else:
            sentence_with_unk.append(token)

        if tag == '.':
            sentence_tags.append(tag)
            sentences_tags.append(sentence_tags)
            sentence_tags = []
        else:
            sentence_tags.append(tag)

    return original_sentences, sentences_with_unk, sentences_tags

def write_new_txt(sentences, fname):
    with open("{}.txt".format(fname), "w") as outfile:
        for sentence in sentences:
            for i, (token, tag) in enumerate(sentence, 1):
                out_write = "{}\t{}\t{}\n".format(i, token, tag)
                outfile.write(out_write)
            out_write = "\n"
            outfile.write(out_write)

def create_word_tag_list(sentences_flatten, output_tags):
    sentences = []
    sentence = []
    for token, tag in zip(sentences_flatten, output_tags):
        if token == '.':
            sentence.append(np.array([token, tag]))
            sentences.append(np.array(sentence))
            sentence = []
        else:
            sentence.append(np.array([token, tag]))

    return sentences
################################################
      ### CALCULATE ACCURACY ###

def calculate_accuracy(system_output, ground_truth):
    count_correct = 0
    nums = 0
    for tag_pred, tag_target in zip(system_output, ground_truth):
        if tag_pred == tag_target:
            count_correct += 1
        nums += 1
    return count_correct/nums


################################################
def main():

    with open('berp-POS-training.txt', 'r') as train_file:
        train_data = train_file.readlines()

    #### Split data into train and test sets
    train_test_split = 0.8
    train_sentences, test_sentences = split_train_test(train_data, train_test_split)
    write_new_txt(train_sentences, "training_set_shuffled")
    write_new_txt(test_sentences, "test_set_shuffled")

    #### BUILD WORD AND TAG SETS FOR THE DATA FILES
    # Entire data
    with open('berp-POS-training.txt', 'r') as entire_file:
        entire_set = entire_file.readlines()

    # Train set
    with open('training_set_shuffled.txt', 'r') as train_file:
        train_set = train_file.readlines()

    # Test set
    with open('test_set_shuffled.txt', 'r') as test_file:
        test_set = test_file.readlines()

    # Build word & tag sets
    entire_tags, entire_words, entire_total_word_list = build_words_tags_sets(entire_set, 1, 2)
    train_tags, train_words, train_total = build_words_tags_sets(train_set, 1, 2)
    test_tags, test_words, test_total = build_words_tags_sets(test_set, 1, 2)

    #### DEAL WITH UNSEEN WORDS IN TRAIN SET
    # Across the given lexicon, count words unseen in train set
    unk_count = 0
    unk_words = []
    for word in entire_words:
        if word != '/n':
            if word not in train_words:
                unk_words.append(word)
                unk_count += entire_total_word_list.count(word)

    #### PROCESS TEST SET: REPLACE UNSEEN WORDS WITH 'UNK' & GET LIST OF GROUND-TRUTH TAGS
    original_sentences, sentences_with_unk, sentences_tags = process_testset(test_set, train_words, 1, 2)

    #### RUN VITERBI ON INPUT TOKENS FROM TEST SET
    SMOOTHING = True
    # Use training data and the UNK counts to build transition and emission matrices
    list_of_tags, list_of_words, word_count, transition_probs, emission_probs = build_lists_and_probabilities(train_set, train_tags, train_words, unk_count, smoothing=SMOOTHING)

    # Run viterbi function on test set, get list of tags
    sequence_tags = []
    for seq_i, seq in enumerate(sentences_with_unk):
        sys.stdout.write("\r Processing sentence [{}/{}] for test_set_shuffled.txt".format(seq_i+1, len(sentences_with_unk)))
        sys.stdout.flush()
        p, back, best_path_pointer, best_path_prob = viterbi_pos_tagger(list_of_tags, seq, transition_probs, emission_probs)
        output_tags = backtrace(back, best_path_pointer, seq, list_of_tags)
        sequence_tags.append(output_tags)

    #### RUN BASELINE ON INPUT TOKENS FROM TEST SET
    word_pos_count = get_word_pos_count(train_set)
    baseline_output = []
    for line in test_set:
        if line != "\n":
            line = line.split()
            word = line[1]
            tag = get_most_frequent_tag(word_pos_count, word)
            baseline_output.append(tag)

    #### CALCULATE ACCURACIES
    # Flatten lists
    flatten_list = lambda target_list : [tag for tag_list in target_list for tag in tag_list]
    ground_truth = flatten_list(sentences_tags)
    viterbi_output = flatten_list(sequence_tags)
    original_sentences_flatten = flatten_list(original_sentences)
    sentences_with_unk_flatten = flatten_list(sentences_with_unk)

    # Calculate accuracy on unks tagging
    total_unk = 0
    accurate_tag = 0
    for i, word in enumerate(sentences_with_unk_flatten):
        if word == "UNK":
            total_unk += 1
            # print(original_sentences_flatten[i])
            # print('Ground truth tag: {}'.format(ground_truth[i]))
            # print('Viterbi tag: {} \n'.format(viterbi_output[i]))
            if ground_truth[i] == viterbi_output[i]:
                accurate_tag +=1
    unk_accuracy = accurate_tag / total_unk

    # Get accuracy
    baseline_accuracy = calculate_accuracy(baseline_output, ground_truth)
    print('\n Baseline accuracy on test set: {}'.format(baseline_accuracy))
    viterbi_accuracy = calculate_accuracy(viterbi_output, ground_truth)
    print('\n Viterbi accuracy on test set: {}'.format(viterbi_accuracy))
    print('\n Viterbi accuracy on unseens in test set: {} \n'.format(unk_accuracy))

    # Write results to file
    sentences = create_word_tag_list(original_sentences_flatten, viterbi_output)
    write_new_txt(sentences, "viterbi_output")


    ##################### RUN TEST FILE assgn2-test-set.txt ####################
    #### Run Viterbi on test file & write output in same format
    # Read test set
    with open('assgn2-test-set.txt', 'r') as assgn2_test_file:
        assgn2_test_set = assgn2_test_file.readlines()

    assgn2_test_tags, assgn2_test_words, assgn2_test_total = build_words_tags_sets(assgn2_test_set, 1, 0)
    assgn2_original_sentences, assgn2_sentences_with_unk, assgn2_sentences_tags = process_testset(assgn2_test_set, train_words, 1, 0)

    # Run viterbi function on test set, get list of tags
    assgn2_sequence_tags = []
    for seq_i, seq in enumerate(assgn2_sentences_with_unk):
        sys.stdout.write("\r Processing sentence [{}/{}] for assgn2-test-set.txt".format(seq_i+1, len(assgn2_sentences_with_unk)))
        sys.stdout.flush()
        p, back, best_path_pointer, best_path_prob = viterbi_pos_tagger(list_of_tags, seq, transition_probs, emission_probs)
        output_tags = backtrace(back, best_path_pointer, seq, list_of_tags)
        assgn2_sequence_tags.append(output_tags)

    # Write results to file
    assgn2_viterbi_output = flatten_list(assgn2_sequence_tags)
    assgn2_original_sentences_flatten = flatten_list(assgn2_original_sentences)
    assgn2_sentences_with_unk_flatten = flatten_list(assgn2_sentences_with_unk)
    assgn2_sentences = create_word_tag_list(assgn2_original_sentences_flatten, assgn2_viterbi_output)
    write_new_txt(assgn2_sentences, "nguyen-dieumy-assgn2-test-output")

main()

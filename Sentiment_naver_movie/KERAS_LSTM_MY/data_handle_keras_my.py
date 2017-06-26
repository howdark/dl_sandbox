import numpy as np
import itertools
from collections import Counter


# tab separated document reading
def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data


# sentiment label to one hot encoding : positive [0, 1] / negative [1, 0]
# Required : pos_tagger = Twitter()
def tokenize(pos_tagger, doc):
    # norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


# sentiment label to one hot encoding : positive [0, 1] / negative [1, 0]
def labeller(labels):
    labels = np.array([[0, 1] if abs(label-1)<0.001 else [1, 0] for label in labels])
    return labels

# Sentence padding
def pad_sentences(sentences, padding_word="#PAD"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # sequence_length = max(len(x) for x in sentences)
    sequence_length = 95
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = [padding_word] * num_padding + sentence
        padded_sentences.append(new_sentence)
    return padded_sentences

# Build vocabulary from sentences
# After build vocabulary, find index of set undefined word as #UNDEFINED
def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv.insert(1, '#UNDEFINED')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # vocabulary['#UNDEFINED'] = len(vocabulary)
    return [vocabulary, vocabulary_inv]

# Build input data (Convert word to dictionary index)
def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

# Build test data (Convert word to dictionary index. If new word comes out, then set to #UNDEFINED)
def build_test_data(sentences, vocabulary, undefined_idx):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] if vocabulary.get(word) is not None else undefined_idx for word in sentence] for sentence in sentences])
    return x


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
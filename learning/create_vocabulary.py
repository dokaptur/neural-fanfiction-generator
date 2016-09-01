import os
import json


def create_vocab(data_dir, vocab_size=1000):
    stories_dir = os.path.join(data_dir, 'stories')
    vocab_filename = os.path.join(data_dir, 'vocabulary')
    vocab_upper_filename = os.path.join(data_dir, 'vocabulary_uppercase')
    vocabulary = {}

    for filename in os.listdir(stories_dir):
        data_file = os.path.join(stories_dir, filename)
        with open(data_file, 'r') as f:
            for line in f:
                for word in line.split():
                    word_original = word
                    word = word.lower()
                    if word not in vocabulary:
                        vocabulary[word] = [0, 0]
                    vocabulary[word][0] += 1
                    if word_original[0].isupper():
                        vocabulary[word][1] += 1

    sorted_vocabulary = sorted(vocabulary.items(), key=lambda x: x[1][0], reverse=True)
    most_common = sorted_vocabulary[:vocab_size]
    vocab_uppercase_map = {s[0]: (s[1][1] / s[1][0] > 0.5) for s in most_common}

    with open(vocab_filename, 'w') as f:
        for (word, count) in most_common:
            f.write(word)
            f.write('\n')

    with open(vocab_upper_filename, 'w') as f:
        json.dump(vocab_uppercase_map, f)


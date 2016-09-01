from learning.indexer import Indexer
from collections import deque
from itertools import takewhile
import numpy as np
import os
import json


PADDING = ''
EOD_TOK = '<EOD>'
EOS_TOK = '<EOS>'
UNKNOWN_TOK = '<UNK>'
GO_TOK = '<GO>'


def create_context(encoded_story, vocab, input_dim):
    i = len(encoded_story)
    context = [vocab.string_to_int(PADDING) for _ in range(0, input_dim)]
    free_space = len(context)
    for j in range(i - 1, -1, -1):
        l = len(encoded_story[j])
        if l <= free_space:
            context[free_space - l: free_space] = encoded_story[j]
            free_space -= l
        else:
            context[: free_space] = encoded_story[j][l - free_space: l]
            break
    return context


def create_entries(encoded_story, vocab, use_in_bow, input_dim, output_dim, use_go_tok=False,
                   use_last_sentence_only=False, generate_bow=False):
    bows = []
    if generate_bow:
        bows = create_bows(encoded_story, len(vocab), use_in_bow)
    entries = []
    padding_int = vocab.string_to_int(PADDING)
    go_int = vocab.string_to_int(GO_TOK)
    for i in range(1, len(encoded_story)):
        if use_last_sentence_only:
            context = create_context(encoded_story[i-1:i], vocab, input_dim)
        else:
            context = create_context(encoded_story[:i], vocab, input_dim)
        output = [padding_int for _ in range(output_dim)]
        l = min(output_dim, len(encoded_story[i]))
        output[:l] = encoded_story[i][:l]
        if use_go_tok:
            output.insert(0, go_int)
            output = output[:output_dim]
        if generate_bow:
            entries.append((np.array(context), np.array(output), bows[i-1]))
        else:
            entries.append((np.array(context), np.array(output), None))
    return entries


def bag_of_words(bow, words_to_add, use_in_bow):
    current = bow.copy()
    for w in words_to_add:
        if w in use_in_bow:
            current[w] += 1
    return current


def decode_bow(bow, vocab):
    results = []
    for i in range(len(bow)):
        results.append((bow[i], vocab.int_to_string(i)))
    results = sorted(results, reverse=True)
    return results


def normalize_bow(bow):
    log_freqs = bow
    norm = np.linalg.norm(log_freqs, 2)
    if norm > 0:
        log_freqs /= norm
    return log_freqs


def create_bows(encoded_story, vocab_len, use_in_bow):
    all_bows = []
    prefix = np.zeros(vocab_len, dtype=np.float32)
    # all_bows.append(prefix)
    for i in range(len(encoded_story)-1):
        current = bag_of_words(prefix, encoded_story[i], use_in_bow)
        all_bows.append(current)
        prefix = current
    normalized_bows = []
    for bow in all_bows:
        normalized_bows.append(normalize_bow(bow))
    return normalized_bows


def encode_with_default_unknown(word, vocab):
    try:
        return vocab.string_to_int(word.lower())
    except KeyError:
        return vocab.string_to_int(UNKNOWN_TOK)


def encode_sentence(sentence, vocab):
    encodings = [encode_with_default_unknown(w, vocab) for w in sentence.split()]
    return encodings


def encode_story(story, vocab):
    res = []
    for sentence in story:
        encodings = encode_sentence(sentence, vocab)
        encodings.append(vocab.string_to_int(EOS_TOK))
        res.append(encodings)
    if len(res) > 0:
        res[-1].append(vocab.string_to_int(EOD_TOK))
    return res


def decode_data(data, vocab, stop_at_eos=False, skip_specials=False):
    words = [vocab.int_to_string(d) for d in data]
    if skip_specials:
        specials = [PADDING, EOS_TOK, EOD_TOK, UNKNOWN_TOK, GO_TOK]
        words = [w for w in words if w not in specials]
    if stop_at_eos:
        words = list(takewhile(lambda x: x != EOS_TOK, words))
    return ' '.join(words).strip()


def load_story(story_file):
    with open(story_file) as f:
        story = []
        for line in f.readlines():
            story.append(line.strip())
    return story


class FixedDataGenerator(object):
    """
    Class that generates data from already crated stories.
    It provides always the same data, so it's good for debugging.
    """

    def __init__(self, n_data, input_dim, output_dim, batch_size, data_dir, vocab_file,
                 wrap_in_batches=True, load_all_stories=False, generate_bow=False,
                 use_last_sentence_only=False):
        self.i = 0
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        # Create vocabulary
        self.stories = []
        self.load_all_stories = load_all_stories
        self.current_story = -1
        self.story_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.n_stories = len(self.story_files)
        if load_all_stories:
            for story_file in self.story_files:
                story = load_story(story_file)
                self.stories.append(story)
        self.vocab = Indexer()
        specials = [PADDING, EOS_TOK, EOD_TOK, UNKNOWN_TOK, GO_TOK]
        for token in specials:
            self.vocab.string_to_int(token)
        with open(vocab_file, 'r') as f:
            for w in f:
                w = w.lower().strip()
                self.vocab.string_to_int(w)
        self.vocab.freeze()
        # cache for generated data
        self.cache = deque()
        self.wrap_in_batches = wrap_in_batches
        self.generate_bow = generate_bow
        self.use_last_sentence_only = use_last_sentence_only
        if not wrap_in_batches:
            self.batch_size = 1
        self.idf = None

        vocab_upper_filename = vocab_file + '_uppercase'
        if os.path.exists(vocab_upper_filename):
            with open(vocab_upper_filename) as f:
                self.vocab_uppercase_map = json.load(f)
        else:
            self.vocab_uppercase_map = {}

        self.use_in_bow = set()
        for s, i in self.vocab.items():
            if self.vocab_uppercase_map.get(s):
                self.use_in_bow.add(i)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def restart(self, shuffle=True):
        self.i = 0
        self.current_story = -1
        self.cache.clear()
        if shuffle:
            np.random.shuffle(self.stories)
            np.random.shuffle(self.story_files)

    def set_new_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.restart()

    def get_next_story(self, i):
        if self.load_all_stories:
            story = self.stories[i]
        else:
            story = load_story(self.story_files[i])
        return story

    def get_weights(self, data_batch):
        padding_int = self.vocab.string_to_int(PADDING)
        dims = data_batch.shape
        two_dim_assert = self.wrap_in_batches and len(dims) == 2 and dims[0] == self.batch_size
        one_dim_assert = (not self.wrap_in_batches) and len(dims) == 1
        assert(two_dim_assert or one_dim_assert)
        weights = np.ones(dims, dtype=np.float32)
        for i in range(dims[0]):
            if two_dim_assert:
                for j in range(dims[1]):
                    word_code = data_batch[i][j]
                    if word_code == padding_int:
                        weights[i][j] = 0
            else:
                word_code = data_batch[i]
                if word_code == padding_int:
                    weights[i] = 0
        return weights

    def next(self):
        if self.i < self.n_data:
            # not enough data in cache, we need to generate and prepare new story
            while len(self.cache) < self.batch_size:
                self.current_story += 1
                if self.current_story == self.n_stories:
                    raise StopIteration
                story = self.get_next_story(self.current_story)
                encoded_story = self.encode_story(story)
                entries = create_entries(encoded_story, self.vocab, self.use_in_bow,
                                         self.input_dim, self.output_dim,
                                         generate_bow=self.generate_bow,
                                         use_last_sentence_only=self.use_last_sentence_only)
                np.random.shuffle(entries)
                self.cache.extend(entries)

            # create input and output entry for training
            self.i += 1
            if self.wrap_in_batches:
                inputs, outputs, bows = [], [], []
                for i in range(self.batch_size):
                    entry = self.cache.popleft()
                    inputs.append(entry[0])
                    outputs.append(entry[1])
                    if self.generate_bow:
                        bows.append(entry[2])
                if self.generate_bow:
                    return np.array(inputs), np.array(outputs), np.array(bows)
                else:
                    return np.array(inputs), np.array(outputs), None
            else:
                entry = self.cache.popleft()
                return entry
        else:
            raise StopIteration()

    def encode_story(self, story):
        return encode_story(story, self.vocab)

    def decode_data(self, data, stop_at_eos=False, skip_specials=False):
        return decode_data(data, self.vocab, stop_at_eos=stop_at_eos,
                           skip_specials=skip_specials)



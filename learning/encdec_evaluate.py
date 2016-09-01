import bisect
import os
import sys
import numpy as np
import tensorflow as tf
from itertools import takewhile

from learning.detokenizer import detokenize
from learning.data_generators import FixedDataGenerator, encode_sentence, \
    create_context, load_story, bag_of_words, normalize_bow, \
    EOS_TOK, UNKNOWN_TOK, PADDING, EOD_TOK
from models.encdec_model import create_prediction_graph, create_training_graph


def generate_next_word(session, ops, choices, sample=True, exclude_unknown_token=True,
                       unknown_int=3, exclude_padding=False, padding_int=0, n_best=10):
    probs = session.run(ops[6])
    sums = np.sum(probs, axis=1)
    assert(np.min(sums) > 0.97 and np.max(sums) < 1.03)
    if exclude_unknown_token:
        probs = np.delete(probs, [unknown_int], axis=1)
    if exclude_padding:
        probs = np.delete(probs, [padding_int], axis=1)
    if sample:
        dim0, dim1 = probs.shape
        if n_best > 0:
            best_prob_indices = np.argsort(probs, axis=1)[:, -n_best:]
            probs = probs[np.reshape(np.arange(dim0), (dim0, 1)), best_prob_indices]
        else:
            best_prob_indices = np.repeat([np.arange(dim1)], dim0, axis=0)
        cumsums = np.cumsum(probs, axis=1)
        reindex_samples = [bisect.bisect(row, row[-1] * np.random.rand())
                           for row in cumsums]
        samples = best_prob_indices[np.arange(dim0), reindex_samples]
    else:
        samples = np.argmax(probs, axis=1)
    if exclude_unknown_token:
        samples = [i if i < unknown_int else i + 1 for i in samples]
    if exclude_padding:
        samples = [i + 1 for i in samples]
    w = session.run(ops[5], feed_dict={choices.name: samples})
    session.run(ops[2:4])
    return w


def generate_whole_story(session, conf, data_generator, placeholders, ops, sample=False,
                         first_sentences=None, sample_from=10):
    eos_tok_int = data_generator.vocab.string_to_int(EOS_TOK)
    eod_tok_int = data_generator.vocab.string_to_int(EOD_TOK)
    unknown_int = data_generator.vocab.string_to_int(UNKNOWN_TOK)
    input_seq, choices, bow_vectors = placeholders
    stories = []
    n_stories = data_generator.n_stories
    for i in range(conf.batch_size):
        if first_sentences is None:
            story_filename = data_generator.story_files[np.random.randint(0, n_stories)]
            first_sentence = load_story(story_filename)[0]
        else:
            first_sentence = first_sentences[i]
        encoded_first_sentence = encode_sentence(first_sentence, data_generator.vocab)
        stories.append([encoded_first_sentence])
    bow = [np.zeros(len(data_generator.vocab)) for _ in range(conf.batch_size)]
    normalized_bow = [np.zeros(len(data_generator.vocab)) for _ in range(conf.batch_size)]
    sl = 0
    eod = [False for _ in range(conf.batch_size)]
    while sl < 100 and (sl < 3 or (not np.all(eod))):
        sl += 1
        for i in range(conf.batch_size):
            bow[i] += bag_of_words(bow[i], stories[i][-1], data_generator.use_in_bow)
            normalized_bow[i] = normalize_bow(bow[i])
        x = [create_context(story, data_generator.vocab, conf.input_length) for story in stories]
        session.run(ops[0:2], {input_seq.name: x, bow_vectors.name: bow})
        preds = []
        for i in range(conf.output_length):
            if sample is None:
                sample = bool(np.random.randint(2))
            w = generate_next_word(session, ops, choices, sample, unknown_int=unknown_int,
                                   exclude_padding=(i < 3), n_best=sample_from)
            preds.append(w)
        y_pred = np.vstack(preds).transpose()
        for i in range(conf.batch_size):
            if eod[i] and sl > 3:
                next_sentence = []
            else:
                next_sentence = list(takewhile(lambda c: c != eos_tok_int, y_pred[i]))
                if eod_tok_int in y_pred[i]:
                    eod[i] = True
            if len(next_sentence) < len(y_pred[i]):
                next_sentence.append(eos_tok_int)
            stories[i].append(next_sentence)

    decoded_stories = []
    for i, story in enumerate(stories):
        story_text = ''
        for sentence in story:
            decoded_sentence = data_generator.decode_data(sentence, skip_specials=True)
            if decoded_sentence.strip() is not '':
                story_text += decoded_sentence
                story_text += '\n'
        decoded_stories.append(story_text)
    return decoded_stories


def get_perplexity(probs):
    log_probs = np.log(probs)
    avg_log_probs = np.mean(log_probs)
    perplexity = np.exp(-avg_log_probs)
    return perplexity


def evaluate_with_perplexity(session, conf, data_generator, placeholders, ops, choices,
                             use_random_data=False):
    input_seq, bow_vectors = placeholders
    padding_int = data_generator.vocab.string_to_int(PADDING)
    data_generator.restart()
    data = [entry for entry in data_generator]
    np.random.shuffle(data)
    perplexities = []
    for x, y, bow in data:
        if use_random_data:
            vocab_len = len(data_generator.vocab)
            x = np.random.randint(vocab_len, size=x.shape)
            y = np.random.randint(vocab_len, size=y.shape)
        session.run(ops[0:2], {input_seq.name: x, bow_vectors.name: bow})
        output_probs_all = []
        for i in range(conf.output_length):
            probs = session.run(ops[6])
            samples = y[:, i]
            output_probs = probs[range(conf.batch_size), samples]
            session.run(ops[5], feed_dict={choices.name: samples})
            session.run(ops[2:4])
            output_probs_all.append(output_probs)
        y_probs = np.vstack(output_probs_all).transpose()
        for i in range(conf.batch_size):
            no_padding_output_len = len(list(takewhile(lambda c: c != padding_int, y[i])))
            if no_padding_output_len > 0:
                perplexity = get_perplexity(y_probs[i, :no_padding_output_len])
                perplexities.append(perplexity)
    avg_perplexity = np.mean(perplexities)
    return avg_perplexity


def main_generate(conf, all_data_dir, model_save_dir, first_sentences=None, sample=True,
                  output_filename=None, sample_from=10):
    data_dir = os.path.join(all_data_dir, 'stories')
    vocab_file = os.path.join(all_data_dir, 'vocabulary')

    if output_filename is None:
        output_file = sys.stdout
    else:
        output_filename = '{}_{}_{}'.format(output_filename, sample, sample_from)
        output_file = open(os.path.join(model_save_dir, output_filename), 'w')

    conf.batch_size = 1
    data_generator = FixedDataGenerator(
        conf.n_batches, conf.input_length, conf.output_length, conf.batch_size,
        data_dir, vocab_file, generate_bow=True)
    # with learnt model
    vocabulary_size = len(data_generator.vocab)

    # inputs to the graph
    input_seq = tf.placeholder(np.int64, (conf.batch_size, conf.input_length), name='input_seq')
    output_seq = tf.placeholder(np.int64, (conf.batch_size, conf.output_length), name='output_seq')
    choices = tf.placeholder(np.int64, (conf.batch_size,), name='choices')
    bow_vector = tf.placeholder(np.float32, (conf.batch_size, vocabulary_size), name='bow_input')
    placeholders = (input_seq, output_seq, None, bow_vector)
    placeholders_decoder = (input_seq, output_seq, choices, bow_vector)

    create_training_graph(placeholders, vocabulary_size, conf.hidden_size, conf.emb_dim,
                          conf.n_layers, conf.learning_rate, conf.hidden_bow_dim,
                          use_bow=conf.use_bow)
    saver = tf.train.Saver(tf.all_variables())
    ops = create_prediction_graph(placeholders_decoder, vocabulary_size, conf.hidden_size,
                                  conf.emb_dim, conf.n_layers, conf.hidden_bow_dim,
                                  use_bow=conf.use_bow)

    decoded_stories = []
    with tf.Session() as session:
        if tf.train.latest_checkpoint(model_save_dir):
            saver.restore(session, tf.train.latest_checkpoint(model_save_dir))
            print('Model restored from {}'.format(
                tf.train.latest_checkpoint(model_save_dir)), file=output_file)
        else:
            print('ERROR: No model stored in {}'.format(model_save_dir), file=output_file)
            return []

        # prediction phase
        n_data = 1
        for i in range(n_data):
            if first_sentences and i < len(first_sentences):
                generator_init = first_sentences[i]
            else:
                generator_init = None
            decoded_stories.extend(generate_whole_story(
                session, conf, data_generator, (input_seq, choices, bow_vector), ops,
                sample=sample, first_sentences=generator_init, sample_from=sample_from))
        detokenized_stories = []
        for st in decoded_stories:
            detokenized_stories.append(detokenize(st, data_generator.vocab_uppercase_map))
    return detokenized_stories


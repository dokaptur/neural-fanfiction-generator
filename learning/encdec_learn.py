import numpy as np
import tensorflow as tf
import os
import re

from learning.data_generators import FixedDataGenerator
from learning.encdec_evaluate import evaluate_with_perplexity
from models.encdec_model import create_training_graph, create_prediction_graph


def train_model(conf, all_data_dir, model_save_dir, warm_start=False):
    data_dir = os.path.join(all_data_dir, 'stories')
    test_data_dir = os.path.join(all_data_dir, 'test/stories')
    vocab_file = os.path.join(all_data_dir, 'vocabulary')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(
        model_save_dir, 'model_{}.pt'.format(conf.to_string()))
    data_generator = FixedDataGenerator(
        conf.n_batches, conf.input_length, conf.output_length, conf.batch_size,
        data_dir, vocab_file, load_all_stories=False, generate_bow=True)
    test_data_generator = FixedDataGenerator(
        conf.n_batches, conf.input_length, conf.output_length, conf.batch_size,
        test_data_dir, vocab_file, load_all_stories=False, generate_bow=True)
    vocabulary_size = len(data_generator.vocab)

    # inputs to the graph
    input_seq = tf.placeholder(np.int64, (conf.batch_size, conf.input_length), name='input_seq')
    output_seq = tf.placeholder(np.int64, (conf.batch_size, conf.output_length), name='output_seq')
    choices = tf.placeholder(np.int64, (conf.batch_size,), name='choices')
    output_weights = tf.placeholder(np.float32, (conf.batch_size, conf.output_length),
                                    name='output_weights')
    bow_vector = tf.placeholder(np.float32, (conf.batch_size, vocabulary_size), name='bow_input')
    placeholders = (input_seq, output_seq, output_weights, bow_vector)
    placeholders_decoder = (input_seq, output_seq, choices, bow_vector)

    average_loss, learning_step, parameters, init_ops = \
        create_training_graph(placeholders, vocabulary_size, conf.hidden_size, conf.emb_dim,
                              conf.n_layers, conf.learning_rate, conf.hidden_bow_dim,
                              use_bow=conf.use_bow)

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=30)

    ops = create_prediction_graph(placeholders_decoder, vocabulary_size, conf.hidden_size,
                                  conf.emb_dim, conf.n_layers, conf.hidden_bow_dim,
                                  use_bow=conf.use_bow)

    # learning phase
    with tf.Session() as session:
        it = 0
        if not warm_start:
            files_to_remove = [os.path.join(model_save_dir, f)
                               for f in os.listdir(model_save_dir)]
            for f in files_to_remove:
                os.remove(f)
        with open(os.path.join(model_save_dir, 'log'), 'a') as log_file:
            if warm_start and tf.train.latest_checkpoint(model_save_dir):
                model_path = tf.train.latest_checkpoint(model_save_dir)
                print(model_path, file=log_file)
                iteration = re.search('-(\d+)$', model_path)
                if iteration is not None:
                    it = int(iteration.group(1))
                saver.restore(session, tf.train.latest_checkpoint(model_save_dir))
                print('Session restored!', file=log_file)
            else:
                session.run(init_ops)
                print('New session created!', file=log_file)
            log_file.flush()
            itj = 0
            loss_value = 0
            for _ in range(conf.n_epochs_training):
                it += 1
                data_generator.restart()
                for x, y, bow in data_generator:
                    w = data_generator.get_weights(y)
                    feed_dict = {input_seq.name: x, output_seq.name: y, output_weights.name: w,
                                 bow_vector.name: bow}
                    res = session.run([average_loss, learning_step], feed_dict=feed_dict)
                    loss_value += res[0]
                    itj += 1
                    if itj == 1:
                        print('Batch nr {}: {}'.format(itj, loss_value), file=log_file)
                    if itj % conf.freq_display == 0:
                        loss_value /= conf.freq_display
                        print('Batch nr {}: {}'.format(itj, loss_value), file=log_file)
                        log_file.flush()
                        loss_value = 0
                test_data_perplexity = \
                    evaluate_with_perplexity(session, conf, test_data_generator,
                                             (input_seq, bow_vector), ops, choices)
                print('\nIteration nr {}: test data perplexity: {}\n'.format(
                    it, test_data_perplexity), file=log_file)
                if it % conf.checkpoint_frequency == 0:
                    new_model_save_path = saver.save(session, model_save_path, global_step=it)
                    print('\nModel saved in file: {}\n'.format(new_model_save_path), file=log_file)

import numpy as np
import tensorflow as tf

from models.utils import batch_same_matmul, scan, softmax_loss
from tensorflow.python.ops.rnn_cell import MultiRNNCell, LSTMCell, InputProjectionWrapper, \
    EmbeddingWrapper


def stacked_rnn_step(input_vocabulary_size, hidden_size=13, emb_dim=11, n_layers=2,
                     variable_scope='encdec'):
    with tf.variable_scope(variable_scope, reuse=None):
        rnn_cell = MultiRNNCell([LSTMCell(hidden_size)] * n_layers)  # stacked LSTM
        proj_wrapper = InputProjectionWrapper(rnn_cell, emb_dim)
    embedding_wrapper = EmbeddingWrapper(proj_wrapper, input_vocabulary_size, emb_dim)
    return embedding_wrapper


def rnn_encoder(input_seq, rnn_step, initial_states, use_bow):
    batch_size, input_length = [v.value for v in input_seq.get_shape()]
    if not use_bow:
        initial_states = rnn_step.zero_state(batch_size, dtype=np.float32)
        full_state_dim = initial_states.get_shape()[-1].value
        initial_states.set_shape((batch_size, full_state_dim))

    encoded_states, encoded_outputs = scan(rnn_step, input_seq, initial_states)
    final_encoded_states = encoded_states[:, input_length - 1, :]
    final_encoded_outputs = encoded_outputs[:, input_length - 1, :]
    return final_encoded_states, final_encoded_outputs


def create_training_graph(placeholders, vocabulary_size, hidden_size=13, emb_dim=11,
                          n_layers=2, learning_rate=0.01, hidden_bow_dim=3,
                          use_bow=True):
    # inputs to the graph
    input_seq, output_seq, output_weights, bow_vectors = placeholders
    # batch_size, input_length = [v.value for v in input_seq.get_shape()]
    batch_size, output_length = [v.value for v in output_seq.get_shape()]

    with tf.variable_scope('modelXYZ', reuse=None):
        multiplier = 2 * n_layers
        bow_projection_variable_1 = tf.get_variable("bow_projection_variable_1",
                                                    (vocabulary_size, hidden_bow_dim), np.float32,
                                                    tf.random_normal_initializer())
        bow_projection_variable_2 = tf.get_variable("bow_projection_variable_2",
                                                    (hidden_bow_dim, hidden_size), np.float32,
                                                    tf.random_normal_initializer())
        bow_hidden_states = tf.matmul(bow_vectors, bow_projection_variable_1)
        initial_states = tf.matmul(bow_hidden_states, bow_projection_variable_2)
        initial_states = tf.tile(initial_states, (1, multiplier))
        rnn_step = stacked_rnn_step(vocabulary_size, hidden_size, emb_dim, n_layers,
                                    variable_scope='encoder')
        final_encoded_states, final_encoded_outputs = \
            rnn_encoder(input_seq, rnn_step, initial_states, use_bow)

        rnn_step_decoder = stacked_rnn_step(vocabulary_size, hidden_size, emb_dim, n_layers,
                                            variable_scope='decoder')
        # rnn_step_decoder = rnn_step
        decoder_states, decoder_outputs = scan(rnn_step_decoder, output_seq[:, 0:output_length - 1],
                                               final_encoded_states,
                                               include_first=final_encoded_outputs, reuse=True)

        output_projection_variable = tf.get_variable("output_projection_variable",
                                                     (hidden_size, emb_dim), np.float32,
                                                     tf.random_normal_initializer())

        with tf.variable_scope('scan', reuse=True):
            embeddings_variable = tf.get_variable("EmbeddingWrapper/embedding")

            proj_outputs = batch_same_matmul(decoder_outputs, output_projection_variable)
            output_word_scores = batch_same_matmul(proj_outputs,
                                                   embeddings_variable[:vocabulary_size, :],
                                                   transpose_b=True)

            losses_matrix = softmax_loss(output_word_scores, output_seq)
            if output_weights is not None:
                losses_matrix = losses_matrix * output_weights
            losses = tf.reduce_sum(losses_matrix, 1)
            average_loss = tf.reduce_mean(losses)
            parameters = tf.trainable_variables()

            # learning by gradient descent
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            learning_step = optimizer.minimize(average_loss)

            init_ops = tf.initialize_all_variables()

    return average_loss, learning_step, parameters, init_ops


def create_prediction_graph(placeholders, vocabulary_size, hidden_size=13, emb_dim=11,
                            n_layers=2, hidden_bow_dim=3, use_bow=True):

    # inputs to the graph
    input_seq, output_seq, choices, bow_vectors = placeholders
    # batch_size, input_length = [v.value for v in input_seq.get_shape()]
    batch_size, _ = [v.value for v in output_seq.get_shape()]

    with tf.variable_scope('modelXYZ', reuse=True):
        full_state_dim = 2 * hidden_size * n_layers
        multiplier = 2 * n_layers
        bow_projection_variable_1 = tf.get_variable("bow_projection_variable_1",
                                                    (vocabulary_size, hidden_bow_dim), np.float32,
                                                    tf.random_normal_initializer())
        bow_projection_variable_2 = tf.get_variable("bow_projection_variable_2",
                                                    (hidden_bow_dim, hidden_size), np.float32,
                                                    tf.random_normal_initializer())
        bow_hidden_states = tf.matmul(bow_vectors, bow_projection_variable_1)
        initial_states = tf.matmul(bow_hidden_states, bow_projection_variable_2)
        initial_states = tf.tile(initial_states, (1, multiplier))
        rnn_step = stacked_rnn_step(vocabulary_size, hidden_size, emb_dim, n_layers,
                                    variable_scope='encoder')
        final_encoded_states, final_encoded_outputs = \
            rnn_encoder(input_seq, rnn_step, initial_states, use_bow)
        # prediction
        decoder_state = tf.Variable(np.zeros((batch_size, full_state_dim)),
                                    dtype=np.float32, name='decoder_state')
        decoder_output = tf.Variable(np.zeros((batch_size, hidden_size)),
                                     dtype=np.float32, name='decoder_output')
        current_observation = tf.Variable(np.zeros((batch_size,)),
                                          dtype=np.int64, name='current_observation')

        output_projection_variable = tf.get_variable("output_projection_variable",
                                                     (hidden_size, emb_dim), np.float32,
                                                     tf.random_normal_initializer())
        proj_output = batch_same_matmul(decoder_output, output_projection_variable)
        embeddings_variable = tf.get_variable("scan/EmbeddingWrapper/embedding")
        decoder_prediction = batch_same_matmul(proj_output,
                                               embeddings_variable[:vocabulary_size, :],
                                               transpose_b=True)

        rnn_step_decoder = stacked_rnn_step(vocabulary_size, hidden_size, emb_dim, n_layers,
                                            variable_scope='decoder')
        # rnn_step_decoder = rnn_step
        with tf.variable_scope('scan', reuse=True):
            new_output, new_state = rnn_step_decoder(current_observation, decoder_state)

        decoder_init_op1 = tf.assign(decoder_state, final_encoded_states)
        decoder_init_op2 = tf.assign(decoder_output, final_encoded_outputs)

        decoder_step_op1 = tf.assign(decoder_state, new_state)
        decoder_step_op2 = tf.assign(decoder_output, new_output)
        decoder_step_op3 = tf.assign(current_observation, tf.arg_max(decoder_prediction, 1))
        decoder_step_op4 = tf.assign(current_observation, choices)
        decoder_step_op5 = tf.nn.softmax(decoder_prediction)

    return decoder_init_op1, decoder_init_op2, decoder_step_op1, decoder_step_op2, \
           decoder_step_op3, decoder_step_op4, decoder_step_op5

import numpy as np
import tensorflow as tf


def batch_same_matmul(batched_vectors, mat, transpose_b=False):
    dims = [a.value for a in batched_vectors.get_shape()]
    if transpose_b:
        final_dim = mat.get_shape()[0].value
    else:
        final_dim = mat.get_shape()[-1].value
    unfolded_length = np.prod(dims[:-1])
    return tf.reshape(tf.matmul(tf.reshape(batched_vectors, (unfolded_length, dims[-1])),
                                mat, transpose_b=transpose_b),
                      dims[:-1] + [final_dim])


def repeat_tensor(tensor, n_repeats, start_dim=0):
    if isinstance(n_repeats, int):
        n_repeats = (n_repeats,)
    dims = tuple(a.value for a in tensor.get_shape())
    tmp = tf.reshape(tensor, dims[:start_dim] + (1,) * len(n_repeats) + dims[start_dim:])
    return tf.tile(tmp, (1,) * start_dim + n_repeats + (1,) * (len(dims) - start_dim))


def softmax_loss(scores, indices):
    n_dims = len(scores.get_shape())
    return -select_entries(scores, indices) + reduce_log_sum_exp(scores, n_dims - 1)


def softmax(mat, reduction_indices=None, name=None):
    return tf.exp(log_softmax(mat, reduction_indices=reduction_indices), name=name)


def log_softmax(mat, reduction_indices=None, name=None):
    if reduction_indices is None:
        reduction_indices = [0] * len(mat.get_shape())
    n_rep = mat.get_shape()[reduction_indices].value
    lse = reduce_log_sum_exp(mat, reduction_indices=reduction_indices)
    probas = mat - repeat_tensor(lse, n_rep, reduction_indices)
    return tf.identity(probas, name=name)


def reduce_log_sum_exp(mat, reduction_indices=None, safe=True):
    dims = tuple(a.value for a in mat.get_shape())
    if not safe:
        return tf.log(tf.reduce_sum(tf.exp(mat), reduction_indices=reduction_indices))
    else:
        maxi = tf.reduce_max(mat, reduction_indices=reduction_indices)
        maxi_rep = repeat_tensor(maxi, dims[reduction_indices], reduction_indices)
        return tf.log(tf.reduce_sum(tf.exp(mat - maxi_rep), reduction_indices=reduction_indices)) + maxi


def select_entries(tensor, idx):
    """
    Select entries in a tensor
    This is similar to the gather operator, but it selects one value at a time

    Args:
        tensor: 3d tensor from which values are extracted
        idx: 2d array of indices corresponding to the last dimension of the tensor

    Returns:
        the operator output which selects the right entries: output[i,j] = tensor[i, j, idx[i,j]]
    """
    mat_dims = tuple(a.value for a in tensor.get_shape())
    k = mat_dims[-1]
    idx_dims = tuple(a.value for a in idx.get_shape())
    if mat_dims[:-1] != idx_dims:
        raise ValueError("Value tensor has size {0} does not begin as the index tensor which has size {1}".format(
                mat_dims, idx_dims
        ))
    mat_reshaped = tf.reshape(tensor, (np.prod(mat_dims),))
    shifts1 = np.dot(np.ones((idx_dims[0], 1)), np.reshape(np.arange(0, idx_dims[1]), (1, -1))) * k
    shifts2 = np.dot(np.reshape(np.arange(0, idx_dims[0]), (-1, 1)), np.ones((1, idx_dims[1]))) * k * mat_dims[-2]
    # print(mat_dims, np.prod(mat_dims))
    # print(shifts1, shifts2)
    # print(mat_reshaped.get_shape())
    idx_reshaped = idx + tf.constant(shifts1 + shifts2, np.int64)
    return tf.gather(mat_reshaped, idx_reshaped)


def scan(transition, inputs, initial_states, scope_name="scan", reuse=None, include_first=None):
    """
    Unfold a RNN by recursively applying a recurrent unit

    Args:
        transition: the transition function of the RNN. An instance of rnn_cell tensorflow class.
            When called with two arguments (input, state), it returns a graph and return a
            pair (output, state)
        inputs: the inputs of the transition. Of size (batch_size, length) or (batch_size, length, input_size) for
            multi-dimensional inputs
        initial_states: initial states of the rnn. Must be of size (batch_size, rnn_state_size)
        scope_name: name of the scope for the variables inside the RNN
        reuse: should the variable be reused from a previously created graph?
        include_first: if given, must be the initial output of size (batch_size, rnn_output_size) and append the initial
            state and initial_output at the beginning of the resulting tensors

    Returns:
        A pair (states, outputs):
        - states are the concatenated internal states of the units and has size
            (batch_size, length, rnn_state_size)
        - ouputs are the concatenated outputs of the units and has size
            (batch_size, length, rnn_output_size)

    """
    if len(inputs.get_shape()) == 2:  # one dimensional input
        batch_size, input_length = [a.value for a in inputs.get_shape()]
        input_size = 1
    else:
        batch_size, input_length, input_size = [a.value for a in inputs.get_shape()]

    batch_size, state_size = [a.value for a in initial_states.get_shape()]
    cur_states = initial_states
    if include_first is not None:
        output_size = include_first.get_shape()[-1].value
        states = [tf.reshape(initial_states, (batch_size, 1, state_size))]
        outputs = [tf.reshape(include_first, (batch_size, 1, output_size))]
    else:
        states = []
        outputs = []
    output_size = None
    for i in range(input_length):
        with tf.variable_scope(scope_name, reuse=(i > 0) or reuse):
            if input_size == 1:
                cur_outputs, cur_states = transition(inputs[:, i], cur_states)
            else:
                cur_outputs, cur_states = transition(inputs[:, i, :], cur_states)
            if output_size is None:
                output_size = cur_outputs.get_shape()[-1].value
            states.append(tf.reshape(cur_states, (batch_size, 1, state_size)))
            outputs.append(tf.reshape(cur_outputs, (batch_size, 1, output_size)))
    all_states = tf.concat(1, states)
    all_outputs = tf.concat(1, outputs)
    return all_states, all_outputs


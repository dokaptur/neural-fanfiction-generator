from learning.configs import BasicConfig
from learning.encdec_learn import train_model

import argparse
import os
import sys


def main():
    conf = BasicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing data.'
                                         ' It\'s structure should be the same as the one of the '
                                         'output directory from prepare_data.py.')
    parser.add_argument('saved_model_dir', help='Output directory for the trained model.')
    parser.add_argument('--warm_start', help='Continue training if model already exists.',
                        action='store_true')

    parser.add_argument('--use_bow',
                        help='Use the whole context\'s bag-of-words as an initial encoder state.',
                        action='store_true')
    parser.add_argument('--batch_size', help='Number of batches for stochastic optimization step.',
                        type=int, default=conf.batch_size)
    parser.add_argument('--emb_dim', help='Dimension of words embeddings.',
                        type=int, default=conf.emb_dim)
    parser.add_argument('--hidden_size', help='Hidden dimension of LSTM cell.',
                        type=int, default=conf.hidden_size)
    parser.add_argument('--hidden_bow_dim', help='Hidden dimension of bag-of-words state.',
                        type=int, default=conf.hidden_bow_dim)
    parser.add_argument('--input_length', help='Number of words from context used by encoder.',
                        type=int, default=conf.input_length)
    parser.add_argument('--output_length', help='Number of words used by decoder.',
                        type=int, default=conf.output_length)
    parser.add_argument('--n_epochs_training', help='Number of epochs in training.',
                        type=int, default=conf.n_epochs_training)
    parser.add_argument('--freq_display',
                        help='Frequency (nr of batches) of error and perplexity logging.',
                        type=int, default=conf.freq_display)
    parser.add_argument('--n_layers', help='Number of layers.',
                        type=int, default=conf.n_layers)
    parser.add_argument('--checkpoint_frequency', help='Frequency (nr of epochs) of saving model.',
                        type=int, default=conf.checkpoint_frequency)
    parser.add_argument('--learning_rate', help='Learning rate.',
                        type=int, default=conf.learning_rate)
    args = parser.parse_args()

    conf.use_bow = args.use_bow
    conf.batch_size = args.batch_size
    conf.emb_dim = args.emb_dim
    conf.hidden_size = args.hidden_size
    conf.hidden_bow_dim = args.hidden_bow_dim
    conf.input_length = args.input_length
    conf.output_length = args.output_length
    conf.n_epochs_training = args.n_epochs_training
    conf.freq_display = args.freq_display
    conf.n_layers = args.n_layers
    conf.checkpoint_frequency = args.checkpoint_frequency
    conf.learning_rate = args.learning_rate

    if not os.path.exists(args.data_dir):
        print('Data directory {} does not exist!'.format(args.data_dir))
        sys.exit(1)

    if not os.path.exists(args.saved_model_dir):
        os.makedirs(args.saved_model_dir)
        print('Directory {} created!'.format(args.saved_model_dir))

    train_model(conf, args.data_dir, args.saved_model_dir, warm_start=args.warm_start)


if __name__ == '__main__':
    main()

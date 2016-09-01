from learning.configs import BasicConfig
from learning.encdec_evaluate import main_generate
from simulation.generate_story import tokenize

import argparse
import os
import sys


def main():
    conf = BasicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing data.'
                                         ' It\'s structure should be the same as the one of the '
                                         'output directory from prepare_data.py.')
    parser.add_argument('saved_model_dir', help='Directory with the trained model.')
    parser.add_argument('--sample_from',
                        help='Number of words with the highest likelihood we want to sample from.',
                        type=int, default=10)
    parser.add_argument('--sample',
                        help='Set to 1 for sampling and 0 for argmax in generation phase.',
                        type=int, default=1)
    parser.add_argument('--first_sentence',
                        help='First sentence of generated story.')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Data directory {} does not exist!'.format(args.data_dir))
        sys.exit(1)
    if not os.path.exists(args.saved_model_dir):
        print('Saved model directory {} does not exist!'.format(args.data_dir))
        sys.exit(1)

    model_files = [f for f in os.listdir(args.saved_model_dir)]
    for f in model_files:
        if f.startswith('model'):
            conf.parse_config_from_model_name(f)
            break
    print(conf.to_string())

    first_sentence = None
    if args.first_sentence is not None:
        first_sentence = [[' '.join([w.lower() for w in tokenize(args.first_sentence).split()])]]

    stories = main_generate(conf, args.data_dir, args.saved_model_dir,
                            first_sentences=first_sentence, sample=bool(args.sample),
                            sample_from=args.sample_from)
    print('\n\n\nStory:\n')
    print(stories[0])


if __name__ == '__main__':
    main()
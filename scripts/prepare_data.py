from simulation.generate_story import tokenize
from learning.create_vocabulary import create_vocab

import argparse
import os
import random
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stories_dir', help='Directory containing stories.')
    parser.add_argument('output_dir', help='Output directory for generated simulations.')
    parser.add_argument('vocabulary_size', help='Size of vocabulary to create.',
                        type=int)
    parser.add_argument('validation_set_size', help='Number of stories for validation set.',
                        type=int)
    args = parser.parse_args()

    if not os.path.exists(args.stories_dir):
        print('Directory {} does not exist!'.format(args.stories_dir))
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Output directory {} created!'.format(args.output_dir))

    output_stories_dir = os.path.join(args.output_dir, 'stories')
    if not (os.path.exists(output_stories_dir) and os.path.isdir(output_stories_dir)):
        os.makedirs(output_stories_dir)

    for s_file in os.listdir(args.stories_dir):
        with open(os.path.join(args.stories_dir, s_file)) as fr:
            story = fr.read()
            tokenized_story = tokenize(story)
            with open(os.path.join(output_stories_dir, s_file), 'w') as fw:
                fw.write(tokenized_story)

    create_vocab(args.output_dir, args.vocabulary_size)

    validation_stories_dir = os.path.join(args.output_dir, 'test/stories')
    if not (os.path.exists(validation_stories_dir) and os.path.isdir(validation_stories_dir)):
        os.makedirs(validation_stories_dir)

    stories = [os.path.join(output_stories_dir, f) for f in os.listdir(output_stories_dir)]
    random.shuffle(stories)

    for i in range(min(args.validation_set_size, len(stories))):
        shutil.move(stories[i], validation_stories_dir)

if __name__ == '__main__':
    main()

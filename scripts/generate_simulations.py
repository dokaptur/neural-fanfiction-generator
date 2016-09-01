from simulation.generate_story import generate_stories
import argparse
import os
import random as rd


def main():
    rd.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Output directory for generated simulations.')
    parser.add_argument('number_of_stories', help='Number of stories to generate.',
                        type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print('Directory {} has been created.')
        os.makedirs(args.output_dir)

    stories = generate_stories(args.number_of_stories, do_tokenize=False)
    for i, s in enumerate(stories):
        filename = os.path.join(args.output_dir, 'Story_{}'.format(i))
        with open(filename, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    main()

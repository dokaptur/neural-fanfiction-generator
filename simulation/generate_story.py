import re
import random as rd
import simulation.generate_world as gw
from simulation.mood import Mood
from nltk import sent_tokenize, word_tokenize


class StoryGenerator(object):
    def __init__(self, world):
        self.world = world

    def get_random_value(self, key):
        return self.world[key][rd.randint(0, len(self.world[key]) - 1)]

    def generate_drama_story(self):
        story = []
        actor1 = self.get_random_value('actor')
        actor2 = self.get_random_value('actor')
        while actor2 == actor1:
            actor2 = self.get_random_value('actor')

        # actor1 is leading
        story.append((actor1.pc_action_in(self.get_random_value('past_cont_action'),
                                          self.get_random_value('location'))))
        actor1.is_leading = True
        actor1.change_mood(self.get_random_value('mood'), '')
        story.append(actor1.mood_elaborate())
        loc2 = self.get_random_value('location')
        while loc2 == actor1.location:
            loc2 = self.get_random_value('location')
        story.append(actor1.go_to(loc2, self.get_random_value('transition')))
        actor1.is_leading = False

        # actor2 appears
        story.append('{} {}'.format(self.get_random_value('conj_adverb'),
                                    actor2.pc_action_in(
                                        self.get_random_value('past_cont_action'), loc2)))
        story.append(actor2.ask_question(actor1, self.get_random_value('question')))
        story.append(actor1.mood_statement())
        story.append(actor2.change_mood(self.get_random_value('mood')))
        actor2.is_leading = True
        story.append(actor2.make_dialog(actor1))
        actor2.is_leading = False

        # interaction
        story.append(actor1.interact(self.get_random_value('interaction'), actor2))
        story.append(actor1.mood_action())
        story.append(self.get_random_value('ending'))
        return [u for s in story for u in s.split('\n')]  # remove '\n'

    def generate_romance_story(self):
        story = []
        actor1 = self.get_random_value('actor')
        actor2 = self.get_random_value('actor')
        while (actor2 == actor1):
            actor2 = self.get_random_value('actor')

        # actor1 is in love
        story.append(actor1.pc_action_in(self.get_random_value('past_cont_action'),
                                         self.get_random_value('location')))
        actor1.is_leading = True
        actor1.change_mood(Mood.create_in_love(actor2.firstname))
        story.append(actor1.mood_elaborate())
        loc2 = self.get_random_value('location')
        while loc2 == actor1.location:
            loc2 = self.get_random_value('location')
        story.append(actor1.go_to(loc2, self.get_random_value('transition')))
        actor1.is_leading = False

        # actor2 appears, feels lonely
        story.append('{} {}'.format(self.get_random_value('conj_adverb'),
                                    actor2.pc_action_in(
                                        self.get_random_value('past_cont_action'), loc2)))
        story.append(actor2.change_mood(Mood.create_lonely()))
        actor2.is_leading = True
        story.append(actor2.mood_statement())

        # actor2 sees actor1
        actor2.is_leading = False
        story.append('{} {} {} {}.'.format(self.get_random_value('adverb'),
                                           actor2.get_name(),
                                           self.get_random_value('recognition'),
                                           actor1.get_name()))
        actor2.is_leading = True
        story.append(actor2.ask_question(actor1, self.get_random_value('question')))
        actor2.is_leading = False
        story.append(actor1.make_dialog(actor2))

        # interaction
        story.append(actor1.interact(self.get_random_value('interaction'), actor2))
        story.append(actor2.change_mood(Mood.create_happy()))
        story.append(actor2.mood_action())
        story.append(actor2.mood_statement())
        story.append(self.get_random_value('ending'))
        # return [u for s in story for u in s.split('\n')]  # remove '\n'
        return story


def tokenize(story):
    sentences = sent_tokenize(story)
    output_lines = []
    for s in sentences:
        pattern = r"'[A-Z]"
        nltk_tokens = word_tokenize(s)
        tokens = []
        for t in nltk_tokens:
            if re.match(pattern, t):
                # print(t)
                tokens.append(t[0])
                tokens.append(t[1:])
            else:
                tokens.append(t)
        output_lines.append(str.join(' ', tokens))
    return str.join('\n', output_lines)


def generate_stories(n, do_tokenize=True):
    stories = []
    for _ in range(n):
        if rd.randint(0, 1) == 1:
            # generate drama
            world = gw.generate_world_drama()
            generator = StoryGenerator(world)
            story_lines = generator.generate_drama_story()
        else:
            world = gw.generate_world_romance()
            generator = StoryGenerator(world)
            story_lines = generator.generate_romance_story()
        story = str.join('\n', story_lines)
        if do_tokenize:
            stories.append(tokenize(story))
        else:
            stories.append(story)
    return stories

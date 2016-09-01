from simulation.actor import Actor
from simulation.mood import Mood


def generate_common():
    model = {}

    model['actor'] = [Actor('Severus', 'Snape'), Actor('Albus', 'Dumbledore'),
                      Actor('Harry', 'Potter'), Actor('Hermione', 'Granger', gender='female'),
                      Actor('Ron', 'Weasley'), Actor('Lord Voldemort'), Actor('Dobby'),
                      Actor('Minerva', 'McGonagall', gender='female'),
                      Actor('Draco', 'Malfoy'), Actor('Sirius', 'Black'), Actor('Remus', 'Lupin')]

    model['location'] = ['Greenhouse', 'Astronomy Tower', 'Potions Classroom',
                         'Forbidden Forest', 'Shrieking Shack', 'Hogsmeade', 'Diagon Alley', 'Grimmuald Place',
                         'Burrow', 'Malfoy Manor']

    model['conj_adverb'] = ['Meanwhile', 'At the same time', '']

    model['transition'] = ['went to', 'run to', 'apparated to',
                           'took a broom and flew to']

    model['past_cont_action'] = ['reading', 'sleeping', 'watching stars',
                                 'learning', 'brewing a potion', 'drinking Butter Beer', 'crying',
                                 'inventing new spell']

    model['adverb'] = ['Suddenly', 'Then', '']

    model['question'] = ['What\'s wrong with you?', 'How are you?',
                         'How do you feel today?']

    return model


def generate_world_drama():
    model = generate_common()

    model['interaction'] = ['hit', 'killed', 'tortured', 'left', 'hexed']

    model['mood'] = [Mood.create_sad('nobody loves me'),
                     Mood.create_sad(),
                     Mood.create_sad('we are going to loose the war'),
                     Mood.create_sad('everything is pointless'),
                     Mood.create_sad('all people will die'),
                     Mood.create_angry('people are supid'),
                     Mood.create_angry(),
                     Mood.create_angry('nobody listen to me'),
                     Mood.create_angry('people don\'t respect me'),
                     Mood.create_angry('others treat me like a child')]

    model['ending'] = ['It was a dark time for the Wizarding World.',
                       'In this moment the world started to fall apart.',
                       'The world was ending.',
                       'Then hell broke loose.',
                       'Everything was lost.']

    return model


def generate_world_romance():
    model = generate_common()

    model['interaction'] = ['kissed', 'hugged', 'looked deep into the eyes of']

    model['ending'] = ['They lived happily ever after.',
                       'Suddenly the world was a better place.',
                       'Love is powerful indeed.']

    model['recognition'] = ['saw', 'spotted', 'recognized', 'noticed']

    return model

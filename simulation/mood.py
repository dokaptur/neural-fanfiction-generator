import random as rd


class Mood(object):
    ANGRY = 'angry'
    SAD = 'sad'
    HAPPY = 'happy'
    IN_LOVE = 'in love'
    LONELY = 'lonely'
    DIZZY = 'dizzy'
    ANXIOUS = 'anxious'
    DELIRIOUS = 'delirious'

    def __init__(self, name, reason, statements=[], actions=[], dialogs=[]):
        self.name = name
        self.reason = reason
        self.statements = statements
        self.actions = actions
        self.dialogs = dialogs

    @staticmethod
    def create_angry(reason='everything is simply wrong'):
        statements = ['I hate all Wizarding World!', 'I\'m so angry!', 'I hate humanity!',
                      'Everything is so annoying!', 'Why all people are so stupid?',
                      'I despise all people I know.']
        stats = ['I hate all Wizarding World', 'I\'m so angry']
        statements_reason = ['{} because {}!'.format(s, reason) for s in stats]
        statements += statements_reason

        actions = ['slammed the door', 'hit the wall', 'started to shout incoherently']

        dialogs = ['I hate you!', 'You are such a bastard!', 'Why are you even here?',
                   'I despise you.', 'You are so annoying!', 'I can\'t stand you!',
                   'Go away or I will kill you.', 'Are you here just to make me angry?']

        return Mood(Mood.ANGRY, reason, statements, actions, dialogs)

    @staticmethod
    def create_sad(reason='everything is simply wrong'):
        statements = ['Why the world is so unfair?', 'I\'m so sad!', 'I can\'t stand it anymore!',
                      'I\'m so miserable!', 'My life is a disaster.',
                      'Why the world punish me so hard?!', 'I\'d rather die than live this way!']
        stats = ['I\'m so sad', 'My life is hell', 'I\'m so miserable']
        statements_reason = ['{} because {}.'.format(s, reason) for s in stats]
        statements += statements_reason

        actions = ['cried quietly', 'decided to commit suicide', 'wept',
                   'sat motionless on a floor',
                   'lay down on a ground and didn\'t want to move']

        dialogs = ['Don\'t trouble yourself, I\'m the lost case anyway.',
                   'You are a good person and I will burn in hell.',
                   'Don\'t waste your time on me.',
                   'We will die sooner or later, so why even bother?',
                   'Why do you want to talk to such a looser as me?',
                   'Memento mori.']

        return Mood(Mood.SAD, reason, statements, actions, dialogs)

    @staticmethod
    def create_happy(reason='the world is beautiful'):
        statements = ['I\'m so happy!', 'My life is wonderful!', 'I\'m so lucky!',
                      'I\'m the happiest person on Earth.']
        stats = ['I\'m so happy', 'I\'m so lucky', 'My life is beautiful']
        statements_reason = ['{} because {}.'.format(s, reason) for s in stats]
        statements += statements_reason

        actions = ['started to dance', 'laughed loudly',
                   'started to sing cheerful songs']
        dialogs = ['Don\'t worry!', 'Think of all good things in your life.',
                   'Look at the stars, they are beautiful.', 'You are a perfect friend.',
                   'Can you dance with me?']
        return Mood(Mood.HAPPY, reason, statements, actions, dialogs)

    @staticmethod
    def create_in_love(reason='God'):
        statements = ['I love {} so much!'.format(reason),
                      '{} is the most perfect person I\'ve ever met.'.format(reason),
                      '{} is so beautiful!'.format(reason),
                      'I can\'t imagine my life without {}.'.format(reason)]

        actions = ['wrote a love poem',
                   'starred into a void with dreamy eyes',
                   'started to sing romantic songs',
                   'sighed heavily']
        dialogs = ['I love you so much!', 'For me, you are perfect.',
                   'Would you marry me?', 'I dream of you every night',
                   'I can\'t live without you anymore']
        return Mood(Mood.IN_LOVE, reason, statements, actions, dialogs)

    @staticmethod
    def create_lonely(reason='nobody loves me'):
        statements = ['I\'m so lonely!', 'I\'m trapped in my solitude.',
                      'Nobody cares if I live or die.', 'My loneliness is so miserable.',
                      'Why people avoid me?', 'Nobody loves me!']

        actions = ['cried quietly', 'decided to commit suicide', 'wept',
                   'sat motionless on a floor',
                   'lay down on a ground and didn\'t want to move']

        dialogs = ['You don\'t understand me anyway.', 'I know you don\'t care about me.',
                   'Leave me alone and stop pretending that you care.',
                   'I don\'t need your sympathy.']

        return Mood(Mood.LONELY, reason, statements, actions, dialogs)

    def get_random_action(self):
        r = rd.randint(0, len(self.actions) - 1)
        return self.actions[r]

    def get_random_statement(self):
        r = rd.randint(0, len(self.statements) - 1)
        return self.statements[r]

    def get_random_question(self):
        r = rd.randint(0, len(self.questions) - 1)
        return self.questions[r]

    def get_random_dialog(self):
        r = rd.randint(0, len(self.dialogs) - 1)
        return self.dialogs[r]

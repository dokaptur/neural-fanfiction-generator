import random as rd


class Actor(object):
    def __init__(self, firstname, surname='', gender='male', house=None, items=[]):
        self.firstname = firstname
        self.surname = surname
        self.fullname = '{} {}'.format(firstname, surname)
        self.gender = gender
        self.house = house
        self.items = items
        self.mood = None 
        self.love = None 
        self.location = None 
        self.state = None 
        self.is_leading = False
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
            
    def get_pronoun(self, uppercase=True):
        if self.gender == 'male':
            if uppercase:
                return 'He'
            else:
                return 'he'
        else:
            if uppercase:
                return 'She'
            else:
                return 'she'
        
    def get_name(self, uppercase=True):
        r = rd.random()
        if self.is_leading:
            if r < 0.5:
                return self.get_pronoun(uppercase)
            else:
                return self.firstname
        else:
            if r < 0.6:
                return self.firstname
            elif r < 0.7:
                if self.surname != '':
                    return self.surname
                else:
                    return self.firstname
            else:
                return self.fullname
        
    def interact(self, interaction, actor):
        return '{} {} {}.'.format(self.get_name(), interaction, actor.get_name())
    
    def change_mood(self, mood, adverb=''):
        self.mood = mood
        uppercase = True
        if adverb != '':
            uppercase = False
        return '{} {} felt {}.'.format(adverb, self.get_name(uppercase), self.mood.name)
            
    def go_to(self, location, transition):
        self.location = location
        return '{} {} {}.'.format(self.get_name(), transition, location)
            
    def pc_action_in(self, past_cont_action, location):
        self.state = past_cont_action
        self.location = location
        return '{} was {} in {}.'.format(self.get_name(), self.state, location)
        
    def mood_elaborate(self):
        action = self.mood_action()
        statement = self.mood_statement()
        return '{}\n{}'.format(action, statement)
        
    def mood_action(self):
        action = '{} was so {} that {} {}.'.format(self.get_name(), \
        self.mood.name, self.get_pronoun(False), self.mood.get_random_action())
        return action
    
    def mood_statement(self):
        statement = '\'{}\' - {} said.'.format(self.mood.get_random_statement(), \
        self.get_name(False))
        return statement
        
    def ask_question(self, actor2, question):
        return '\'{}\' {} asked {}.'.format(question, self.get_name(False), actor2.get_name())
        
    def make_dialog(self, actor2):
        lines = rd.randint(3,6)
        actors = [self, actor2]
        index = 0
        dialog = ''
        for l in range(lines):
            dialog += '\'{}\''.format(actors[index].mood.get_random_dialog())
            if l == 0:
                dialog += ' {} said.'.format(actors[index].get_name(False))
            if l != lines-1:
                dialog += '\n'
            index = (index+1) % 2
        return dialog


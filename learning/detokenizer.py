import json

PUNCT = ['.', ',', '?', '!', ';']
CLOSING_PUNCT = ['\"', ')']
OPENING_PUNCT = ['``', '(']
EOS_PUNCT = ['.', '?', '!']


def should_join(word1, word2):
    # word2 is shortcut
    if '\'' in word2 and len(word2) <= 3:
        return True
    # word2 is punctuation
    if word2 in PUNCT:
        return True
    # word2 in closing punctuation
    if word2 in CLOSING_PUNCT:
        return True
    # word1 in opening punctuation
    if word1 in OPENING_PUNCT:
        return True
    return False


def to_upper_if_needed(word, after_eos, vocab_uppercase_map):
    if len(word) > 0 and word.isalpha() and (vocab_uppercase_map.get(word) or after_eos):
        word = word[0].upper() + word[1:]
    return word


def detokenize(story, vocab_uppercase_map):
    is_eos = True
    lines = story.split('\n')
    n_story = ''
    for line in lines:
        words = line.split()
        if len(words) == 0:
            continue
        upper_words = []
        joined_words = []
        for word in words:
            n_word = to_upper_if_needed(word, is_eos, vocab_uppercase_map)
            upper_words.append(n_word)
            if n_word.isalpha():
                is_eos = False
            if n_word in EOS_PUNCT:
                is_eos = True
        last = upper_words[0]
        for i in range(1, len(upper_words)):
            to_join = should_join(last, upper_words[i])
            if to_join:
                last += upper_words[i]
            else:
                joined_words.append(last)
                last = upper_words[i]
        joined_words.append(last)
        n_story += ' '.join(joined_words)
        n_story += '\n'
    return n_story



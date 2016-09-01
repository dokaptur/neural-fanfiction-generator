class Indexer(object):

    def __init__(self, first_words=()):
        self._index = {}
        self._index_to_string = []
        self._frozen = False
        self._reference_ids = [0]
        for w in first_words:
            self.string_to_int(w)

    def freeze(self, frozen = True):
        self._frozen = frozen

    def remember_state(self):
        if self._frozen:
            raise ValueError('Cannot remember the state of an Indexer that is frozen.')
        self._reference_ids.append(len(self._index_to_string))

    def string_to_int(self, string):
        if string in self._index:
            return self._index[string]
        else:
            if not self.frozen:
                result = len(self._index_to_string)
                self._index[string] = result
                self._index_to_string.append(string)
                return result
            else:
                raise KeyError('{} not indexed yet and indexer is frozen'.format(string))

    def int_to_string(self, int):
        return self._index_to_string[int]

    def inv(self, string):
        return self.int_to_string(string)

    def __call__(self, string):
        return self.string_to_int(string)

    def __iter__(self):
        return self._index.__iter__()

    def items(self):
        return self._index.items()

    def ints(self, *strings):
        return [self.string_to_int(string) for string in strings]

    def strings(self, *ints):
        return [self.int_to_string(i) for i in ints]

    def __len__(self):
        return len(self._index_to_string)

    @property
    def index(self):
        return self._index

    @property
    def index(self):
        return self._index

    @property
    def reference_ids(self):
        return self._reference_ids

    @property
    def frozen(self):
        return self._frozen

    @property
    def frozen(self):
        return self._frozen

    def __str__(self):
        l = len(self._index_to_string)
        if l > 20:
            a = min(l, 10)
            b = max(len(self._index_to_string) - 10, 0)
            mid = ', ..., '
        else:
            a, b = l, l
            mid = ''
        return "%s(%s%s%s)" % (self.__class__.__name__,
                               ', '.join([str(x) for x in self._index_to_string[:a]]),
                               mid, ', '.join([str(x) for x in self._index_to_string[b:]]))

    def __repr__(self):
        s = str(self)
        return s + ' with references %s' % str(self._reference_ids)
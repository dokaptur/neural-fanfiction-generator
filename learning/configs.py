import numpy as np
import re


class BasicConfig(object):
    def __init__(self):
        self.batch_size = 15
        self.n_batches = np.inf  # for FixedGenerator it will use all data in directory
        self.emb_dim = 17
        self.hidden_size = 101  # 101
        self.hidden_bow_dim = 10  # 20
        self.input_length = 60  # 80  # maximum length of an input sentence
        self.output_length = 30  # maximum length of an input sentence
        self.n_epochs_training = 30
        self.freq_display = 50
        self.n_layers = 1
        self.checkpoint_frequency = 10
        self.learning_rate = 0.01
        self.use_bow = False

    def to_string(self):
        name = 'config_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(
            self.hidden_bow_dim, self.emb_dim,
            self.hidden_size, self.input_length, self.output_length, self.n_layers,
            self.use_bow
        )
        return name

    def parse_config_from_model_name(self, model_name):
        pattern = "config_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(True|False)"
        p = re.compile(pattern)
        m = p.search(model_name)
        groups = m.groups()
        if len(groups) != 7:
            raise(ValueError("Model name doesn't match expected pattern."))
        self.hidden_bow_dim = int(groups[0])
        self.emb_dim = int(groups[1])
        self.hidden_size = int(groups[2])
        self.input_length = int(groups[3])
        self.output_length = int(groups[4])
        self.n_layers = int(groups[5])
        if groups[6] == 'True':
            self.use_bow = True
        else:
            self.use_bow = False

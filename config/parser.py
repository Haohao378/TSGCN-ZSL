import numpy as np
import torch
import yaml

class YAMLParser:

    def __init__(self, config):
        self.reset_config()
        self.parse_config(config)

    def parse_config(self, file):
        with open(file, encoding='utf-8') as fid:
            yaml_config = yaml.load(fid, Loader=yaml.FullLoader)
        self.parse_dict(yaml_config)

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._device

    @property
    def loader_kwargs(self):
        return self._loader_kwargs

    def reset_config(self):
        self._config = {}
        self._config['data'] = {}

    def update(self, config):
        self.reset_config()
        self.parse_config(config)

    def parse_dict(self, input_dict, parent=None):
        if parent is None:
            parent = self._config
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in parent.keys():
                    parent[key] = {}
                self.parse_dict(val, parent[key])
            else:
                parent[key] = val

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def merge_configs(self, run):
        config = {}
        for key in run.keys():
            if len(run[key]) > 0 and run[key][0] == '{':
                config[key] = eval(run[key])
            else:
                config[key] = run[key]
        self.parse_dict(self._config, config)
        self.combine_entries(config)
        return config

    @staticmethod
    def combine_entries(config):
        if 'spiking_neuron' in config.keys():
            config['model']['spiking_neuron'] = config['spiking_neuron']
            config.pop('spiking_neuron', None)
        return config
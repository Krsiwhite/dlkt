# config.py
import yaml
from types import SimpleNamespace
import json

def dict_to_namespace(d):
    return json.loads(json.dumps(d), object_hook=lambda item: SimpleNamespace(**item))

with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

Config = dict_to_namespace(config_dict)
import yaml
import os


def get_config(path):
    cwd = os.getcwd()
    
    while os.path.basename(cwd) != 'ssic-classifier':
        cwd = os.path.dirname(cwd)
        
    config_path = cwd + path
    with open(config_path, 'r') as f:
        config_dic = yaml.safe_load(f)
    return config_dic
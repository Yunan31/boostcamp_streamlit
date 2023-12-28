import yaml
import os
import gdown
import torch
from model import Model

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def load_model():
    if not os.path.isfile('model.pt'):
        print('No model found, download starting...')
        gdown.download(config['model_path'], 'model.pt', quiet=False)

    model = torch.load('model.pt', map_location=torch.device('cpu'))
    return model
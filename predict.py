import yaml
import os
import gdown
import torch
from model import Model
import transformers

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def load_model():
    if not os.path.isfile('model.pt'):
        print('No model found, download starting...')
        gdown.download(config['model_path'], 'model.pt', quiet=False)

    model = torch.load('model.pt', map_location=torch.device('cpu'))
    return model


def predict(model, sentence_1, sentence_2):
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])

    input = tokenizer(sentence_1, sentence_2, padding=True, truncation=True, return_tensors='pt').to('cpu')
    output = model(input['input_ids'])
    pred = output.detach().numpy().item()

    return round(pred, 1)
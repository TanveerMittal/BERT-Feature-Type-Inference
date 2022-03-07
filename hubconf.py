dependencies = ['torch', 'transformers']
from models import *
import torch
from transformers import AutoModel

model_urls = {"with stats": "https://drive.google.com/uc?export=download&id=1V5Y1Sg54YWVjnYRu47Mv2YO1YD0rwOs2",
              "no stats": "https://drive.google.com/uc?export=download&id=1nM737i1Vr9N4DYAI8rdWmSvR1mrIyWRc"}

def BERT_fti_with_stats(pretrained=True, **kwargs):
    # load pretrained bert_model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # freeze bert weights
    for param in bert.parameters():
        param.requires_grad = False

    # initialize model
    model = transformer_cnn(bert, 256, [1, 2, 3, 5])

    # load pretrained cnn weights
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls["with stats"]), progress=True)

    return model

def BERT_fti_no_stats(pretrained=True, **kwargs):
    # load pretrained bert_model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # freeze bert weights
    for param in bert.parameters():
        param.requires_grad = False

    # initialize model
    model = transformer_cnn(bert, 256, [1, 2, 3, 5])

    # load pretrained cnn weights
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls["no stats"]), progress=True)

    return model
dependencies = ['torch', 'transformers', 'os']
from models import *
import os
import torch
from transformers import AutoModel

model_urls = {"with stats": "https://drive.google.com/uc?export=download&id=1V5Y1Sg54YWVjnYRu47Mv2YO1YD0rwOs2",
              "no stats": "https://drive.google.com/uc?export=download&id=1nM737i1Vr9N4DYAI8rdWmSvR1mrIyWRc"}

def load_statedict_from_online(name):
    torchhome = torch.hub._get_torch_home()
    home = os.path.join(torchhome, "weights")
    if not os.path.exists(home):
        os.makedirs(home)
    filepath = os.path.join(home, "%s.pt" % name)
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None, progress=True)
    return torch.load(filepath)


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
        model.load_state_dict(load_statedict_from_online("BERT_fti_with_stats"))

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
        model.load_state_dict(load_statedict_from_online("BERT_fti_no_stats"))

    return model
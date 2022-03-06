import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from process_data import *
from evaluation import evaluate

repo = "TanveerMittal/BERT-Feature_Type_Inference"

if __name__ == "__main__":
    # check if gpu is avilable
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load benchmark data
    df_zoo_train, df_zoo_test = load_data("data")

    train_data = preprocess(df_zoo_train, " [SEP] ")
    test_data = preprocess(df_zoo_test, " [SEP] ")

    x_train, x_val, y_train, y_val = train_test_split(train_data[['text', "features"]], train_data['label'], 
                                                                        random_state=2018, 
                                                                        test_size=0.2, 
                                                                        stratify=train_data['label'])

    # initialize torch dataloader objects
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(x_train, y_train, x_val, y_val,
                                                                            test_data, model="bert")

    # define loss function
    cross_entropy  = nn.NLLLoss()

    # load and evaluate models
    print("loading BERT CNN model using descriptive statistics")
    model = torch.hub.load(repo, 'BERT_fti_with_stats', pretrained=True)
    model = model.to(device)

    loss, preds, acc = evaluate(model, val_dataloader, cross_entropy, y_val)
    print("validation accuracy:", acc)

    loss, preds, acc = evaluate(model, test_dataloader, cross_entropy, test_data["label"])
    print("test accuracy:", acc)

    print("loading BERT CNN model using only text features")
    model = torch.hub.load(repo, 'BERT_fti_no_stats', pretrained=True)
    model = model.to(device)

    loss, preds, acc = evaluate(model, val_dataloader, cross_entropy, y_val)
    print("validation accuracy:", acc)

    loss, preds, acc = evaluate(model, test_dataloader, cross_entropy, test_data["label"])
    print("test accuracy:", acc)
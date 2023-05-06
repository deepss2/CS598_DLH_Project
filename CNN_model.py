import torch
import pandas as pd
import numpy as np
import functools
import common_utils
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

prediction_class = [
    'Obesity', 'Non.Adherence', 'Developmental.Delay.Retardation',
    'Advanced.Heart.Disease', 'Advanced.Lung.Disease',
    'Schizophrenia.and.other.Psychiatric.Disorders', 'Alcohol.Abuse',
    'Other.Substance.Abuse', 'Chronic.Pain.Fibromyalgia',
    'Chronic.Neurological.Dystrophies', 'Advanced.Cancer', 'Depression',
    'Dementia'
]

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, pred_class):
        self.x = df[['embedding']]
        self.y = df[[pred_class]]
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_val = self.x.iloc[index, :].values
        y_val = self.y.iloc[index, :].values
        return (torch.transpose(torch.Tensor(x_val[0]), 0, 1), torch.Tensor(y_val))

def get_token_embedding(tokens, w2v_model):
    result = []
    for token in tokens:
        if token in w2v_model.vocab:
            result.append(w2v_model.get_vector(token))
        else:
            result.append(np.random.uniform(-0.25, 0.25, 100))
    return np.array(result)
        

def add_embedding(data, max_token_len, w2v_model):
    data['padded_clean_text'] = data.apply(
        lambda x : x['summary_token'] + ['<pad>'] * (max_token_len - len(x['summary_token'])), axis=1)
    data['embedding'] = data.apply(lambda x: get_token_embedding(x['padded_clean_text'], w2v_model), axis=1)
    return data

def generating_embedding_dataloader(train_data, test_data, val_data, embedding_model):
    max_input_length = 0
    for txt in train_data.summary_token.values:
        max_input_length = max(max_input_length, len(txt))
    train_data = add_embedding(train_data, max_input_length, embedding_model)
    test_data = add_embedding(test_data, max_input_length, embedding_model)
    val_data = add_embedding(val_data, max_input_length, embedding_model)
    return (train_data, test_data, val_data, max_input_length)

def generate_data_loader(train_data, test_data, val_data, pred_class):
    train_data_loader = DataLoader(MyDataset(train_data, pred_class), batch_size=32, shuffle=True)
    test_data_loader = DataLoader(MyDataset(test_data, pred_class), batch_size=32, shuffle=False)
    val_data_loader = DataLoader(MyDataset(val_data, pred_class), batch_size=32, shuffle=False)
    return train_data_loader, test_data_loader, val_data_loader

class CNNModelPoolPostDrop(torch.nn.Module):
    def __init__(self, conv_width, max_input_size, embedding_size):
        super().__init__()
        self.conv_width = conv_width
        self.embedding_size= embedding_size
        self.conv_layer = []
        self.output_feature_map = 100
        self.conv1 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 1)
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.maxpool1 = torch.nn.MaxPool1d(max_input_size - 1 + 1)
        
        self.conv2 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 2)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.maxpool2 = torch.nn.MaxPool1d(max_input_size - 2 + 1)
        
        self.conv3 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 3)
        self.drop3 = torch.nn.Dropout(p=0.5)
        self.maxpool3 = torch.nn.MaxPool1d(max_input_size - 3 + 1)
        
        self.conv4 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 4)
        self.drop4 = torch.nn.Dropout(p=0.5)
        self.maxpool4 = torch.nn.MaxPool1d(max_input_size - 4 + 1)
        
        self.linear1 = torch.nn.Linear(self.conv_width * self.output_feature_map,  2)
        
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv2.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv3.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv4.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.linear1.weight, a=-0.05, b=0.05)
        
    

    def forward(self, x):
        outputs = []
        # print(x.shape)
        if(self.conv_width >= 1):
            outputs.append(self.maxpool1(self.drop1(torch.nn.ReLU()(self.conv1(x)))))
            # outputs.append(self.drop1(self.maxpool1(torch.nn.ReLU()(self.conv1(x)))))
        if(self.conv_width >= 2):
            outputs.append(self.maxpool2(self.drop2(torch.nn.ReLU()(self.conv2(x)))))
            # outputs.append(self.drop2(self.maxpool2(torch.nn.ReLU()(self.conv2(x)))))
        # print(self.maxpool3(self.drop3(torch.nn.ReLU()(self.conv3(x)))).shape)
        if(self.conv_width >= 3):
            outputs.append(self.maxpool3(self.drop3(torch.nn.ReLU()(self.conv3(x)))))
            # outputs.append(self.drop3(self.maxpool3(torch.nn.ReLU()(self.conv3(x)))))
        if(self.conv_width >= 4):
            outputs.append(self.maxpool4(self.drop4(torch.nn.ReLU()(self.conv4(x)))))
            # outputs.append(self.drop4(self.maxpool4(torch.nn.ReLU()(self.conv4(x)))))
        flatten_output = torch.nn.Flatten()(torch.cat(outputs, 1))
        result = self.linear1(flatten_output)
        return torch.nn.Softmax(dim=1)(result)

class CNNModelDropPostPool(torch.nn.Module):
    def __init__(self, conv_width, max_input_size, embedding_size):
        super().__init__()
        self.conv_width = conv_width
        self.embedding_size= embedding_size
        self.conv_layer = []
        self.output_feature_map = 100
        self.conv1 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 1)
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.maxpool1 = torch.nn.MaxPool1d(max_input_size - 1 + 1)
        
        self.conv2 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 2)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.maxpool2 = torch.nn.MaxPool1d(max_input_size - 2 + 1)
        
        self.conv3 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 3)
        self.drop3 = torch.nn.Dropout(p=0.5)
        self.maxpool3 = torch.nn.MaxPool1d(max_input_size - 3 + 1)
        
        self.conv4 = torch.nn.Conv1d(embedding_size, self.output_feature_map, 4)
        self.drop4 = torch.nn.Dropout(p=0.5)
        self.maxpool4 = torch.nn.MaxPool1d(max_input_size - 4 + 1)
        
        self.linear1 = torch.nn.Linear(self.conv_width * self.output_feature_map,  2)
        
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv2.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv3.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.conv4.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.linear1.weight, a=-0.05, b=0.05)
        
    

    def forward(self, x):
        outputs = []
        # print(x.shape)
        if(self.conv_width >= 1):
            # outputs.append(self.maxpool1(self.drop1(torch.nn.ReLU()(self.conv1(x)))))
            outputs.append(self.drop1(self.maxpool1(torch.nn.ReLU()(self.conv1(x)))))
        if(self.conv_width >= 2):
            # outputs.append(self.maxpool2(self.drop2(torch.nn.ReLU()(self.conv2(x)))))
            outputs.append(self.drop2(self.maxpool2(torch.nn.ReLU()(self.conv2(x)))))
        # print(self.maxpool3(self.drop3(torch.nn.ReLU()(self.conv3(x)))).shape)
        if(self.conv_width >= 3):
            # outputs.append(self.maxpool3(self.drop3(torch.nn.ReLU()(self.conv3(x)))))
            outputs.append(self.drop3(self.maxpool3(torch.nn.ReLU()(self.conv3(x)))))
        if(self.conv_width >= 4):
            # outputs.append(self.maxpool4(self.drop4(torch.nn.ReLU()(self.conv4(x)))))
            outputs.append(self.drop4(self.maxpool4(torch.nn.ReLU()(self.conv4(x)))))
        flatten_output = torch.nn.Flatten()(torch.cat(outputs, 1))
        result = self.linear1(flatten_output)
        return torch.nn.Softmax(dim=1)(result)


def evaluate_predictions(truth, pred):
    """
    TODO: Evaluate the performance of the predictoin via AUROC, and F1 score
    each prediction in pred is a vector representing [p_0, p_1].
    When defining the scores we are interesed in detecting class 1 only
    (Hint: use roc_auc_score and f1_score from sklearn.metrics, be sure to read their documentation)
    return: auroc, f1
    """
    from sklearn.metrics import roc_auc_score, f1_score

    # your code here
    precision, recall, thresholds = precision_recall_curve(truth, pred)
    f1_scores = 2*recall*precision/(recall+precision+1e-5)
    threshold = thresholds[np.argmax(f1_scores)]
    precision = (precision_score(truth, pred > threshold))   
    recall = recall_score(truth, pred > threshold)
    f1 = f1_score(truth, pred > threshold)
    roc_auc = (roc_auc_score(truth, pred))
    return precision, recall, f1, roc_auc

def eval_model(model, dataloader, device=None):
    device = device or torch.device('cpu')
    model.eval()
    pred_all = []
    Y_test = []
    for X, Y in dataloader:
        y_pred = model(X)[:,1].unsqueeze(1)
        y_pred = y_pred.detach().numpy()
        pred_all.append(y_pred)
        Y_test.append(Y)
    pred_all = np.concatenate(pred_all, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    p, r, f1, auc = evaluate_predictions(Y_test, pred_all)
    return p, r, f1, auc
    
def train(train_loader, val_loader, model, optimizer, criterion):
    torch.manual_seed(0)
    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()
        y_pred = model(data[0])[:,1].unsqueeze(1)
        loss = criterion(y_pred, data[1])
        loss.backward()
        optimizer.step()
    val_loss = 0
    for data in val_loader:  # Iterate in batches over the training/test dataset.
        y_pred = model(data[0])[:,1].unsqueeze(1)
        loss = criterion(y_pred, data[1])
        val_loss += loss.item()
    val_loss /= (len(val_loader))
    print("EPOCH VAl Loss : " + str(val_loss))
    print(eval_model(model, val_loader))

def train_model_for_phenotype(train_data, val_data, test_data, max_input_length, pred_class, conv_width=1, embedding_size=100):
    train_data_loader, test_data_loader, val_data_loader = generate_data_loader(train_data, test_data, val_data, pred_class)
    model = CNNModelDropPostPool(conv_width, max_input_length, embedding_size)
    print(model)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    criterion = torch.nn.BCELoss()
    for epoch in range(0, 20):
        train(train_data_loader, val_data_loader, model, optimizer, criterion)
    print("Test Result : ")
    p, r, f1, auc = eval_model(model, test_data_loader)
    result = []
    result.append((pred_class, conv_width, p, r, f1, auc))
    print(result)
    return result;

def main():
    for pred in prediction_class:
        result = train_model_for_phenotype(train_data, val_data, test_data, 9797, pred)
        with open(r'./cnn_result/' + pred + '.txt' , 'w') as fp:
            for res in result:
                fp.write(functools.reduce(lambda x, y: str(x) + "," + str(y), res, '') + "\n")
        

data = common_utils.read_data('MIMIC-III.csv')
label = common_utils.read_label('annotation.csv')
w2v_model = common_utils.read_word2vec_model('mimiciii_word2vec.wordvectors')
merged_data = common_utils.join_data_with_label(data, label)
train_data, val_data, test_data = common_utils.train_test_val_split(merged_data, 0.7, 0.2, 0.1)
train_data, val_data, test_data, max_input_length = generating_embedding_dataloader(train_data, val_data, test_data, w2v_model)
# train_data_loader, test_data_loader, val_dataloader = generate_data_loader(train_data, test_data, val_data, 'Obesity')
# train_model_for_phenotype(train_data, test_data, val_data, 9797, 'Obesity', 2)
main()


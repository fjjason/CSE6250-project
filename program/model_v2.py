import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from time import time
import pickle
import lightgbm as lgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Embedding
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def show_plots(df):
    plt.figure(figsize=[10, 10])
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=0, y=1, hue='y', data=df)
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=4, y=5, hue='y', data=df)
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=8, y=9, hue='y', data=df)
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=12, y=13, hue='y', data=df)
    plt.show()


X, y, _ = pickle.load(open('data/model_data_v1.sav', 'rb'))

X = X[:, :12]
df = pd.DataFrame(X)
df['y'] = y
df.to_csv('data/model_data_v2.csv', index=False)

# show_plots(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1)


test_batch_size = X_test.shape[0]


torch.manual_seed(1)

trainset = TensorDataset(torch.from_numpy(X_train.astype('float32')).unsqueeze(
    2), torch.from_numpy(y_train.astype('long')))
trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

validset = TensorDataset(torch.from_numpy(X_valid.astype('float32')).unsqueeze(
    2), torch.from_numpy(y_valid.astype('long')))
validloader = DataLoader(trainset, batch_size=10, shuffle=False, num_workers=2)

testset = TensorDataset(torch.from_numpy(X_test.astype('float32')).unsqueeze(
    2), torch.from_numpy(y_test.astype('long')))
testloader = DataLoader(
    trainset, batch_size=test_batch_size, shuffle=False, num_workers=2)

classes = ('Cassette', 'Telemetry')


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=64,
                          num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x


model = RNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


train_losses = []
valid_losses = []

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_counter = 0
    print(f'Epoch: {epoch}')
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * inputs.size(0))
        train_counter += inputs.size(0)

    train_losses.append(train_loss/train_counter)

    model.eval()
    valid_loss = 0.0
    valid_counter = 0
    with torch.no_grad():
        for i, data in enumerate(validloader, 0):
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += (loss.item() * inputs.size(0))
            valid_counter += inputs.size(0)
    valid_losses.append(valid_loss / valid_counter)

print('finished training')

plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc="best")
plt.show()


class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        # get the inputs
        inputs, targets = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets).squeeze()
        for i, label in enumerate(targets):
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %s : %2d%% out of %d cases' %
          (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))


data = next(iter(testloader))
inputs, targets = data
outputs = model(inputs)
probability, predicted = torch.max(outputs.data, 1)
c = (predicted == targets).squeeze()


eval_metrics = pd.DataFrame(np.empty([2, 4]))
eval_metrics.index = ["baseline"] + ['RNN']
eval_metrics.columns = ["Accuracy", "ROC AUC", "PR AUC", "Log Loss"]

pred = np.repeat(0, len(y_test))
pred_proba = np.repeat(0.5, len(y_test))
eval_metrics.iloc[0, 0] = accuracy_score(y_test, pred)
eval_metrics.iloc[0, 1] = roc_auc_score(y_test, pred_proba)
eval_metrics.iloc[0, 2] = average_precision_score(y_test, pred_proba)
eval_metrics.iloc[0, 3] = log_loss(y_test, pred_proba)

eval_metrics.iloc[1, 0] = accuracy_score(y_test, predicted)
eval_metrics.iloc[1, 1] = roc_auc_score(y_test, probability)
eval_metrics.iloc[1, 2] = average_precision_score(y_test, probability)
eval_metrics.iloc[1, 3] = 0  # log_loss(y_test, pred_proba[:, 1])

print(eval_metrics)
#           Accuracy   ROC AUC   PR AUC  Log Loss
# baseline  0.928000  0.500000  0.07200  0.693147
# RNN       0.869333  0.482014  0.07012        ??

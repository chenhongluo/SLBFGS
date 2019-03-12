import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
import time
import sys
sys.path.append('../../functions/')

from utils import compute_stats, get_grad
from LBFGS import LBFGS

# customized class for RCV1 dataset
class RCV1DataSet(Dataset):
    def __init__(self, data_files, root_dir, transform=None):
        self.data_files = data_files
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.doc_id = []
        self.label = []
        self.dim = 47237
        self.n_docs = 0
        for filename in data_files:
            path = os.path.join(root_dir, filename)
            file = open(path)
            lines = file.readlines()
            for line in lines:
                elems = line.split(' ')
                elems.pop()  # remove last '\n'
                self.data.append([])
                self.doc_id.append(int(elems[0]))
                self.label.append(int(elems[1]))
                for i in range(2, len(elems), 2):
                    index = int(elems[i])
                    value = float(elems[i+1])
                    self.data[-1].append(index)
                    self.data[-1].append(value)

            file.close()
        self.n_docs = len(self.data)
        print('%d documents in total' % self.n_docs)

    def __len__(self):
        return self.n_docs

    def __getitem__(self, idx):
        doc = np.zeros(shape=(self.dim,), dtype=np.float32)
        for i in range(0, len(self.data[idx]), 2):
            index = self.data[idx][i]
            value = self.data[idx][i + 1]
            doc[index] = value
        label = self.label[idx]
        # print('select %d' % idx)
        return torch.tensor(doc), torch.tensor(label)

    def getItems(self):
        docs = np.zeros(shape=(len(self.data),self.dim), dtype=np.float32)
        labels = np.zeros(shape=(len(self.data)), dtype=np.float32)
        for k in range(len(self.data)):
            for i in range(0, len(self.data[k]), 2):
                index = self.data[k][i]
                value = self.data[k][i + 1]
                docs[k][index] = value
            labels[k] = self.label[k]
        return docs,labels


# Hyper Parameters
input_size = 47237
num_classes = 2
num_epochs = 1
batch_size = 10000
learning_rate = 1.0
max_iter = 200                      # note each iteration is NOT an epoch
ghost_batch = 128
overlap_ratio = 0.25                # should be in (0, 0.5)

print(torch.cuda.is_available())
device = torch.device("cuda")

train_files = []
for i in range(0, 30):
    train_files.append('RCV1_%d-80.data' % i)
test_files = []
for i in range(30, 80):
    test_files.append('RCV1_%d-80.data' % i)

# Dataset
train_dataset = RCV1DataSet(data_files=train_files, root_dir='./data/RCV1')

test_dataset = RCV1DataSet(data_files=test_files, root_dir='./data/RCV1')

# batch_size = train_dataset.n_docs
# Dataset Loader (Input Pipline)
# batch_size = len(train_dataset)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=100,
#                                           shuffle=False)

(X_train, y_train) = train_dataset.getItems()
(X_test, y_test) = test_dataset.getItems()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes).to(device)
opfun = lambda X: model.forward(torch.from_numpy(X).to(device))

predsfun = lambda op: np.argmax(op.cpu().data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100

# %% Define optimizer

optimizer = LBFGS(model.parameters(), lr=learning_rate, history_size=10, line_search='None', debug=True)

# %% Main training loop
# # Training the Model

Ok_size = int(overlap_ratio * batch_size)
Nk_size = int((1 - 2 * overlap_ratio) * batch_size)

# sample previous overlap gradient
random_index = np.random.permutation(range(X_train.shape[0]))
Ok_prev = random_index[0:Ok_size]
g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)

# main loop
end = 0

for n_iter in range(max_iter):
    begin = time.time()
    # training mode
    model.train()

    # sample current non-overlap and next overlap gradient
    random_index = np.random.permutation(range(X_train.shape[0]))
    Ok = random_index[0:Ok_size]
    Nk = random_index[Ok_size:(Ok_size + Nk_size)]

    # compute overlap gradient and objective
    g_Ok, obj_Ok = get_grad(optimizer, X_train[Ok], y_train[Ok], opfun)

    # compute non-overlap gradient and objective
    g_Nk, obj_Nk = get_grad(optimizer, X_train[Nk], y_train[Nk], opfun)

    # compute accumulated gradient over sample
    g_Sk = overlap_ratio * (g_Ok_prev + g_Ok) + (1 - 2 * overlap_ratio) * g_Nk

    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-g_Sk)

    # perform line search step
    lr = optimizer.step(p, g_Ok, g_Sk=g_Sk)

    # compute previous overlap gradient for next sample
    Ok_prev = Ok
    g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)

    # curvature update
    optimizer.curvature_update(g_Ok_prev, eps=0.2, damping=True)

    # compute statistics

    end = time.time() - begin


    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test,
                                                    y_test, opfun, accfun, ghost_batch=128)

    # print data
    print('Iter:', n_iter + 1, 'lr:', lr, 'Training Loss:', train_loss,
          'Test Loss:', test_loss, 'Test Accuracy:', test_acc, 'training time: %.2f seconds' %end)

# begin = time.time()
# # Training the Model
# for epoch in range(num_epochs):
#     for i, (docs, labels) in enumerate(train_loader):
#         docs = docs.to(device)
#         labels = labels.to(device)
#
#         # Forward + Backward + Optimize
#         def closure():
#             optimizer.zero_grad()
#             outputs = model(docs)
#             loss = criterion(outputs, labels)
#             print('Loss: %e'
#                   % (loss.data))
#             loss.backward()
#             return loss
#
#
#         optimizer.step(closure)
# end = time.time()
# print('training time: %.2f seconds' % (end-begin))
# # Test the Model
# correct = 0
# total = 0
# for docs, labels in test_loader:
#     docs = docs.to(device)
#     labels = labels.to(device)
#     outputs = model(docs)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Accuracy of the model on the test set: %.2f %%' % (100.0 * correct / total))

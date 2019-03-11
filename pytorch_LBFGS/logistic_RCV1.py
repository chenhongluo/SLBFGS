import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
import time

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


# Hyper Parameters
input_size = 47237
num_classes = 2
num_epochs = 1
batch_size = 10000
learning_rate = 1.0

print(torch.cuda.is_available())
device = torch.device("cuda")

train_files = []
for i in range(0, 5):
    train_files.append('RCV1_%d-80.data' % i)
test_files = []
for i in range(5, 15):
    test_files.append('RCV1_%d-80.data' % i)

# Dataset
train_dataset = RCV1DataSet(data_files=train_files, root_dir='./data/RCV1')

test_dataset = RCV1DataSet(data_files=test_files, root_dir='./data/RCV1')

batch_size = train_dataset.n_docs

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes).to(device)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=200, history_size=100)

begin = time.time()
# Training the Model
for epoch in range(num_epochs):
    for i, (docs, labels) in enumerate(train_loader):
        docs = docs.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        def closure():
            optimizer.zero_grad()
            outputs = model(docs)
            loss = criterion(outputs, labels)
            print('Loss: %e'
                  % (loss.data))
            loss.backward()
            return loss


        optimizer.step(closure)
end = time.time()
print('training time: %.2f seconds' % (end-begin))
# Test the Model
correct = 0
total = 0
for docs, labels in test_loader:
    docs = docs.to(device)
    labels = labels.to(device)
    outputs = model(docs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the test set: %.2f %%' % (100.0 * correct / total))

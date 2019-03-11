"""
Full-Overlap L-BFGS Implementation with Stochastic Wolfe Line Search

Demonstrates how to implement full-overlap L-BFGS with stochastic weak Wolfe line
search without Powell damping to train a simple convolutional neural network using the 
LBFGS optimizer. Full-overlap L-BFGS is a stochastic quasi-Newton method that uses 
the same sample as the one used in the stochastic gradient to perform quasi-Newton 
updating, then resamples an entirely independent new sample in the next iteration.

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 9/21/18.

Requirements:
    - Keras (for CIFAR-10 dataset)
    - NumPy
    - PyTorch

Run Command:
    python full_overlap_lbfgs_example.py

Based on stable quasi-Newton updating introduced by Schraudolph, Yu, and Gunter in
"A Stochastic Quasi-Newton Method for Online Convex Optimization" (2007)

"""

import sys
import time

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from keras.datasets import cifar10 # to load dataset

from utils import compute_stats, get_grad
from LBFGS import LBFGS

#%% Parameters for L-BFGS training

max_iter = 5000
ghost_batch = 128
batch_size = 512

#%% Load data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 127.5 - 1
X_test = X_test / 127.5 - 1

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

#%% Define network

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 1000)
#         self.fc2 = nn.Linear(1000, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

#%% Check cuda availability
        
cuda = torch.cuda.is_available()
    
#%% Create neural network model
        
if(cuda):
    torch.cuda.manual_seed(2018)
    model = ResNet20().cuda()
else:
    torch.manual_seed(2018)
    model = ResNet20()

# model = LogisticRegression(1024, 10).to('cuda')
#%% Define helper functions

# Forward pass
if(cuda):
    opfun = lambda X: model.forward(torch.from_numpy(X).cuda())
else:
    opfun = lambda X: model.forward(torch.from_numpy(X))

# Forward pass through the network given the input
if(cuda):
    predsfun = lambda op: np.argmax(op.cpu().data.numpy(), 1)
else:
    predsfun = lambda op: np.argmax(op.data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun  = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100

#%% Define optimizer

optimizer = LBFGS(model.parameters(), lr=1.0, history_size=200, line_search='Wolfe', debug=False)

#%% Main training loop

begin_total = time.time()
timer_grad = 0.0
timer_two_loop = 0.0
timer_line_search = 0.0
timer_data = 0.0
timer_update = 0.0
timer_eval = 0.0
# main loop
for n_iter in range(max_iter):
    
    # training mode
    model.train()

    begin = time.time()
    # sample batch
    random_index = np.random.permutation(range(X_train.shape[0]))
    Sk = random_index[0:batch_size]
    end = time.time()
    timer_data = timer_data + end - begin

    begin = time.time()
    # compute initial gradient and objective
    grad, obj = get_grad(optimizer, X_train[Sk], y_train[Sk], opfun)
    end = time.time()
    timer_grad = timer_grad + end - begin

    begin = time.time()
    # two-loop recursion to compute search direction
    p = optimizer.two_loop_recursion(-grad)
    end = time.time()
    timer_two_loop = timer_two_loop + end - begin

    # define closure for line search
    def closure():              
        
        optimizer.zero_grad()
        
        if(torch.cuda.is_available()):
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)

        l2_reg = None
        for W in model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        l2_reg = l2_reg * 0.0001

        for subsmpl in np.array_split(Sk, max(int(batch_size / ghost_batch), 1)):

            ops = opfun(X_train[subsmpl])

            if (torch.cuda.is_available()):
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()

            loss_fn += (F.cross_entropy(ops, tgts) + l2_reg) * (len(subsmpl) / batch_size)

        return loss_fn

    begin = time.time()
    # perform line search step
    options = {'closure': closure, 'current_loss': obj}
    obj, grad, lr, _, _, _, _, _ = optimizer.step(p, grad, options=options)
    # obj, grad, lr, _, _, _, _, _ = optimizer.step(p, grad, options=options)
    end = time.time()
    timer_line_search = timer_line_search + end - begin

    begin = time.time()
    # curvature update
    optimizer.curvature_update(grad)
    end = time.time()
    timer_update = timer_update + end - begin

    begin = time.time()
    # compute statistics
    if n_iter * batch_size - ((n_iter * batch_size) // 50000) * 50000 < batch_size:
        model.eval()
        train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test,
                                                        y_test, opfun, accfun, ghost_batch=ghost_batch)

        # print data
        print('Iter: %d lr: %f Training loss: %f Test loss: %f Test acc: %.2f%%' % (n_iter+1, 1.0, train_loss, test_loss, test_acc))
        end = time.time()
        timer_eval = timer_eval + end -begin

end_total = time.time()

print('%.2f seconds in total' % (end_total-begin_total))
print('%.2f seconds for data' % timer_data)
print('%.2f seconds for grad' % timer_grad)
print('%.2f seconds for two-loop' % timer_two_loop)
print('%.2f seconds for line search' % timer_line_search)
print('%.2f seconds for update' % timer_update)
print('%.2f seconds for eval' % timer_eval)

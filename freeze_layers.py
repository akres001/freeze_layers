import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import numpy as np
import random
from pprint import pprint

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden_task1 = nn.Linear(3, 3, bias=False)
        self.hidden_task2 = nn.Linear(2, 3, bias=False)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(3, 4, bias=False)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, task='task1'):
        # Pass the input tensor through each of our operations
        if task == 'task1': 
            x = self.hidden_task1(x)
        else:
            x = self.hidden_task2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x
    
    def freeze_params(self, params_str):
        for n, p in self.named_parameters():
            if n in params_str:
                p.grad = None
                
    def freeze_params_grad(self, params_str):
        for n, p in self.named_parameters():
            if n in params_str:
                p.requires_grad = False
                
    def unfreeze_params_grad(self, params_str):
        for n, p in self.named_parameters():
            if n in params_str:
                p.requires_grad = True
                
def main(args):
    
    device = "cuda:0"
        
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
  
    net = Network()
    net = net.to(device)  
 
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()

    # create your optimizer
    if args.sgd:
        optimizer = optim.SGD(net.parameters(), lr=0.9)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=0.9)

    input = torch.randn(10, 3).to(device)
    input1 = torch.randn(10, 2).to(device)
    
    target = torch.randint(0, 4, (10, )).long().to(device)  
    target1 = torch.randint(0, 4, (10, )).long().to(device)  
    
    
    original_param = {n : p.clone() for (n, p) in net.named_parameters()}
    print("Original params ")
    pprint(original_param)
    print(100 * "=")
    
    # set requires_grad to False for selected layers
    if args.grad_freeze:
        net.freeze_params_grad(['hidden_task2.weight'])
        
    # for _ in range(10):
    output = net(input, task='task1')
    optimizer.zero_grad()   # zero the gradient buffers
    loss1 = criterion(output, target)
    loss1.backward()
    print("Gradients")
    for n, p in net.named_parameters():
        print(n , p.grad)
    
    if not args.grad_freeze:
        net.freeze_params(['hidden_task2.weight'])
    optimizer.step()    # Does the update
        
    
    if args.grad_freeze:
        net.unfreeze_params_grad(['hidden_task2.weight'])
    
    print("Params after task 1 update ")
    params_hid = {n : p.clone() for (n, p) in net.named_parameters()}
    pprint(params_hid)
    print(100 * "=")

    changed_parameters(original_param, params_hid)
    
    if args.grad_freeze:
        net.freeze_params_grad(['hidden_task1.weight'])
        
    # for _ in range(10):
    output = net(input1, task='task2')
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    loss2 = criterion1(output, target1)
    loss2.backward()
    
    print("Gradients")
    for n, p in net.named_parameters():
        print(n , p.grad)

    if not args.grad_freeze:
        net.freeze_params(['hidden_task1.weight'])

    optimizer.step()    # Does the update
    
    if args.grad_freeze:
        net.unfreeze_params_grad(['hidden_task1.weight'])
    
    print("Params after task 2 update ")
    params_hid1 = {n : p.clone() for (n, p) in net.named_parameters()}
    pprint(params_hid1)
    changed_parameters(params_hid, params_hid1)

        
def changed_parameters(initial, final):
    for n, p in initial.items():
        if not torch.allclose(p, final[n]):
            print("Changed : ", n)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int) 
    parser.add_argument('--grad_freeze', action='store_true')
    parser.add_argument('--sgd', action='store_true')
    
    
    args = parser.parse_args()
    
    print(args)

    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import numpy as np
import random
from pprint import pprint
import torch.distributed as dist

from torch.distributed import init_process_group

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden_task1 = nn.Linear(3, 3, bias=False)
        self.hidden_task2 = nn.Linear(2, 3, bias=False)
        self.output = nn.Linear(3, 4, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, task='task1', mode='normal', freeze_params=None):
        if mode == 'normal':
            if task == 'task1': 
                x = self.hidden_task1(x)
            else:
                x = self.hidden_task2(x)
            x = self.sigmoid(x)
            x = self.output(x)
            x = self.softmax(x)
            return x
        elif mode == 'require_grad_do':
            self.freeze_params_grad(freeze_params)
        elif mode == 'require_grad_undo':
            self.unfreeze_params_grad(freeze_params)
        elif mode == 'grad_freeze_none':
            self.freeze_params(freeze_params)
    
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
    if args.distributed:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = int(os.environ["LOCAL_RANK"])
    else:
        device = "cuda:0"
        
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.distributed:
        rank = dist.get_rank() == 0
    else:
        rank = True

    net = Network()
    net = net.to(device)  
    
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()

    # create your optimizer
    if args.sgd:
        optimizer = optim.SGD(net.parameters(), lr=0.9)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.9)

    input = torch.randn(10, 3).to(device)
    input1 = torch.randn(10, 2).to(device)
    
    target = torch.randint(0, 4, (10, )).long().to(device)  
    target1 = torch.randint(0, 4, (10, )).long().to(device)  
    
    
    original_param = {n : p.clone() for (n, p) in net.named_parameters()}
    if rank:
        print("Original params ")
        pprint(original_param)
        print(100 * "=")
    
    if args.grad_freeze:
        net(None, mode='require_grad_do', freeze_params = ['hidden_task2.weight'])
        
    for _ in range(10):
        output = net(input, task='task1')
        optimizer.zero_grad()   # zero the gradient buffers
        loss1 = criterion(output, target)
        loss1.backward()
        if not args.grad_freeze:
            net(None, mode='grad_freeze_none', freeze_params = ['hidden_task2.weight'])
        optimizer.step()    # Does the update
        
    if args.grad_freeze:
        net(None, mode='require_grad_undo', freeze_params = ['hidden_task2.weight'])
    
    if rank:
        print("Params after hidden ")
        params_hid = {n : p.clone() for (n, p) in net.named_parameters()}
        pprint(params_hid)
        print(100 * "=")
        
        changed_parameters(original_param, params_hid)

    if args.grad_freeze:
        net(None, mode='require_grad_do', freeze_params = ['hidden_task1.weight'])
        
    for _ in range(10):
        output = net(input1, task='task2')
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        loss2 = criterion1(output, target1)
        loss2.backward()
        
        if not args.grad_freeze:
            net(None, mode='grad_freeze_none', freeze_params = ['hidden_task1.weight'])
            
        optimizer.step()    # Does the update
    
    if args.grad_freeze:
        net(None, mode='require_grad_undo', freeze_params = ['hidden_task1.weight'])
    
    if rank:
        print("Params after hidden 1 ")
        params_hid1 = {n : p.clone() for (n, p) in net.named_parameters()}
        pprint(params_hid1)
        changed_parameters(params_hid, params_hid1)

        
def changed_parameters(initial, final):
    for n, p in initial.items():
        if not torch.allclose(p, final[n]):
            print("Changed : ", n)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--seed', default=0, type=int) 
    parser.add_argument('--grad_freeze', action='store_true')
    parser.add_argument('--sgd', action='store_true')
    
    
    args = parser.parse_args()
    
    print(args)

    main(args)
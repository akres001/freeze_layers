### For distributed run using with SGD (remove --sgd to use ADAM) and freezing with required_grad
```
!torchrun --standalone --nproc_per_node=2 freeze_layers_distr.py --distributed --sgd
``` 
### For distributed run using freezing setting .grad to None
```
!torchrun --standalone --nproc_per_node=2 freeze_layers_distr.py --distributed --grad_freeze 
```
### For not distributed run (--sgd and --grad_freeze as explained above)
```
!python freeze_layers.py 
```

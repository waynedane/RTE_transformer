# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 

# Requirement
- python 3.7+
- pytorch 1.3.1
- torchtext 0.4.0
- tqdm
- jsonlines
- dill
- numpy

- transformer

# Usage
### 1) data目录下，解压出数据
data/SNLI/SNLI.zip
```bash
unzip data/SNLI/SNLI.zip
```

### 2) Train the model
假设我们只在一台机器上运行，可用卡数是4
```bash
python -m torch.distributed.launch --nproc_per_node=4 train_net.py
```
或者假设只用0，1，2，3号卡
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_net.py
```




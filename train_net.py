import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformer.Models import Encoder,Encoder_Sub,get_pad_mask
from transformer.Optim import ScheduledOptim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import jsonlines
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import logging
def log(name='info',log_path="info.log",level=logging.DEBUG):
    #创建logger，如果参数为空则返回root logger
    logger = logging.getLogger(name)
#     level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    logger.setLevel(level)  #设置logger日志等级

    #这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not logger.handlers:
        #创建handler
        fh = logging.FileHandler(log_path,encoding="utf-8")
        ch = logging.StreamHandler()

        #设置输出日志格式
        formatter = logging.Formatter(
            fmt="%(lineno)d %(asctime)s %(name)s %(filename)s %(levelname)s %(message)s ",
            datefmt="%Y/%m/%d %X"
            )

        #为handler指定输出格式
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        #为logger添加的日志处理器
        logger.addHandler(fh)
        logger.addHandler(ch)
    #use:
    # logger = log()
    # logger.warning("泰拳警告")
    # logger.info("提示")
    # logger.error("错误")
    # logger.debug("查错")
    return logger #直接返回logger
#创建一个全局logger进行训练纪录
logger=log()



def load_data_from_file(path):
    with open(path,mode='r+',encoding='utf8') as f:
        data_list=[]
        for step,item in enumerate(jsonlines.Reader(f)):
            if item['gold_label']=='-':
                continue
            data_list.append({'index':step,'sentence1':item['sentence1'],'sentence2':item['sentence2'],'gold_label':item['gold_label']})
    return data_list


# 创建数据读取器data_reader
def snli_data_reader(data_path,label_dict,padding_idx=0):
#     with open(data_path, "rb") as pkl:
    datas=load_data_from_file(data_path)
    # newdatas=datas.copy()
    # 打乱数据
    # datas=load_dataset(data_path)
    # random.shuffle(newdatas)
    # 开始获取每个数据和标签
    #分别处理'premises', 'hypotheses'
    #1.先分别计算出最大长度
    premises_lengths=[]
    hypotheses_lengths=[]
    #
    all_data_list=[]
    label_list=[]
    all_data_index=[]
#     all_data_mask=[]
    #
    for item in datas:
        premises_lengths.append(len(item['sentence1']))
        hypotheses_lengths.append(len(item['sentence2']))
        label_list.append(label_dict[item['gold_label']])
        
        token_itemA = tokenizer(item['sentence1'],return_tensors="pt")
        token_itemB = tokenizer(item['sentence2'],return_tensors="pt")
        
        all_data_list.append((token_itemA.input_ids,token_itemB.input_ids))#A,B输入序列的id
        
        #ab句子对应的mask位置0代表a巨，1代表b句的位置
        all_data_index.append((token_itemA.token_type_ids,token_itemB.token_type_ids))
        
#         all_data_mask.append(token_item.attention_mask)
    
    #最大长度得算上cls+seq
    #分别处理a，b,找到两者最大的长度
    maxlen=max(max(premises_lengths),max(hypotheses_lengths))+2
    
    pad_data_list=[]
    pad_index_list=[]
#     pad_mask_list=[]
    
    #2.对数据进行pad填充
    for data_id,data_item in enumerate(zip(all_data_list,all_data_index)):
        p_data,data_index=data_item
#         pad_data,pad_mask = len_process(p_data,data_mask,maxlen,padding_idx)#填充pad值
        #填充
        a_pad_value = torch.full([1, maxlen-p_data[0].shape[1]], padding_idx,dtype = torch.int64)
        b_pad_value = torch.full([1, maxlen-p_data[1].shape[1]], padding_idx,dtype = torch.int64)
        
#         pad_mask = torch.cat([data_mask,pad_value],1)
        
        a_pad_index = torch.cat([data_index[0],a_pad_value],1).view(-1)
        b_pad_index = torch.cat([data_index[1],b_pad_value],1).view(-1)
        
        a_pad_data = torch.cat([p_data[0],a_pad_value],1).view(-1)
        b_pad_data = torch.cat([p_data[1],b_pad_value],1).view(-1)
        
        pad_data_list.append((a_pad_data,b_pad_data))
        pad_index_list.append((a_pad_index,b_pad_index))
#         pad_mask_list.append(pad_mask)
    return pad_data_list,pad_index_list,label_list,maxlen

class SNLI_Dataset(Dataset):
    def __init__(self,filepath,label_dict,pad_id):
#         pad_data_list,pad_mask_list,all_data_index,label_list
        self.data,self.data_index,self.cls,self.maxlen=snli_data_reader(filepath,label_dict,pad_id)
        self.len = len(self.cls)
        
    def data_max_len(self):
        return self.maxlen
    
    def __getitem__(self, index):
        return (self.data[index][0],self.data_index[index][0],#premise
                self.data[index][1],self.data_index[index][1],#hypothese
                self.cls[index])
    
    def __len__(self):
        return self.len


class Dual_Encoder(nn.Module):
    def __init__(
            self, n_src_vocab,src_pad_idx,class_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            ):

        super().__init__()

        self.pad_idx = src_pad_idx
        #
        self.sup_enc = Encoder(n_src_vocab = n_src_vocab, n_position = n_position,
                d_word_vec = d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout)

        self.sub_enc=Encoder_Sub(
                    d_model=d_model, d_inner=d_inner,
                    n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                    dropout=dropout)
        
        #batch,len,emb_sice
        self.fc = nn.Linear(d_word_vec*2, class_num)

    #txtEmb = 
    # input_ids（batch,len）: txt a ,b进行cls-pad-seq填充后的句子,id表示
    # index_mask（bacth,len）：标识A-B句子对应位置的mask,1-A,2-B，
    def forward(self,txta_ids,txta_index,txtb_ids,txtb_index):
#         txta_ids,txta_index:batch,len
        inputa_mask = get_pad_mask(txta_ids, 0)
        inputb_mask = get_pad_mask(txtb_ids, 0)

    #     print(input_mask)
    #     new_ipt_mask = torch.squeeze(input_mask,dim=1)
        #
        embA_out = self.sup_enc(txta_ids,inputa_mask)[0]
        embB_out = self.sup_enc(txtb_ids,inputb_mask)[0]
        #batch,len,emb_size
        inter_emb_out = self.sub_enc(embB_out,inputb_mask,embA_out)[0]
        #batch,len,emb_size
        mean_emb = inter_emb_out.max(dim=1).values
        max_emb = inter_emb_out.mean(dim=1)
        #batch,emb_size
        cat_out = torch.cat([mean_emb,max_emb],1)
        #batch,emb_size*2
        fc_out = self.fc(cat_out)
        #batch,cls_num
        logit = F.log_softmax(fc_out, dim=1)
        return logit#batch,cls_num



    
def Model(arg):
    net = Dual_Encoder(arg['n_src_vocab'],arg['src_pad_idx'],arg['class_num'],
            arg['d_word_vec'], arg['d_model'], arg['d_inner'],
            arg['n_layers'], arg['n_head'], arg['d_k'], arg['d_v'], arg['dropout'], arg['n_position'],
            )
    return net

        
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



        
def train(dataLoader, net, criterion, optimizer,device):
    #训练
    net.train()
    
    acc_list=[]
    loss_list=[]
    for i, items in enumerate(dataLoader):
        # 将数据转移到 controller 所在 GPU
        txta_ids,txta_index,txtb_ids,txtb_index,cls=[d.to(device) for d in items]
        # txta_ids,txta_index,txtb_ids,txtb_index,cls=items

        optimizer.zero_grad()
        
        output = net(txta_ids,txta_index,txtb_ids,txtb_index)
        loss = criterion(output, cls)
        # measure accuracy and record loss
        acc, = accuracy(output, cls, topk=(1,))[0]
        loss.backward()
        optimizer.step_and_update_lr()
        # optimizer.step()
        #
        acc_list.append(acc.item())
        loss_list.append(loss.item())
        
    return sum(acc_list)/len(acc_list),sum(loss_list)/len(loss_list)


    
def valid(val_loader, model, criterion,device):
    # switch to evaluate mode
    model.eval()
    
    acc_list=[]
    loss_list=[]
    with torch.no_grad():
        for i, items in enumerate(val_loader):
            # compute output
            txta_ids,txta_index,txtb_ids,txtb_index,cls=[d.to(device) for d in items]
            
            output = model(txta_ids,txta_index,txtb_ids,txtb_index)
            loss = criterion(output, cls)
            # measure accuracy and record loss
            acc= accuracy(output, cls, topk=(1, ))[0]
            acc_list.append(acc.item())
            loss_list.append(loss.item())
    return sum(acc_list)/len(acc_list),sum(loss_list)/len(loss_list)

# def load_model_param(model,param_path,optimizer=None):
#     checkpoint = torch.load(param_path)
#     model.load_state_dict(checkpoint['state_dict'])
    
#     if optimizer:
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         return model,optimizer
#     else:
#         return model

def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
    if checkpoint != 'No':
        logger.info("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        logger.info('PreTrain-Model valid Acc:%s，Loss:%s',modelCheckpoint['valid_acc'],modelCheckpoint['valid_loss'])
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        logger.info('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        logger.info("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer == True:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            logger.debug('loaded! optimizer')
        else:
            logger.debug('not loaded optimizer')
    else:
        logger.warning('No checkpoint is included')
    return model, optimizer

#
def test(dataloader,model,model_path):
    #读取模型传入参数进行测试
    model, optimizer = load_checkpoint(model,model_path,None,loadOptimizer=False)

    model.eval()
    acc_list=[]
    with torch.no_grad():
        for i, items in enumerate(dataloader):
            # compute output
            # txta_ids,txta_index,txtb_ids,txtb_index,cls=[d.to(device) for d in items]
            txta_ids,txta_index,txtb_ids,txtb_index,cls=items
            output = model(txta_ids,txta_index,txtb_ids,txtb_index)
            # measure accuracy and record loss
            acc= accuracy(output, cls, topk=(1, ))[0]
            acc_list.append(acc.item())

    return sum(acc_list)/len(acc_list)


    
def main_worker(local_rank,ngpus_per_node,args):
    #args:config
    # pad_data_list,pad_mask_list,all_data_index,label_list=snli_data_reader(dev_path,0)
    #读取数据
    train_dataset=SNLI_Dataset(args['train_path'],args['label_dict'],args['src_pad_idx'])
    valid_dataset=SNLI_Dataset(args['dev_path'],args['label_dict'],args['src_pad_idx'])
    test_dataset=SNLI_Dataset(args['test_path'],args['label_dict'],args['src_pad_idx'])

    #args['n_position']=train_dataset.data_max_len()
    
    batch_size = int(args['batch_size'] / ngpus_per_node)
    #设置dalaloder
    TrainLoader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True,pin_memory=True)
    ValidLoader= DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    TestLoader= DataLoader(test_dataset,batch_size=args['batch_size'],shuffle=False,pin_memory=True)
    
    #配置网络和设置
    net=Model(args)

    if torch.cuda.is_available() and args['use_gpu']:
        #使用GPU多卡训练
        #1.初始化
        torch.distributed.init_process_group(backend="nccl")
        # 2）GPU可用则配置每个进程的gpu
        # local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda",local_rank)

        # 3）使用DistributedSampler
        TrainLoader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,sampler=DistributedSampler(train_dataset))
        
        # 4) 封装之前要把模型移到对应的gpu
        net.to(device)
        if torch.cuda.device_count() > 1:
            print('Total:use', torch.cuda.device_count(), "GPUs!")
            # 5) 封装
            net = torch.nn.parallel.DistributedDataParallel(net,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
    else:
        #使用cpu
        device = torch.device("cpu")
        net.to(device)
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    

    optimizer = ScheduledOptim(
        torch.optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args['lr'], args['d_model'], args['n_warmup_steps'])

    criterion = nn.NLLLoss()
    #开始训练epoch轮
    best_acc = 0
    iterator = tqdm(range(args['epoch']))
    logger.info('****Start Train')
    train_info={
        'train_accs':[],
        'train_loss':[],
        'valid_accs':[],
        'valid_loss':[],
    }
    
    for epoch in iterator:
        if args['use_gpu']:
        # DDP：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
            TrainLoader.sampler.set_epoch(epoch)
        train_acc,train_loss = train(TrainLoader,net,criterion,optimizer,device)
        logger.debug('Train,epoch:%s,acc:%s,loss:%s',epoch,train_acc,train_loss)
        train_info['train_accs'].append(train_acc)
        train_info['train_loss'].append(train_loss)
        
        valid_acc,valid_loss = valid(ValidLoader,net,criterion,device)
        logger.debug('Valid,epoch:%s,acc:%s,loss:%s',epoch,valid_acc,valid_loss)
        train_info['valid_accs'].append(valid_acc)
        train_info['valid_loss'].append(valid_loss)
        
        # DDP:
        # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
        #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
        # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
        #请设置为只输出rank=0的进程,logging请设置为只输出rank=0的进程
        
        if args['use_gpu']:
            if torch.distributed.get_rank() == 0:
                logger.info('Train,epoch:%s,acc:%s,loss:%s',epoch,train_acc,train_loss)
                logger.info('Valid,epoch:%s,acc:%s,loss:%s',epoch,valid_acc,valid_loss)
                state={
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_acc': train_acc,
                'train_loss':train_loss,
                'valid_acc':valid_acc,
                'valid_loss':valid_loss,
                }
                filepath="model_ckpt_epoch_%s.pth"%(epoch)
                torch.save(state, filepath)         
                
                if best_acc<valid_acc:
                    best_acc=valid_acc

                    filepath=args['bets_model_path']
                    torch.save(state, filepath)
                    
                    logger.info('Best ACC：%s; model saved!',best_acc)
                    logger.info(filepath)
        else:
                state={
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_acc': train_acc,
                'train_loss':train_loss,
                'valid_acc':valid_acc,
                'valid_loss':valid_loss,
                }
                filepath="model_ckpt_epoch_%s.pth"%(epoch)
                torch.save(state, filepath)

                if best_acc<valid_acc:
                    best_acc=valid_acc

                    filepath=args['bets_model_path']
                    torch.save(state, filepath)
                    
                    logger.info('Best ACC：%s; model saved!',best_acc)
                    logger.info(filepath)
    #训练结束,进行一次测试
    
    test_acc = test(TestLoader,net,args['bets_model_path'])
    logger.info('Test:ACC:%s',test_acc)
    logger.info(train_info)
    logger.info('****END Train')

def main(args):
    ipt_args = parser.parse_args()
    main_worker(ipt_args.local_rank, 4, args)


# 新增2：从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数，后面还会介绍。所以不用考虑太多，照着抄就是了。
#       argparse是python的一个系统库，用来处理命令行调用，如果不熟悉，可以稍微百度一下，很简单！
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

snli_train_config={
    'train_path':'data/SNLI/snli_1.0_train.jsonl',
    'test_path':'data/SNLi/snli_1.0_test.jsonl',
    'dev_path':'data/SNLi/snli_1.0_dev.jsonl',
    'label_dict':{  "entailment": 0,
                    "neutral": 1,
                    "contradiction": 2},
    #encoder_config
    'n_src_vocab':tokenizer.vocab_size,
    'src_pad_idx':0,
    'd_word_vec':512,
    'd_model':512,
    'd_inner':2048,
    'n_layers':2,
    'n_head':8,
    'd_k':64,
    'd_v':64,
    'dropout':0.1,
    'n_position':512,
    'class_num':3,
    #
    'use_gpu':True,
    'epoch':100,
    'batch_size':512,
    'lr':0.1,
    'n_warmup_steps':300,
    #
    'bets_model_path':'bets_model.pth'
    
}
if __name__ == '__main__':
    main(snli_train_config)


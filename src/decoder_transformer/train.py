import torch
from transformer_model import LanguageModel,block_size

from data_prep import get_batch,decode


max_steps=5000      
eval_interval=500
eval_iters=300
device='cuda' if torch.cuda.is_available() else 'cpu'
batch_size=32

#for optimizer
lr=3e-4
betas=(0.9, 0.999)
wt_decay=0.1

#loss _calc

@torch.no_grad()
def loss_estimation():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb=get_batch(split,block_size,batch_size,device)
            _,loss=model(xb,yb)
            losses[k]=loss
        out[split]=losses.mean()
    model.train()
    return out


model=LanguageModel()
#defining optimizer
optimizer= model.config_optimizer(wt_decay, lr, betas)

#model training
for iters in range(max_steps):

    if iters % eval_interval == 0:
        losses=loss_estimation()
        print(f"step{iters}  train_loss {losses['train']:.5f} and validation_loss is {losses['val']:.5f}")

    xb,yb=get_batch('train',block_size,batch_size,device)
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



import time

t0=time.time()

print('First Generation ---->')
print(decode(model.generate(torch.zeros((1,1),dtype=torch.long),max_new_tokens=1000)[0]))

print('time taken',time.time()-t0)
print('Second Generation ---->')
print(decode(model.generate(torch.zeros((2,1),dtype=torch.long),max_new_tokens=500)[1]))
'''
without training model when the weights are random, i inferred model to get output and following were 
observations regarding usage of KV cache
w/o KV cache
for 10 tokens time taken 0.07384586334228516
for 100 tokens time taken 0.12468314170837402
for 1000 tokens time taken 2.203747272491455

kv cache
for 10 tokens time taken 0.04920196533203125
for 100 tokens time taken 0.08963203430175781
for 1000 tokens time taken 1.2654902935028076

This clearly shows that implementations of KV was succesful and memory compensate inference time


If anyone interested in observing kv cache implementation, one should just comment out all the lines
except 1,2,34, 53-60 and comment out other lines and make use_cache =True in transformer_model.py
'''


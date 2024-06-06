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

print('First Generation ---->')
print(decode(model.generate(torch.zeros((2,1),dtype=torch.long),max_new_tokens=500)[0]))
print('Second Generation ---->')
print(decode(model.generate(torch.zeros((2,1),dtype=torch.long),max_new_tokens=500)[1]))

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

block_size=8
batch_size=32
max_steps=3000
eval_interval=300
lr=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_emb=32

torch.manual_seed(1334)
# reading input data
text=open('input.txt','r',encoding='utf-8').read()

#character vocabulary 
ch=sorted(set(text))
vocab_size=len(ch)

# mapping from characters to integers
stoi={s:i for i,s in enumerate(ch)}
itos={i:s for s,i in stoi.items()}

# encoder and decoder for tokens
encode= lambda s:torch.tensor(list(stoi[i] for i in s),dtype=torch.long)
decode=lambda i:(''.join(itos[s.item()] for s in i))

#encoding data
data=encode(text)

#train_val split
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#creating input batches
def get_batch(split):
    data=train_data if split =='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:block_size+i] for i in ix])
    y=torch.stack([data[i+1:block_size+i+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def loss_estimation():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb=get_batch(split)
            _,loss=model(xb,yb)
            losses[k]=loss
        out[split]=losses.mean()
    model.train()
    return out

#creating bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens_embedding_table=nn.Embedding(vocab_size,vocab_size)



    def forward(self,idx,target=None):
        logits=self.tokens_embedding_table(idx)#B,T,C(Batch,Time,Channel)

        if target is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            target=target.view(B*T)
            loss=F.cross_entropy(logits,target)
        
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            prob=F.softmax(logits,dim=1)
            idx_next=torch.multinomial(prob,num_samples=1)  
            idx=torch.cat((idx,idx_next),dim=1)
        return idx 

model=BigramLanguageModel()

model.to(device)
#defining optimizer
optimizer=torch.optim.AdamW(model.parameters(),lr)




#model training
for iters in range(max_steps):

    if iters % eval_interval == 0:
        losses=loss_estimation()
        print(f"step{iters}  train_loss {losses['train']:.5f} and validation_loss is {losses['val']:.5f}")

    xb,yb=get_batch('train')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#model output/generation
context=torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context,max_new_tokens=100)[0]))
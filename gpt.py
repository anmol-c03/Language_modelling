import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

block_size=128
batch_size=32
max_steps=5000
eval_interval=500
lr=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=300
n_emb=128
n_hidden=200
num_blocks=4
dropout=0.2
num_heads=4
#------------------------------------------------------------------------------------------------------------------------------
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

class  head(nn.Module):
    def __init__(self,n_emb,n_head):
        super().__init__()
        self.query=nn.Linear(n_emb,n_head,bias=False)
        self.key=nn.Linear(n_emb,n_head,bias=False)
        self.value=nn.Linear(n_emb,n_head,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x) #(B,T,nhead)
        k=self.key(x)   #(B,T,nhead)
        v=self.value(x)  #(B,T,nhead)
        wei=q @ k.transpose(-2,-1) *k.shape[-1]**-0.5#(B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,-1)
        wei=self.dropout(wei)
        self.out=wei @ v
        return self.out
        
class Multiplehead(nn.Module):
    def __init__(self,num_heads,head_dim) :
        super().__init__()
        self.heads=nn.ModuleList([head(n_emb,head_dim) for _ in range(num_heads)])
        self.proj=nn.Linear(n_emb,n_emb)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        self.out=torch.cat([h(x) for h in self.heads],-1)
        self.out=self.dropout(self.proj(self.out))
        return self.out

class FeedForward_NN(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.ffnn=nn.Sequential(
            nn.Linear(n_emb,4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb,n_emb),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        self.out=self.ffnn(x)
        return self.out

class Blocks(nn.Module):
    def __init__(self,n_emb,num_heads):
        super().__init__()
        heads_dim=n_emb//num_heads
        self.sa_head=Multiplehead(num_heads,heads_dim)
        self.feedforwd_nn=FeedForward_NN(n_emb)
        self.ln1=nn.Linear(n_emb,n_emb)
        self.ln2=nn.Linear(n_emb,n_emb)
    
    def forward(self,x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.feedforwd_nn(self.ln2(x))
        return x

#creating bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens_embedding_table=nn.Embedding(vocab_size,n_emb)
        self.position_embedding_table=nn.Embedding(block_size,n_emb)
        self.n_blocks=nn.Sequential(*[Blocks(n_emb,num_heads=4) for _ in range(num_blocks)])
        self.layer_norm=nn.LayerNorm(n_emb)
        self.prediction_layer=nn.Linear(n_emb,vocab_size)


    def forward(self,idx,target=None):
        B,T=idx.shape
        tokens_emb=self.tokens_embedding_table(idx)#B,T,C(Batch,Time,Channel)
        position_emb=self.position_embedding_table(torch.arange(T))#,device=device) # (T,C)
        x=tokens_emb+position_emb
        x=self.n_blocks(x)
        x=self.layer_norm(x)     # before prediction layer we have to add layer normalization
        logits=self.prediction_layer(x)
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
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
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
context=torch.zeros((2,1), dtype=torch.long, device=device)
print(decode(model.generate(context,max_new_tokens=500)[0]))
print('\n second generation')
print(decode(model.generate(context,max_new_tokens=500)[1]))



# step0  train_loss 4.26686 and validation_loss is 4.26062
# step500  train_loss 2.65473 and validation_loss is 2.66813
# step1000  train_loss 2.50300 and validation_loss is 2.51350
# step1500  train_loss 2.42030 and validation_loss is 2.43303
# step2000  train_loss 2.38351 and validation_loss is 2.40324
# step2500  train_loss 2.34365 and validation_loss is 2.35963
# step3000  train_loss 2.31878 and validation_loss is 2.34562
# step3500  train_loss 2.28596 and validation_loss is 2.31826
# step4000  train_loss 2.27834 and validation_loss is 2.30968
# step4500  train_loss 2.26485 and validation_loss is 2.28830

# step0  train_loss 4.61728 and validation_loss is 4.61589
# step500  train_loss 2.36120 and validation_loss is 2.39303
# step1000  train_loss 2.23126 and validation_loss is 2.26866
# step1500  train_loss 2.16700 and validation_loss is 2.20171
# step2000  train_loss 2.11482 and validation_loss is 2.17714
# step2500  train_loss 2.09351 and validation_loss is 2.15185
# step3000  train_loss 2.04239 and validation_loss is 2.13744
# step3500  train_loss 2.02750 and validation_loss is 2.11864
# step4000  train_loss 2.02627 and validation_loss is 2.10695
# step4500  train_loss 1.99220 and validation_loss is 2.08844



# step0  train_loss 4.19931 and validation_loss is 4.19753
# step500  train_loss 2.50559 and validation_loss is 2.50502
# step1000  train_loss 2.35493 and validation_loss is 2.37058
# step1500  train_loss 2.26379 and validation_loss is 2.29113
# step2000  train_loss 2.19128 and validation_loss is 2.23382
# step2500  train_loss 2.16184 and validation_loss is 2.19825
# step3000  train_loss 2.13765 and validation_loss is 2.17829
# step3500  train_loss 2.11278 and validation_loss is 2.17630
# step4000  train_loss 2.09589 and validation_loss is 2.16821
# step4500  train_loss 2.08342 and validation_loss is 2.15595
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

block_size=128
batch_size=3
eval_iters=300
device='cuda' if torch.cuda.is_available() else 'cpu'
n_emb=128
n_hidden=200
num_blocks=4
dropout=0.2
n_head=4
#for optimizer
lr=3e-4
betas=(0.9, 0.999)
wt_decay=0.1
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

class  self_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(n_emb,n_head,bias=False)
        self.key=nn.Linear(n_emb,n_head,bias=False)
        self.value=nn.Linear(n_emb,n_head,bias=False)
        self.c_proj = nn.Linear(n_emb, n_emb)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.register_buffer(
            "tril", 
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size), persistent=False)
        self.dropout=nn.Dropout(dropout)
        self.n_head=n_head

    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x) #(B,T,nhead)
        k=self.key(x)   #(B,T,nhead)
        v=self.value(x)  #(B,T,nhead)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))#(B,T,T)
        wei=wei.masked_fill(self.tril[:,:,:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,-1)
        wei=self.dropout(wei)
        y=wei @ v
        self.out = y.transpose(1, 2).contiguous().view(B, T, C) 
        return self.c_proj(self.out)
        

class FeedForward_NN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        self.out=self.dropout(x)
        return self.out

class Blocks(nn.Module):
    def __init__(self,n_emb,num_heads):
        super().__init__()
        heads_dim=n_emb//num_heads
        self.sa_head=self_attention()
        self.feedforwd_nn=FeedForward_NN(n_emb)
        self.ln1=nn.LayerNorm(n_emb,n_emb)
        self.ln2=nn.LayerNorm(n_emb,n_emb)
    
    def forward(self,x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.feedforwd_nn(self.ln2(x))
        return x

#creating language  model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens_embedding_table=nn.Embedding(vocab_size,n_emb)
        self.position_embedding_table=nn.Embedding(block_size,n_emb)
        self.n_blocks=nn.Sequential(*[Blocks(n_emb,num_heads=4) for _ in range(num_blocks)])
        self.layer_norm=nn.LayerNorm(n_emb)
        self.prediction_layer=nn.Linear(n_emb,vocab_size)
        #using the concept of weight tying to reduce total number of params
        self.tokens_embedding_table.weight=self.prediction_layer.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blocks))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
            Return the number of parameters in the model.
            in this case we have implemented absolute positional embedding which are non_trainable
            so we subtract them from total no of parameters 
            and same goes for  rotary embedding Since RoPE uses a fixed mathematical formula to compute positional embeddings,
            there are no parameters to be learned during training.
            The position of each token is encoded by applying rotations based on these predefined functions. 
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_emdedding_table.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def config_optimizer(self,weight_decay,lr,betas):
        params={pn:p for pn,p in self.named_parameters()}
        params={pn:p for pn,p in params.items() if p.requires_grad}
        decayed_points={p for pn,p in params.items() if p.dim()>=2}
        non_decayed_points={p for pn,p in params.items() if p.dim()<2}
        optimizer_groups=[
            {'params':decayed_points,'weight_decay':weight_decay},
            {'params':non_decayed_points,'weight_decay':0}
        ]
        num_decay_params=sum(p.numel() for p in decayed_points)
        num_nodecay_params=sum(p.numel() for p in non_decayed_points)
        print(f'there are {len(decayed_points)} decayed_params with total num of parameters = {num_decay_params}')
        print(f'there are {len(non_decayed_points)} non_decayed_params with total num of parameters = {num_nodecay_params}')
        optimizer=torch.optim.AdamW(optimizer_groups,betas=betas,lr=lr)
        return optimizer

    def forward(self,idx,target=None):
        B,T=idx.shape
        tokens_emb=self.tokens_embedding_table(idx)#B,T,C(Batch,Time,Channel)
        position_emb=self.position_embedding_table(torch.arange(T))#,device=device) # (T,C)
        #this is an implementation of absolute positional encoding
        #if long term delay required , one should prefer rotary positional embedding
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
    
    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            prob=F.softmax(logits,dim=1)
            idx_next=torch.multinomial(prob,num_samples=1)  
            idx=torch.cat((idx,idx_next),dim=1)
        return idx 

model=LanguageModel()
#defining optimizer
optimizer= model.config_optimizer(wt_decay, lr, betas)


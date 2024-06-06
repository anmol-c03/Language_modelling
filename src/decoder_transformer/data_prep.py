import torch

# reading input data
text=open('/Users/anmolchalise/Desktop/language_modelling/Language_models/gpt_2/input.txt','r',encoding='utf-8').read()

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
def get_batch(split,block_size,batch_size,device):
    data=train_data if split =='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:block_size+i] for i in ix])
    y=torch.stack([data[i+1:block_size+i+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y
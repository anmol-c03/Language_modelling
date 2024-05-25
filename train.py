import torch
from transformer_model import model,loss_estimation,get_batch,optimizer



max_steps=5000      
eval_interval=500
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

import torch
from torch import nn

class QuarterSelfAttention(nn.Module):
 def __init__(self):
  super().__init__()
  self.project=nn.Linear(384,64)
  self.position=nn.Embedding(256,64)
  layer=nn.TransformerEncoderLayer(64,2,256,.1,batch_first=True,norm_first=True,activation='gelu')
  self.encoder=nn.TransformerEncoder(layer,1,norm=nn.LayerNorm(64),enable_nested_tensor=False)
  self.dropout=nn.Dropout(.1)
  self.head=nn.Linear(64,3007)
 def forward(self,x,lengths):
  n,t,_=x.shape;assert t<=256
  pos=torch.arange(t,device=x.device)
  pad=pos[None,:]>=lengths.to(x.device)[:,None]
  causal=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
  z=self.dropout(self.project(x)+self.position(pos)[None,:,:])
  z=self.encoder(z,mask=causal,src_key_padding_mask=pad)
  return self.head(z[torch.arange(n,device=x.device),lengths.to(x.device)-1])

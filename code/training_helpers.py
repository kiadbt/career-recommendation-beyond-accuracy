"""CPU-safe task helpers; no data loading or device allocation on import."""
import ast,hashlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_source_classes(source_path):
 source=Path(source_path).read_text(encoding='utf-8-sig');tree=ast.parse(source)
 nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'MLPTransformation','CareerBiLSTM'}]
 assert len(nodes)==2;ns={'torch':torch,'nn':nn,'F':F}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),ns)
 return ns['MLPTransformation'],ns['CareerBiLSTM']

def pool_history(x,lengths):
 """0.8-decayed quarter mean with correct padding and normalization."""
 pos=torch.arange(x.shape[1],device=x.device)[None,:];mask=pos<lengths[:,None]
 weight=torch.pow(.8,(lengths[:,None]-1-pos).clamp(min=0))*mask
 mean=(x*weight.unsqueeze(-1)).sum(1)/weight.sum(1,keepdim=True)
 return F.normalize(mean,dim=1)

def multi_target_loss(logits,members,sizes,prefix_weights,global_weight_scale):
 """Unbiased resume-balanced loss under uniform training-query sampling."""
 assert logits.ndim==2 and members.ndim==2 and bool((sizes>0).all())
 mask=torch.arange(members.shape[1],device=members.device)[None,:]<sizes[:,None]
 values=-F.log_softmax(logits,dim=1).gather(1,members.clamp(min=0))
 per_query=(values*mask).sum(1)/sizes
 return (per_query*prefix_weights*global_weight_scale).mean()

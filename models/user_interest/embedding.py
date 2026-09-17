"""Large-vocabulary embeddings for TRM-UI.

The item embedding uses sparse=True. Its weight matrix is still dense in memory,
but backward produces sparse gradients. The trainer therefore uses sparse SGD
for the item table and AdamW for the dense category/TRM parameters.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
class UserInterestEmbedding(nn.Module):
    def __init__(self,num_items:int,num_categories:int,item_embedding_dim:int=128,category_embedding_dim:int=128,padding_idx:int=0,sparse_item_embedding:bool=True,device:torch.device|str|None=None)->None:
        super().__init__()
        if num_items<=padding_idx: raise ValueError(f'num_items must be greater than padding_idx ({padding_idx}), got {num_items}')
        if num_categories<=padding_idx: raise ValueError(f'num_categories must be greater than padding_idx ({padding_idx}), got {num_categories}')
        self.num_items=num_items; self.num_categories=num_categories; self.item_embedding_dim=item_embedding_dim; self.category_embedding_dim=category_embedding_dim; self.padding_idx=padding_idx; self.sparse_item_embedding=sparse_item_embedding
        self.item_embedding=nn.Embedding(num_items,item_embedding_dim,padding_idx=padding_idx,sparse=sparse_item_embedding,device=device)
        self.category_embedding=nn.Embedding(num_categories,category_embedding_dim,padding_idx=padding_idx,sparse=False,device=device)
        with torch.no_grad(): self.item_embedding.weight[padding_idx].zero_(); self.category_embedding.weight[padding_idx].zero_()
    @property
    def output_dim(self)->int: return self.item_embedding_dim+self.category_embedding_dim
    def encode_history(self,item_ids,category_ids):
        self._validate_history_inputs(item_ids,category_ids)
        return torch.cat([self.item_embedding(item_ids),self.category_embedding(category_ids)],dim=-1)
    def encode_target(self,item_ids,category_ids):
        self._validate_target_inputs(item_ids,category_ids)
        return torch.cat([self.item_embedding(item_ids),self.category_embedding(category_ids)],dim=-1)
    def forward(self,history_items,history_categories,target_item,target_category)->Dict[str,torch.Tensor]:
        return {'history':self.encode_history(history_items,history_categories),'target':self.encode_target(target_item,target_category)}
    def _validate_history_inputs(self,item_ids,category_ids):
        if item_ids.ndim!=2 or category_ids.ndim!=2: raise ValueError('history item/category IDs must have shape [batch, sequence_length]')
        if item_ids.shape!=category_ids.shape: raise ValueError(f'history item/category tensors must have identical shapes, got {tuple(item_ids.shape)} and {tuple(category_ids.shape)}')
        self._validate_id_range(item_ids,self.num_items,'history item'); self._validate_id_range(category_ids,self.num_categories,'history category')
    def _validate_target_inputs(self,item_ids,category_ids):
        if item_ids.ndim!=1 or category_ids.ndim!=1: raise ValueError('target item/category IDs must have shape [batch]')
        if item_ids.shape!=category_ids.shape: raise ValueError(f'target item/category tensors must have identical shapes, got {tuple(item_ids.shape)} and {tuple(category_ids.shape)}')
        self._validate_id_range(item_ids,self.num_items,'target item'); self._validate_id_range(category_ids,self.num_categories,'target category')
    @staticmethod
    def _validate_id_range(ids,vocabulary_size,name):
        if ids.numel()==0: return
        minimum=int(ids.min().item()); maximum=int(ids.max().item())
        if minimum<0: raise ValueError(f'{name} IDs cannot be negative. Found minimum ID: {minimum}')
        if maximum>=vocabulary_size: raise ValueError(f'{name} ID {maximum} is outside configured vocabulary size {vocabulary_size}. Valid IDs are 0..{vocabulary_size-1}.')
if __name__=='__main__':
    m=UserInterestEmbedding(10000,1000,sparse_item_embedding=True)
    h=torch.randint(1,10000,(2,50)); c=torch.randint(1,1000,(2,50)); t=torch.randint(1,10000,(2,)); tc=torch.randint(1,1000,(2,)); o=m(h,c,t,tc)
    print('history',o['history'].shape,'target',o['target'].shape,'sparse',m.item_embedding.sparse)

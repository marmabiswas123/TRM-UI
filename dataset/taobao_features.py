"""
TAOBAO-MM feature vocabulary and ID remapping.

0 = PAD; 1..N = known IDs; N+1 = UNK.
An optional max_size prevents a builder from growing beyond its configured cap.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict,List,Sequence
import torch
PAD_INDEX=0
class IDMapper:
    def __init__(self,max_size:int|None=None)->None:
        if max_size is not None and max_size<1: raise ValueError(f'max_size must be >= 1, got {max_size}')
        self.max_size=max_size; self._raw_to_index={}; self._index_to_raw={}
    def add(self,raw_id:int)->int:
        raw_id=int(raw_id)
        if raw_id==0: return PAD_INDEX
        if raw_id not in self._raw_to_index:
            if self.max_size is not None and len(self._raw_to_index)>=self.max_size: return self.unk_index
            idx=len(self._raw_to_index)+1; self._raw_to_index[raw_id]=idx; self._index_to_raw[idx]=raw_id
        return self._raw_to_index[raw_id]
    def encode(self,raw_id:int)->int:
        raw_id=int(raw_id)
        return PAD_INDEX if raw_id==0 else self._raw_to_index.get(raw_id,self.unk_index)
    def encode_sequence(self,ids:Sequence[int])->List[int]: return [self.encode(x) for x in ids]
    def decode(self,index:int)->int:
        index=int(index)
        if index in (PAD_INDEX,self.unk_index): return 0
        if index not in self._index_to_raw: raise KeyError(f'Unknown embedding index: {index}')
        return self._index_to_raw[index]
    @property
    def vocab_size(self)->int: return len(self._raw_to_index)+2
    @property
    def unk_index(self)->int: return len(self._raw_to_index)+1
    def __len__(self)->int: return len(self._raw_to_index)
    def state_dict(self)->Dict:
        state={'raw_to_index':{str(k):v for k,v in self._raw_to_index.items()},'index_to_raw':{str(k):v for k,v in self._index_to_raw.items()}}
        if self.max_size is not None: state['max_size']=self.max_size
        return state
    def load_state_dict(self,state:Dict)->None:
        self.max_size=state.get('max_size',None)
        self._raw_to_index={int(k):int(v) for k,v in state['raw_to_index'].items()}
        self._index_to_raw={int(k):int(v) for k,v in state['index_to_raw'].items()}
def save_mapper(mapper:IDMapper,path:str|Path)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump(mapper.state_dict(),f)
def load_mapper(path:str|Path)->IDMapper:
    with open(Path(path),'r') as f: state=json.load(f)
    m=IDMapper(); m.load_state_dict(state); return m
def build_sample_vocabularies(sample):
    im=IDMapper(); cm=IDMapper()
    for x in sample['history_items'].tolist(): im.add(x)
    for x in sample['history_categories'].tolist(): cm.add(x)
    im.add(int(sample['target_item'].item())); cm.add(int(sample['target_category'].item()))
    return im,cm
def remap_sample(sample,item_mapper:IDMapper,category_mapper:IDMapper):
    return {'history_items':torch.tensor(item_mapper.encode_sequence(sample['history_items'].tolist()),dtype=torch.long),
            'history_categories':torch.tensor(category_mapper.encode_sequence(sample['history_categories'].tolist()),dtype=torch.long),
            'target_item':torch.tensor(item_mapper.encode(int(sample['target_item'].item())),dtype=torch.long),
            'target_category':torch.tensor(category_mapper.encode(int(sample['target_category'].item())),dtype=torch.long),
            'label':sample['label'].clone()}

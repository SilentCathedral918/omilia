import torch.nn as nn
import sentencepiece as spm

class Tokeniser(nn.Module):
  def __init__(
    self, 
    model_path='lexatomos/anglikis/anglikis.model', 
  ):
    super().__init__()

    self.model_path = model_path
    
    self.sp = spm.SentencePieceProcessor()
    self.sp.load(model_path)

  def encode(self, prompt: str, add_bos=True, add_eos=True):
    ids_ = self.sp.encode(prompt, out_type=int, add_bos=add_bos, add_eos=add_eos)
    labels_ = self.sp.encode(prompt, out_type=str, add_bos=add_bos, add_eos=add_eos)
    
    return {
      'ids': ids_,
      'labels': labels_
    }
  
  def decode(self, token_ids):
    return self.sp.piece_to_id(token_ids) if isinstance(token_ids, str) else self.sp.decode(token_ids)
  
  def forward(self, prompt: str, add_bos=True, add_eos=True):
    return self.encode(prompt, add_bos, add_eos)

  def is_known_token(self, token: str):
    return self.sp.piece_to_id(token) != self.sp.unk_id()

  def vocab_size(self):
    return self.sp.get_piece_size()

  def token_id(self, token: str):
    return self.sp.piece_to_id(token)

  def unk_id(self):
    return self.sp.unk_id()
  
  def bos_id(self):
    return self.sp.bos_id()
  
  def eos_id(self):
    return self.sp.eos_id()
  
  def pad_id(self):
    return self.sp.pad_id()


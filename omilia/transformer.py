import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
  def __init__(self, n_dims, n_heads, max_len, bias=False, dropout=0.2):
    """
      n_dims: number of embedding dimensions \n
      n_heads: number of attention heads \n
      max_len: max length of input sequences (in tokens) \n
      bias: whether or not to use bias in linear layers \n
      dropout: probability of dropout
    """

    super().__init__()
    assert n_dims % n_heads == 0, "Error: number of embedding dimensions must be divisible by number of heads."

    self.n_dims = n_dims
    self.n_heads = n_heads
    
    # dropouts: regularisation technique to prevent overfitting
    # NOTE: 
    #   1. for regularisation, it introduces penalty to the model's complexity
    #   2. during training, these layers deactivates a percentage of "neurons" in given layer
    #     the deactivated "neurons" are removed from network's computations in that training iteration
    #   3. essentially encourage the network to learn more on features that work well on unseen data
    #     since the network is practically prevented from relying too heavily on specific "neurons"  
    self.dropout_attn = nn.Dropout(dropout)
    self.dropout_proj = nn.Dropout(dropout)

    # Query, Key, Value projections for all heads, in batch for computation efficiency
    # NOTE: out_features = n_dim * 3 since the batch includes Query, Key, and Value
    self.c_attn = nn.Linear(n_dims, n_dims * 3, bias)

    # linear projection of concatenated attention-head outputs
    self.c_proj = nn.Linear(n_dims, n_dims, bias)

    # causal-mask: ensure attention is only applied to the left of a given token in the sequence
    self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

  def forward(self, x):
    # batch_size_ = batch size
    # seq_len_ = sequence length (in tokens) 
    batch_size_, seq_len_, _ = x.size()

    # compute Query, Key, Value vectors for all heads
    # then split the output into separate Query, Key, Value tensors 
    query_, key_, value_ = self.c_attn(x).split(self.n_dims, dim=2)

    # reshape tensors into sequences of smaller token vectors for each head
    # n_dims -> n_dims // n_heads
    # transposing converts [batch_size_, seq_len_, n_heads, n_dims // n_heads] to [batch_size_, n_heads, seq_len_, n_dims // n_heads]
    #   ensuring computation for attention per head (Q @ K.T) across dimension d_k 
    query_ = query_.view(batch_size_, seq_len_, self.n_heads, self.n_dims // self.n_heads).transpose(1, 2)
    key_ = key_.view(batch_size_, seq_len_, self.n_heads, self.n_dims // self.n_heads).transpose(1, 2)
    value_ = value_.view(batch_size_, seq_len_, self.n_heads, self.n_dims // self.n_heads).transpose(1, 2)

    # compute attention-matrix
    # (Causal) Attention(Q, K, V) = softmax(masked((Q @ K.T) / sqrt(d_k))) @ V
    
    # 1. (Q @ K.T) / sqrt(d_k)
    atten_ = (query_ @ key_.transpose(-2, -1)) * (1.0 / math.sqrt(key_.size(-1)))

    # 2. perform masking -> masked((Q @ K.T) / sqrt(d_k))
    atten_ = atten_.masked_fill(self.causal_mask[:, :, :seq_len_, :seq_len_] == 0, float('-inf'))

    # 3. apply softmax -> softmax(masked((Q @ K.T) / sqrt(d_k)))
    atten_ = F.softmax(atten_, dim=-1)

    # 4. apply attention dropout
    atten_ = self.dropout_attn(atten_)

    # 5. perform mat-mul with Value vectors to get output vectors -> softmax(masked((Q @ K.T) / sqrt(d_k))) @ V
    y_ = atten_ @ value_

    # concatenate outputs from each attention head
    y_ = y_.transpose(1, 2).contiguous().view(batch_size_, seq_len_, self.n_dims)

    # linearly project
    y_ = self.dropout_proj(self.c_proj(y_))
    
    return y_

class FFNN(nn.Module):
  def __init__(self, n_dims, bias=False, dropout=0.2):
    """
      n_dims: number of embedding dimensions \n
      bias: whether or not to use bias in linear layers \n
      dropout: probability of dropout
    """
    
    super().__init__()

    self.layer_in = nn.Linear(n_dims, n_dims * 4, bias)
    self.layer_hid = nn.GELU()
    self.layer_out = nn.Linear(n_dims * 4, n_dims, bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # 1. we expand the dimensionality (typically by 4) 
    # -> gives the model more space to represent richer features, allowing more combinations/features to emeger
    # NOTE: think of it like "deeper thinking" per token
    y_ = self.layer_in(x) # [batch_size, seq_len, n_dims * 4]
    
    # 2. we apply non-linearity here; this is the activation layer, also hidden layer
    # -> introduces complexity and non-linear mapping, applies sophisticated rules
    # NOTE: think of it like apply deep thoughts and non-linear reasoning
    y_ = self.layer_hid(y_)
    
    # 3. we compress the dimensionality back to original embedding size (by the same expanding factor)
    # -> keeps the final shape compatible with the residual path, also preventing parameter explosion
    y_ = self.layer_out(y_) # [batch_size, seq_len, n_dims]
    
    # 4. apply dropout to enforce regularisation
    y_ = self.dropout(y_)
    return y_

class TransformerBlock(nn.Module):
  def __init__(self, n_dims, n_heads, max_len, bias=False, dropout=0.2):
    """
      n_dims: number of embedding dimensions \n
      n_heads: number of attention heads \n
      max_len: max length of input sequences (in tokens) \n
      bias: whether or not to use bias in linear layers \n
      dropout: probability of dropout
    """

    super().__init__()
    self.ln_atten = nn.LayerNorm(n_dims)
    self.atten = CausalSelfAttention(n_dims, n_heads, max_len, bias, dropout)
    self.ln_ffnn = nn.LayerNorm(n_dims)
    self.ffnn = FFNN(n_dims, bias, dropout)

  def forward(self, x):
    #   1. we take the input vectors and send it through Layer Normalisation 
    #   2. then we take the normalised input vectors through Mask Multi-Head Attention
    #   3. we got the output from the Attention, we ADD it to the unnormalised Input vectors
    y_ = x + self.atten(self.ln_atten(x))
    
    #   4. we take the ADDED output and send it through another Layer Normalisation
    #   5. then take it through the point-wise Feed-Forward Neural-Network
    #   6. we got the output from the FFNN, we ADD it to the ADDED output 
    y_ = y_ + self.ffnn(self.ln_ffnn(y_))
    
    #   7. we got the final output
    return y_

class Transformer(nn.Module):
  def __init__(self, n_dims=256, n_heads=8, max_len=1024, vocab_size=48000, n_layers=6, bias=False, dropout=0.2):
    """
      n_dims: number of embedding dimensions \n
      n_heads: number of attention heads \n
      max_len: max length of input sequences (in tokens) \n
      vocab_size: size of token vocabulary \n
      n_layers: number of transformer blocks \n
      bias: whether or not to use bias in linear layers \n
      dropout: probability of dropout
    """
    super().__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    self.transformer = nn.ModuleDict(dict(
      token_embed = nn.Embedding(vocab_size, n_dims),
      pos_embed = nn.Embedding(max_len, n_dims),
      dropout = nn.Dropout(dropout),
      blocks = nn.ModuleList([TransformerBlock(n_dims, n_heads, max_len, bias, dropout) for _ in range(n_layers)]),
      ln_final = nn.LayerNorm(n_dims),
      head = nn.Linear(n_dims, vocab_size, bias)
    ))

  def forward(self, token_ids, targets=None, pad_id=-1):
    """
      token_ids: tensor of shape [batch_size, seq_len], matrix of token indices \n
      targets: tensor of shape [batch_size, seq_len], matrix of next (target) token indices \n
    """
    
    # here, we grab the sequence length of given token ids ... 
    _, seq_len_ = token_ids.size()
    # ... to generate a list of token positions (0, 1, 2, ..., seq_len - 1)
    pos_ = torch.arange(0, seq_len_, dtype=torch.long, device=self.device)

    # generate token and position embeddings
    token_embed_ = self.transformer.token_embed(token_ids) # [batch_size, seq_len, n_dims]
    pos_embed_ = self.transformer.pos_embed(pos_) # [batch_size, seq_len]
    x_ = self.transformer.dropout(token_embed_ + pos_embed_)

    # pass the embeddings through transformer blocks 
    for block_ in self.transformer.blocks:
      x_ = block_(x_)
    x_ = self.transformer.ln_final(x_)

    # if 'targets' is detected, that means the user signaled training
    if targets is not None:
      # compute logits for the full sequence
      logits_ = self.transformer.head(x_)
      
      # loss would be computed
      loss_ = F.cross_entropy(
        logits_.view(-1, logits_.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id
      )

    # otherwise, for inference
    else:
      # only compute logits for the last token
      logits_ = self.transformer.head(x_[:, [-1], :])
      
      # no loss computation
      loss_ = None

    return logits_, loss_


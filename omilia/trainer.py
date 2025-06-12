import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler

class Trainer:
  def __init__(
    self, 
    model: nn.Module, 
    tokeniser: nn.Module, 
    train_dataset, 
    eval_dataset, 
    optimiser=None,
    n_epochs=5, 
    lr=5e-5,
    batch_size=4
  ):
    """
      model: transformer model to train \n
      tokeniser: tokeniser to be used \n
      train_dataset: training dataset, expected format: [] \n
      eval_dataset: evaluation dataset, expected format: [] \n
      optimiser: if `None`, the trainer would use AdamW as default optimiser \n
      n_epochs: number of total training epochs \n
      lr: learning rate of training \n
      batch_size: number of training/evaluating samples per batch to load
    """

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model.to(self.device)
    self.tokeniser = tokeniser
    self.optimiser = optimiser or torch.optim.AdamW(model.parameters(), lr)
    self.n_epochs = n_epochs

    self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=self._collate_fn)
    self.eval_dataloader = DataLoader(eval_dataset, batch_size, shuffle=True, collate_fn=self._collate_fn)

    self.lr_scheduler = get_scheduler(
      name='linear', 
      optimizer=self.optimiser,
      num_warmup_steps=0,
      num_training_steps=n_epochs * len(self.train_dataloader)
    )

  def _collate_fn(self, batch) -> torch.Tensor:
    token_ids_ = [torch.tensor(self.tokeniser.encode(text_)['ids']) for text_ in batch]
    token_ids_ = pad_sequence(token_ids_, padding_value=self.tokeniser.pad_id(), batch_first=True)
    return token_ids_

  def train(self, best_model_save_path='model/omilia.pt'):
    from tqdm.autonotebook import trange

    best_loss_ = float('inf')
    n_training_steps_ = self.n_epochs * len(self.train_dataloader)
    progress_bar_ = trange(n_training_steps_)

    self.model.train()

    for epoch_ in range(self.n_epochs):
      total_loss_ = 0
      
      progress_bar_.set_description(f'Epoch {epoch_ + 1}/{self.n_epochs}')

      for batch_ in self.train_dataloader:
        token_ids_ = batch_.to(self.device)

        inputs_ = token_ids_[:, :-1]
        targets_ = token_ids_[:, 1:]

        _, loss_ = self.model(inputs_, targets_, self.tokeniser.pad_id())

        loss_.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()
        self.lr_scheduler.step()
        self.optimiser.zero_grad()        

        progress_bar_.update(1)
        progress_bar_.set_postfix({'loss': loss_.item()})

        total_loss_ += loss_.item()

      avg_loss_ = total_loss_ / len(self.train_dataloader)
      print(f'Epoch {epoch_ + 1}/{self.n_epochs} _ Average Loss: {avg_loss_:.4f}')

      with torch.no_grad():
        eval_loss_ = self.evaluate()
      
      if eval_loss_ < best_loss_:
        best_loss_ = eval_loss_
        torch.save(self.model.state_dict(), best_model_save_path)

        print(f"New best model saved to '{best_model_save_path}', eval_loss: {eval_loss_:.4f}")

  @torch.no_grad()
  def evaluate(self):
    from tqdm.autonotebook import trange

    self.model.eval()

    progress_bar_ = trange(len(self.eval_dataloader))
    progress_bar_.set_description(f'Evaluation Progress')

    total_loss_ = 0

    for batch_ in self.eval_dataloader:
      token_ids_ = batch_.to(self.device)

      inputs_ = token_ids_[:, :-1]
      targets_ = token_ids_[:, 1:]

      _, loss_ = self.model(inputs_, targets_, self.tokeniser.pad_id())

      progress_bar_.update(1)
      progress_bar_.set_postfix({'loss': loss_.item()})

      total_loss_ += loss_.item()

    avg_loss_ = total_loss_ / len(self.eval_dataloader)
    print(f'Evaluation _ Average Loss: {avg_loss_:.4f}')

    return avg_loss_

  def view_training_data(self):
    train_batches_ = [batch_ for batch_ in self.train_dataloader]
    eval_batches_ = [batch_ for batch_ in self.eval_dataloader]
    
    return {
        'train_batch': train_batches_,
        'eval_batch': eval_batches_
    }
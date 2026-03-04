from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


class DatatroveSFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_samples: int,
        strategy,
        input_template=None,
        pretrain_mode=True,  # Add pretrain_mode parameter with default True
        num_processors=8,  # Specify the number of processors you want to use
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.max_samples = max_samples
        self.multiturn = multiturn
        
        # Limit dataset size if needed
        if len(dataset) > self.max_samples:
            dataset = torch.utils.data.Subset(dataset, range(self.max_samples))

        
        self.prompts = dataset
        self.responses = [None] * len(self.prompts)
        
      
        # For pretrain mode, set to 0 (entire sequence is used for loss)
        self.prompt_ids_lens = [0] * len(self.prompts)
        
        self.response_ranges = None
 

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):

        input_ids = torch.tensor(self.prompts[idx]['input_ids']).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        loss_mask = self.get_loss_mask(input_ids, idx)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask


    def get_loss_mask(self, input_ids, idx):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            for start_idx, end_idx in response_ranges:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks

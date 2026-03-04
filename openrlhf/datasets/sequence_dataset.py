from typing import Callable, Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset, Subset

 
class SequenceDataset(Dataset):
    """
    Simplified dataset for parallel generation.
    Just returns full sequences - block structure is handled during generation.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_samples: int,
    ) -> None:
        super().__init__()

        self.dataset = dataset if len(dataset) <= max_samples else Subset(dataset, range(max_samples))
        self.tokenizer = tokenizer
        
        # Store sequences
        self._sequences = []
        for i in range(len(self.dataset)):
            token_ids = self.dataset[i]["input_ids"]
            self._sequences.append(token_ids[:-1])
             
    def __len__(self) -> int:
        return len(self._sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return   self._sequences[idx]#[:16]
    
    def collate_fn(self, batch):
        # Converts list of list of 0-D tensors to list of list of integers
        return batch
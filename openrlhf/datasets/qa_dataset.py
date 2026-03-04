from torch.utils.data import Dataset
from tqdm import tqdm

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F



def preprocess_data(data, input_key, label_key) -> str:
    
    prompt = data[input_key]

    # for Reinforced Fine-tuning
    label = data[label_key]
    return prompt, label

def pack_to_fixed_chunks(
    texts: Optional[List[str]],
    tokenizer,
    seq_len: int = 1024,
    add_eos_between: bool = True,
    pad_last: bool = True,
    qa_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Concatenate-tokenize-pack texts into fixed-length chunks.
    Produces a list of (seq_len,) tensors. No padding except the final chunk.

    Returns:
        chunks:             List[Tensor] with shape (seq_len,)
        doc_id_chunks:      List[Tensor] with shape (seq_len,),
                            where doc_id_chunks[k][t] = doc index of token t
                            (0,1,2,...) and -1 for padding positions.
        answer_mask_chunks: List[Tensor] with shape (seq_len,),
                            1 for answer tokens, 0 otherwise (question/separators/padding).
    """
    # Separator (EOS) between docs
    sep = []
    if add_eos_between:
        if tokenizer.eos_token_id is not None:
            sep = [tokenizer.eos_token_id]
        else:
            raise ValueError("tokenizer must have eos_token_id to use as separator")

    # Pad id for final chunk
    pad_id = None
    if pad_last:
        if tokenizer.pad_token_id is not None:
            pad_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id
        else:
            raise ValueError("Need pad_token_id or eos_token_id for padding")

    # Build per-example token ids and corresponding answer masks
    if qa_pairs is not None:
        prompts = [p for p, _ in qa_pairs]
        answers = [a for _, a in qa_pairs]

        prompt_tok = tokenizer(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        answer_tok = tokenizer(
            answers,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        ids_per_ex: List[List[int]] = []
        ans_mask_per_ex: List[List[int]] = []
        for p_ids, a_ids in zip(prompt_tok, answer_tok):
            ids = list(p_ids) + list(a_ids)
            mask = [0] * len(p_ids) + [1] * len(a_ids)
            ids_per_ex.append(ids)
            ans_mask_per_ex.append(mask)
    else:
        # Fallback: treat entire text as question (mask all zeros)
        ids_per_ex = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        ans_mask_per_ex = [[0] * len(ids) for ids in ids_per_ex]

    # Flatten with separators AND parallel doc_ids and answer masks
    flat_ids: List[int] = []
    flat_doc_ids: List[int] = []
    flat_ans_mask: List[int] = []

    for doc_idx, ids in enumerate(ids_per_ex):
        # actual tokens
        flat_ids.extend(ids)
        flat_doc_ids.extend([doc_idx] * len(ids))
        flat_ans_mask.extend(ans_mask_per_ex[doc_idx])

        # separator tokens (not part of Q/A; mask=0)
        if sep and doc_idx != len(ids_per_ex) - 1:
            flat_ids.extend(sep)
            flat_doc_ids.extend([doc_idx] * len(sep))
            #flat_ans_mask.extend([1] * len(sep))
            flat_ans_mask.extend([0] * len(sep))

    if not flat_ids:
        return [], [], []

    stream = torch.tensor(flat_ids, dtype=torch.long)
    stream_doc = torch.tensor(flat_doc_ids, dtype=torch.long)
    stream_mask = torch.tensor(flat_ans_mask, dtype=torch.long)
    L = seq_len
    n_full = stream.numel() // L

    chunks: List[torch.Tensor] = []
    doc_id_chunks: List[torch.Tensor] = []
    answer_mask_chunks: List[torch.Tensor] = []

    if n_full:
        chunks.extend(stream[: n_full * L].view(n_full, L).unbind(0))
        doc_id_chunks.extend(stream_doc[: n_full * L].view(n_full, L).unbind(0))
        answer_mask_chunks.extend(stream_mask[: n_full * L].view(n_full, L).unbind(0))

    # remainder
    rem = stream[n_full * L :]
    rem_doc = stream_doc[n_full * L :]
    rem_mask = stream_mask[n_full * L :]

    if rem.numel():
        if rem.numel() < L and pad_last:
            # pad tokens
            rem = F.pad(rem, (0, L - rem.numel()), value=pad_id)
            # mark padding with doc_id = -1 and mask=0
            rem_doc = F.pad(rem_doc, (0, L - rem_doc.numel()), value=-1)
            rem_mask = F.pad(rem_mask, (0, L - rem_mask.numel()), value=0)
        chunks.append(rem)
        doc_id_chunks.append(rem_doc)
        answer_mask_chunks.append(rem_mask)

    return chunks, doc_id_chunks, answer_mask_chunks


class QADataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        max_samples,
        # prompt_max_len,
        separate_prompt_label: bool = False,
        seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.separate_prompt_label = separate_prompt_label

        # chat_template
        self.seq_len = seq_len
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
         

        self.prompts = []
        self.labels = []
        self.datasources = []
        self.sequences = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_key, label_key)
            # always keep prompts/labels
            self.prompts.append(prompt)
            self.labels.append(label)
            # Track datasource/direction when available (e.g., translation datasets often use "source")
            if hasattr(data, "get"):
                self.datasources.append(data.get("source", data.get("datasource", "default")))
            else:
                self.datasources.append("default")
            if not self.separate_prompt_label:
                # optional: raw concatenation for debugging/inspection
                self.sequences.append(prompt + label)
        
        if not self.separate_prompt_label and len(self.prompts) != 0:
            qa_pairs = list(zip(self.prompts, self.labels))
            # list of tensors. doc_ids is same shape as sequences. int corresponding to what doc it is from
            self.sequences, self.doc_ids, self.answer_masks = pack_to_fixed_chunks(
                texts=None,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                add_eos_between=True,
                pad_last=True,
                qa_pairs=qa_pairs,
            )
        if max_samples != -1:
            if self.separate_prompt_label:
                self.prompts = self.prompts[:max_samples]
                self.labels = self.labels[:max_samples]
                self.datasources = self.datasources[:max_samples]
            else:
                self.sequences = self.sequences[:max_samples]
                self.doc_ids = self.doc_ids[:max_samples]
                self.answer_masks = self.answer_masks[:max_samples]

    def __len__(self):
        if self.separate_prompt_label:
            length = len(self.prompts) 
        else:
            length = len(self.sequences)
        return length

    def __getitem__(self, idx):
        if self.separate_prompt_label:
            return {"prompt": self.prompts[idx], "label": self.labels[idx], "datasource": self.datasources[idx]}
        else:
            return self.sequences[idx], self.doc_ids[idx], self.answer_masks[idx]
        
    def collate_fn(self, batch):
        # Converts list of list of 0-D tensors to list of list of integers
        if self.separate_prompt_label:
            return batch
        else:
            #return torch.stack(batch, dim=0)
            return list(zip(*batch))  # sequences, doc_ids, answer_masks


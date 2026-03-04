from typing import Optional, List

import deepspeed
import torch
import torch.nn as nn
from typing import Optional, Union
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, BitsAndBytesConfig, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import torch.nn.functional as F
from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

logger = init_logger(__name__)


def get_comet_model_for_text_embedding(
    model_name_or_path: str,
    bf16=False,
    **kwargs,
) -> nn.Module:

    from comet import download_model, load_from_checkpoint
    model_path = download_model(model_name_or_path)
    model = load_from_checkpoint(model_path)

    # COMET uses PyTorch Lightning internally which has complex dtype handling
    # Keep it in float32 to avoid dtype mismatches with its internal components
    # Ensure all model parameters are in float32
    model = model.to(dtype=torch.float32)

    class RewardModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            input_sequences,
            gt_sequences,
            ct_sequences,
        ) -> List[float]:
            data = [
                {'src': ct_sequences[i],
                 'mt': input_sequences[i],
                 'ref': gt_sequences[i],}
                for i in range(len(input_sequences))
            ]

            # COMET model's predict method uses PyTorch Lightning internally
            # Ensure it runs in float32 to avoid dtype conflicts
            outputs = self.model.predict(
                data, batch_size=len(data)
            )['scores']

            return outputs

    reward_model = RewardModel(model)
    tokenizer = reward_model.model.encoder.tokenizer
    return reward_model, tokenizer

def get_llm_for_text_embedding(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    device_map=None,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.
    ...(docstring)...
    """
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
     
    # Check if it's a token classification model (PRM)
    is_token_classification = False
    if hasattr(config, 'architectures') and config.architectures:
        is_token_classification = any(
            "ForTokenClassification" in arch
            for arch in config.architectures
        )

    if is_token_classification:
        base_class = AutoModelForTokenClassification._model_mapping[type(config)]
    else:
        base_class = AutoModel._model_mapping[type(config)]

    # FIX: Use base_class directly instead of base_class.__base__
    cls_class = _get_encoding_model(base_class, is_token_classification=is_token_classification)

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    return model, tokenizer


# FIX: Simplified signature - only need one base class
def _get_encoding_model(base_model_class, *, is_token_classification: bool = False):
    class RewardModel(base_model_class):
        supports_gradient_checkpointing = True
        _is_token_classification = bool(is_token_classification)
        
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_token_embeddings: bool = False,
            token_embeddings_layer: Optional[int] = None,
        ) -> torch.Tensor:
            # Only request all hidden states when we actually need them (BERTScore token embeddings,
            # or token-classification backbones where we extract embeddings from hidden states).
            need_hidden_states = bool(return_token_embeddings) or bool(self._is_token_classification)
            outputs = super().forward(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden_states,
            )

            # Access last hidden state
            if need_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
            else:
                # Base models return last_hidden_state as first tuple element.
                try:
                    last_hidden = outputs[0]
                except Exception:
                    last_hidden = getattr(outputs, "last_hidden_state", None)
                if last_hidden is None:
                    raise RuntimeError("Failed to obtain last_hidden_state from embedding backbone outputs.")

            # For BERTScore-style metrics we need token-level contextual embeddings.
            # Return the full (batch, seq_len, hidden) tensor when requested.
            if return_token_embeddings:
                if not (hasattr(outputs, "hidden_states") and outputs.hidden_states is not None):
                    # If the backbone didn't return hidden states, fall back to last_hidden_state.
                    return last_hidden
                hs = outputs.hidden_states
                if token_embeddings_layer is None:
                    return hs[-1]
                try:
                    idx = int(token_embeddings_layer)
                except Exception:
                    idx = -1
                if idx < 0:
                    idx = len(hs) + idx
                if idx < 0 or idx >= len(hs):
                    idx = len(hs) - 1
                return hs[idx]

            if hasattr(outputs, "logits") and outputs.logits is not None:
                # For PRMs, use last non-padding token embedding
                seq_lengths = attention_mask.sum(1) - 1
                batch_size = last_hidden.shape[0]
                last_token_embeddings = last_hidden[
                    torch.arange(batch_size, device=last_hidden.device),
                    seq_lengths,
                    :,
                ]
                return last_token_embeddings
            else:
                # # For regular models, use CLS token embedding (first token)
                # cls_embeddings = last_hidden[:, 0, :]
                # return cls_embeddings
 
                # For Sentence Transformers, use mean pooling
                # Mask out padding tokens
                masked_embeddings = last_hidden * attention_mask.unsqueeze(-1)
                # Sum and average over non-padding tokens
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                mean_embeddings = sum_embeddings / sum_mask
                return mean_embeddings

    return RewardModel



 






#################################################################################################################
#################################################################################################################
#################################################################################################################




# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model.from_config(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model.from_config(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # (1, total_seqs)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel

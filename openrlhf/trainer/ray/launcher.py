import logging
import os
import socket
from typing import Dict, Optional, Type, List

import ray
import torch
import torch.nn.functional as F
import math
from collections import Counter, defaultdict
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from openrlhf.models import Actor, get_llm_for_sequence_regression, get_llm_for_text_embedding, get_comet_model_for_text_embedding
from openrlhf.models.utils import build_strided_attention_mask_and_positions
from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.deepspeed import DeepspeedStrategy



class BaseDistributedActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseModelActor(BaseDistributedActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Process input data by calling specified function for each item in the lists.

        Args:
            method_name (str): Name of the function to execute
            kwargs: Reference to the chunk of data to process

        Returns:
            List[Any]: List of results from function execution
        """

        # Get the first parameter to determine list length
        kwargs = {key: value[start_idx:end_idx] for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        # Verify all parameters have same length
        for param_name, param_value in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(f"Parameter {param_name} has length {len(param_value)}, expected {list_length}")

        # Get the function to execute
        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        results = []
        for i in tqdm(range(list_length), desc=f"{method_name}", disable=not self.strategy.is_rank_0()):
            # Create kwargs for single item
            sample_kwargs = {param_name: param_value[i] for param_name, param_value in kwargs.items()}

            result = func(**sample_kwargs)
            results.append(result)

        return results


@ray.remote(num_gpus=1)
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
        override_temperature: bool = False,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
                override_temperature=override_temperature,
            )
        return log_probs.to("cpu")




@ray.remote(num_gpus=1)
class EBFTCometRewardModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)

        # COMET model must stay in float32 due to PyTorch Lightning internal dtype handling
        # Pass bf16=False regardless of strategy.args.bf16 to avoid dtype mismatches
        model, self.tokenizer = get_comet_model_for_text_embedding(
                pretrain,
                bf16=False,  # Always use float32 for COMET
                load_in_4bit=strategy.args.load_in_4bit,
                # ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        # Don't wrap COMET model with DeepSpeed - it causes dtype conflicts
        # with PyTorch Lightning's internal handling. Just use it as a plain model.
        # Move model to GPU manually
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.model = model.to(device)
        self.model.eval()

    def forward(
        self,
        input_sequences: List[str],
        gt_sequences: List[str],
        ct_sequences: List[str],
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            outputs = self.model(
                input_sequences,
                gt_sequences,
                ct_sequences,
            )

        # COMET model returns a list of scores, convert to tensor
        if isinstance(outputs, list):
            outputs = torch.tensor(outputs, dtype=torch.float32)

        return outputs.to("cpu")





@ray.remote(num_gpus=1)
class EBFTReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

        self.debug = strategy.args.debug
    

    def forward_strided_blocks(
        self,
        full_sequences: torch.LongTensor,  # Z: full sequence (full prompt + generated)
        action_mask: torch.Tensor,
        prompt_length: int,  # Original full prompt length
        generation_step: int, #
        num_blocks: int,  # Number of prediction blocks
        stride: int,  # Context stride between blocks
        context_length: int,
        doc_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute action log probabilities for full sequences using strided attention.

        This function computes forward passes for sequences generated using the strided
        multi-token prediction approach. It builds appropriate attention masks that allow
        each prediction block to only see its designated context window.

        Args:
            full_sequences: The full sequence tensor (Z) containing full prompt + generations
            action_mask: Mask indicating which positions are actions (generated tokens)
            prompt_length: Length of the original full prompt (X)
            num_blocks: Number of parallel prediction blocks
            stride: Context stride between consecutive blocks

        Returns:
            Action log probabilities for the full sequence
        """
        device = torch.cuda.current_device()
        with torch.no_grad():
            # Convert augmented_sequence to tensor if it's a list
            full_sequences_batch = torch.tensor(full_sequences, dtype=torch.long)
            if full_sequences_batch.dim() == 1:
                full_sequences_batch = full_sequences_batch.unsqueeze(0)

            # Build the strided attention mask for the augmented sequence
            # This mask ensures each block only attends to its designated context window
            # attention_mask, position_ids = build_strided_attention_mask_and_positions(
            #     full_sequence_length=full_sequences_batch.size(1),
            #     prompt_length=prompt_length,
            #     generation_step=generation_step,
            #     max_generation_length=generation_step,
            #     stride=stride,
            #     num_blocks=num_blocks,
            #     device=device,
            #     document_masking=self.args.document_masking,
            # )
            attention_mask, position_ids = build_strided_attention_mask_and_positions(
                full_sequence_length=full_sequences_batch.size(1),  # Total sequence length
                prompt_length=prompt_length,  # Original prompt length,
                context_length=context_length,
                generation_step=generation_step,  # Number of tokens generated
                max_generation_length=generation_step,  # Total number of tokens to generate
                stride=stride,
                num_blocks=num_blocks,
                device=device,
                document_masking=self.strategy.args.document_masking,
                doc_ids=doc_ids,
            )

            # Debug: Print forward pass information if debug flag is enabled
            if self.debug:
                print(f"\n{'='*60}")
                print(f"REFERENCE FORWARD STRIDED BLOCKS")
                print(f"{'='*60}")

                # Visualize the attention mask
                vis_mask = attention_mask.detach().cpu().int().numpy()
                vis_mask[vis_mask != 0] = 1.0  # Convert non-zero values to 1 for visualization

                print(f"\n📊 ATTENTION MASK (0=can attend, 1=masked):")
                print(f"   Shape: {vis_mask.shape}")
                if vis_mask.shape[1] <= 50:  # Only print small matrices for readability
                    print(f"   Matrix:\n{vis_mask}")
                else:
                    print(f"   Matrix too large to display (>{50} tokens)")

                print(f"\n📦 FULL SEQUENCE:")
                print(f"   Length: {full_sequences_batch.size(1)}")
                print(f"   Shape: {full_sequences_batch.shape}")

                print(f"\n🔢 POSITION IDS:")
                print(f"   Values: {position_ids}")

                print(f"\n📍 FORWARD PARAMETERS:")
                print(f"   Prompt length: {prompt_length}")
                print(f"   Generation step: {generation_step}")
                print(f"   Number of blocks: {num_blocks}")
                print(f"   Stride: {stride}")
                print(f"{'='*60}\n")

            # Forward pass through the actor model to compute log probabilities
            log_probs = self.model(
                full_sequences_batch.to(device),
                None, #torch.ones_like(action_mask).to(device), #action mask
                attention_mask.to(device),
                pos_ids=position_ids,
                return_logprobs=True,
                ring_attn_group=self.strategy.ring_attn_group,
                prompt_len=prompt_length,
                context_len=context_length,
                num_blocks=num_blocks,
                stride=stride,
            )

        return log_probs.to("cpu")







@ray.remote(num_gpus=1)
class EBFTRewardModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model, self.tokenizer = get_llm_for_text_embedding(
                pretrain,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                # ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),

        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=False)
        self.model.eval()
        self._pretrain_name = str(pretrain)
        # Cached BERTScore settings (computed lazily on first use)
        self._bertscore_num_layers = None
        self._bertscore_baseline_f = None

    def forward(
        self,
        input_sequences: List[str],
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        
        # Replace empty sequences with a single space to ensure valid tokenization
        input_sequences = [seq if seq and seq.strip() else " " for seq in input_sequences]
        
        with torch.no_grad():
            # Tokenize sequences
            # input_tokens = self.tokenizer(input_sequences, padding=True, return_tensors='pt')
            
             
            input_tokens = self.tokenizer(
                input_sequences, 
                padding=True, 
                truncation=True,  # Add this
                max_length=512,   # Add this - BERT's max
                return_tensors='pt'
            )

            # DEBUG: Check token lengths to verify if sequences exceed BERT's 512 max
            seq_length = input_tokens['input_ids'].shape[1]
            print(f"[EBFTRewardModelActor] Tokenized batch shape: {input_tokens['input_ids'].shape}, max_seq_len: {seq_length}")
            if seq_length > 512:
                print(f"[EBFTRewardModelActor] WARNING: Sequence length {seq_length} exceeds BERT's max position embeddings (512)!")

            # Move to device while explicitly preserving integer dtypes for index tensors
            input_tokens = {
                k: v.to(
                    device=device, 
                    dtype=torch.long if k in ['input_ids', 'attention_mask', 'token_type_ids'] else v.dtype
                )
                for k, v in input_tokens.items()
            }
            
            # Forward
            input_embeddings = self.model(**input_tokens)

        return input_embeddings.to("cpu")

    def bertscore_f1(
        self,
        hypotheses: List[str],
        references: List[str],
        *,
        batch_size: int = 8,
        max_length: int = 512,
        idf: bool = True,
        rescale_with_baseline: bool = True,
        lang: str = "en",
    ) -> List[float]:
        """Compute BERTScore F1 with IDF weighting + baseline rescaling (bert-score compatible).

        This is designed to be close to the community-standard `bert_score` package:
        - IDF weighting: enabled by default (computed from the full `references` corpus).
        - Baseline rescaling: enabled by default, using vendored baseline constants from
          `bert-score==0.3.13`.
        - Layer selection: uses bert-score's recommended `num_layers` for this model type.

        We run this on GPU inside the existing reward_pretrain Ray actor to avoid introducing
        new model-loading paths in the trainer.
        """
        device = torch.cuda.current_device()

        n = min(len(hypotheses), len(references))
        if n <= 0:
            return []

        # Normalize inputs (match bert-score behavior: empty string becomes [CLS][SEP]).
        hyps_all = [("" if h is None else str(h)).strip() for h in hypotheses[:n]]
        refs_all = [("" if r is None else str(r)).strip() for r in references[:n]]

        # Determine bert-score layer selection and baseline rescaling constants.
        model_type = getattr(self, "_pretrain_name", None) or ""
        num_layers = getattr(self, "_bertscore_num_layers", None)
        baseline_f = getattr(self, "_bertscore_baseline_f", None)

        if num_layers is None or (rescale_with_baseline and baseline_f is None):
            from openrlhf.utils.bertscore_official import get_default_num_layers, load_baseline_vals

            if num_layers is None:
                num_layers = get_default_num_layers(model_type) or 0
                self._bertscore_num_layers = int(num_layers)
            if rescale_with_baseline and baseline_f is None:
                baseline = load_baseline_vals(lang=str(lang), model_type=str(model_type), num_layers=int(num_layers))
                baseline_f = float(baseline.f)
                self._bertscore_baseline_f = baseline_f

        # Compute IDF dictionary from the full reference corpus (bert-score compatible).
        if idf:
            num_docs = len(refs_all)
            idf_count = Counter()
            for sent in refs_all:
                token_ids = self.tokenizer.encode(
                    sent,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
                idf_count.update(set(token_ids))
            idf_dict = defaultdict(lambda: math.log((num_docs + 1) / 1.0))
            idf_dict.update({idx: math.log((num_docs + 1) / (c + 1.0)) for (idx, c) in idf_count.items()})
        else:
            idf_dict = defaultdict(lambda: 1.0)
            # Match bert-score: exclude CLS/SEP by setting weight 0.
            try:
                if getattr(self.tokenizer, "sep_token_id", None) is not None:
                    idf_dict[int(self.tokenizer.sep_token_id)] = 0.0
                if getattr(self.tokenizer, "cls_token_id", None) is not None:
                    idf_dict[int(self.tokenizer.cls_token_id)] = 0.0
            except Exception:
                pass

        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = 8
        batch_size = max(1, batch_size)

        out_scores: List[float] = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, n, batch_size):
                hyps = hyps_all[start : start + batch_size]
                refs = refs_all[start : start + batch_size]

                hyp_tokens = self.tokenizer(
                    hyps,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                ref_tokens = self.tokenizer(
                    refs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                # Move to GPU (preserve integer dtype for index tensors)
                hyp_tokens = {
                    k: v.to(
                        device=device,
                        dtype=torch.long if k in ["input_ids", "attention_mask", "token_type_ids"] else v.dtype,
                    )
                    for k, v in hyp_tokens.items()
                }
                ref_tokens = {
                    k: v.to(
                        device=device,
                        dtype=torch.long if k in ["input_ids", "attention_mask", "token_type_ids"] else v.dtype,
                    )
                    for k, v in ref_tokens.items()
                }

                hyp_ids = hyp_tokens.get("input_ids")
                ref_ids = ref_tokens.get("input_ids")
                hyp_mask = hyp_tokens.get("attention_mask")
                ref_mask = ref_tokens.get("attention_mask")
                if hyp_ids is None or ref_ids is None or hyp_mask is None or ref_mask is None:
                    raise RuntimeError("Tokenizer outputs missing input_ids/attention_mask for BERTScore.")

                hyp_mask = hyp_mask.bool()
                ref_mask = ref_mask.bool()

                # Token embeddings at bert-score-selected layer: [bs, L, H]
                hyp_emb = self.model(**hyp_tokens, return_token_embeddings=True, token_embeddings_layer=int(num_layers))
                ref_emb = self.model(**ref_tokens, return_token_embeddings=True, token_embeddings_layer=int(num_layers))

                # Normalize for cosine similarity (match bert-score: normalize in-place)
                hyp_emb = hyp_emb.float()
                ref_emb = ref_emb.float()
                hyp_emb = hyp_emb / torch.norm(hyp_emb, dim=-1, keepdim=True)
                ref_emb = ref_emb / torch.norm(ref_emb, dim=-1, keepdim=True)

                # Similarity + padding masking (match greedy_cos_idf)
                sim = torch.bmm(hyp_emb, ref_emb.transpose(1, 2))
                pair_mask = torch.bmm(hyp_mask.unsqueeze(2).float(), ref_mask.unsqueeze(1).float())
                sim = sim * pair_mask

                word_precision = sim.max(dim=2).values  # [bs, Lh]
                word_recall = sim.max(dim=1).values     # [bs, Lr]

                # IDF weights per token (padding -> 0)
                hyp_idf = torch.tensor(
                    [[float(idf_dict[int(t)]) for t in row] for row in hyp_ids.detach().cpu().tolist()],
                    device=device,
                    dtype=torch.float32,
                )
                ref_idf = torch.tensor(
                    [[float(idf_dict[int(t)]) for t in row] for row in ref_ids.detach().cpu().tolist()],
                    device=device,
                    dtype=torch.float32,
                )
                hyp_idf = hyp_idf * hyp_mask.float()
                ref_idf = ref_idf * ref_mask.float()

                # Normalize idf weights (match bert-score: in-place division, allow NaNs for empty sentences)
                hyp_idf = hyp_idf / hyp_idf.sum(dim=1, keepdim=True)
                ref_idf = ref_idf / ref_idf.sum(dim=1, keepdim=True)

                P = (word_precision * hyp_idf).sum(dim=1)
                R = (word_recall * ref_idf).sum(dim=1)
                F1 = (2.0 * P * R) / (P + R)

                # Empty sentence handling (match bert-score: mask sum == 2 means only [CLS][SEP])
                hyp_zero_mask = hyp_mask.sum(dim=1).eq(2)
                ref_zero_mask = ref_mask.sum(dim=1).eq(2)
                if torch.any(hyp_zero_mask):
                    P = P.masked_fill(hyp_zero_mask, 0.0)
                    R = R.masked_fill(hyp_zero_mask, 0.0)
                if torch.any(ref_zero_mask):
                    P = P.masked_fill(ref_zero_mask, 0.0)
                    R = R.masked_fill(ref_zero_mask, 0.0)
                F1 = F1.masked_fill(torch.isnan(F1), 0.0)

                if rescale_with_baseline:
                    # Only rescale F1 (we only log F1).
                    F1 = (F1 - float(baseline_f)) / (1.0 - float(baseline_f))

                out_scores.extend(F1.detach().float().cpu().tolist())

        return out_scores


@ray.remote(num_gpus=1)
class RewardModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            normalize_reward=strategy.args.normalize_reward,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_sequence: bool = False,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(
                sequences.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
            )
        return reward.to("cpu")


class RayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BaseModelActor]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        pg_bundle_start (int, optional): Bundle index offset to allow sharing a
            placement group across different actor groups.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
        pg_bundle_start: int = 0,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        # duplicate actors is ring_attn_size * tensor_parallel_size
        self.duplicate_actors = duplicate_actors

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node
        self._pg_bundle_start = pg_bundle_start

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        bundle_offset = self._pg_bundle_start
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle_offset
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker_actor
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_offset + rank,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        """Run method on all actors with batched input data asynchronously using round-robin scheduling.
        Each actor processes one chunk of data at a time. Actors in the same ring / tensor parallel group process the same chunk.

        Args:
            method_name (str): Name of the method to run
            **kwargs: Keyword arguments for the method. Each value should be a list/tensor of the same length.

        Returns:
            List[ray.ObjectRef]: List of remote object references to the results
        """
        # Check if all kwargs parameters are iterable
        for key, value in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")

        # Get the length of the first parameter as reference
        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)

        # Verify all parameters have the same length
        for key, value in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
                )

        # Calculate chunk size based on number of effective actors (considering ring groups)
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        chunk_size = total_length // effective_actors
        assert (
            total_length >= effective_actors
        ), f"Total length {total_length} must be greater than or equal to effective actors {effective_actors}"
        if total_length % effective_actors != 0:
            chunk_size += 1

        all_data_ref = ray.put(kwargs)

        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)

            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]

                refs.append(actor.execute_batch.remote(method_name, all_data_ref, start_idx, end_idx))

        return refs
import math
import os
import time
import signal
import re
from abc import ABC
from contextlib import contextmanager
from datetime import timedelta
import torch
from collections import Counter
from torch.optim import Optimizer
from tqdm import tqdm
import torch.nn.functional as F
from openrlhf.models import SFTLoss, get_llm_for_text_embedding
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.math_verifier import verify_llm_answer
from openrlhf.utils.utils import zero_pad_sequences
from vllm import LLM, SamplingParams
from transformers import AutoConfig

from sacrebleu import sentence_bleu
from comet import download_model, load_from_checkpoint
import numpy as np




logger = init_logger(__name__)


MBPP_FEWSHOT_PREFIX = '''"""
Write a function to find the similar elements from the given two tuple lists.
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
"""
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)

"""
Write a python function to identify non-prime numbers.
assert is_not_prime(2) == False
"""
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result

"""
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
"""
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
'''


# 8-shot examples for GSM8K evaluation (standard from the GSM8K paper)
GSM8K_FEWSHOT_PREFIX = '''Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5 cars are now in the parking lot.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74 chocolates. After eating 35, they had 74 - 35 = 39 pieces left.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. He now has 12 lollipops. So he gave Denny 20 - 12 = 8 lollipops.
#### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He got 2 toys from his mom and 2 toys from his dad. So he got 2 + 2 = 4 toys. Now he has 5 + 4 = 9 toys.
#### 9

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. 5 computers were installed each day from Monday to Thursday, so 5 * 4 = 20 computers were installed. 9 + 20 = 29 computers are now in the server room.
#### 29

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35 golf balls. After losing 2 more on wednesday, he had 35 - 2 = 33 golf balls.
#### 33

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. She bought 5 bagels for 3 dollars each, so she spent 5 * 3 = 15 dollars. She has 23 - 15 = 8 dollars left.
#### 8

Question: '''


# 4-shot examples for MATH evaluation (standard from Minerva/common benchmarks)
MATH_FEWSHOT_PREFIX = r'''Problem: Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.
Solution: To find the domain, we need both square roots to be defined and the denominator to be non-zero.
For $\sqrt{x-2}$ to be defined, we need $x-2 \geq 0$, so $x \geq 2$.
For $\sqrt{5-x}$ to be defined and non-zero, we need $5-x > 0$, so $x < 5$.
Combining these conditions: $2 \leq x < 5$.
The answer is $\boxed{[2,5)}$.

Problem: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12$, then find $\det (\mathbf{A} \mathbf{B})$.
Solution: We have the property that $\det(\mathbf{A}\mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})$.
Therefore, $\det(\mathbf{A}\mathbf{B}) = 2 \cdot 12 = 24$.
The answer is $\boxed{24}$.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: With two 20-pound weights lifted 12 times, Terrell lifts a total of $2 \cdot 20 \cdot 12 = 480$ pounds.
With two 15-pound weights, each lift is $2 \cdot 15 = 30$ pounds.
To lift 480 pounds total, he needs $\frac{480}{30} = 16$ lifts.
The answer is $\boxed{16}$.

Problem: If the system of equations \begin{align*} 6x-4y&=a,\\ 6y-9x &=b. \end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b}$, assuming $b$ is nonzero.
Solution: Multiply the first equation by $-\frac{3}{2}$ to get $-9x + 6y = -\frac{3a}{2}$.
Since $-9x + 6y = b$, we have $b = -\frac{3a}{2}$.
Therefore, $\frac{a}{b} = \frac{a}{-\frac{3a}{2}} = -\frac{2}{3}$.
The answer is $\boxed{-\frac{2}{3}}$.

Problem: '''


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time using signal.alarm.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutException: If execution exceeds the time limit
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Code execution timed out after {seconds} seconds")

    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def _sanitize_generated_code(response: str, keep_leading_def: bool = False) -> str:
    if not response:
        return ""
    text = response
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            block_lines = block.splitlines()
            if block_lines and block_lines[0].strip().lower().startswith(("python", "py")):
                block = "\n".join(block_lines[1:])
            text = block
    lines = text.splitlines()
    if keep_leading_def:
        # Drop leading explanations until we hit code-like content.
        start_idx = 0
        while start_idx < len(lines):
            stripped = lines[start_idx].strip()
            if not stripped or stripped.startswith("#"):
                start_idx += 1
                continue
            if stripped.startswith(("def ", "class ", "import ", "from ", "@")):
                break
            start_idx += 1
        lines = lines[start_idx:]
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped:
            lowered = stripped.lower()
            if stripped.startswith(('"""', "'''")):
                if not keep_leading_def:
                    break
                quote = stripped[:3]
                # Skip a leading docstring block.
                if stripped.count(quote) >= 2 and stripped != quote:
                    i += 1
                    continue
                i += 1
                while i < len(lines):
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1
                continue
            if lowered.startswith(("assert ", "print(", "print ", "if __name__")):
                break
            if (
                stripped.startswith(("def ", "class "))
                and not line.startswith((" ", "\t"))
                and not keep_leading_def
            ):
                break
            if lowered.startswith(("# test", "# tests", "#test", "#tests")):
                break
        cleaned_lines.append(line)
        i += 1
    return "\n".join(cleaned_lines).rstrip()


def _build_mbpp_code(
    prompt: str,
    response: str,
    function_name: str,
    helper_code: str,
    function_signature: str = None,
    keep_leading_def: bool = None,
) -> str:
    prompt_lines = (prompt or "").splitlines()
    prompt_has_target_def = False
    if function_name:
        prompt_has_target_def = any(
            line.lstrip().startswith(f"def {function_name}(") for line in prompt_lines
        )
    if not function_name and prompt_lines:
        prompt_has_target_def = any(
            line.lstrip().startswith(("def ", "class ")) for line in prompt_lines
        )
    if keep_leading_def is None:
        keep_leading_def = not prompt_has_target_def
    cleaned = _sanitize_generated_code(response or "", keep_leading_def=keep_leading_def)
    if not cleaned.strip():
        cleaned = ""

    lines = cleaned.splitlines() if cleaned else []
    def_indices = [
        i for i, line in enumerate(lines)
        if line.lstrip() == line and (line.startswith("def ") or line.startswith("class "))
    ]

    if def_indices:
        preamble = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                preamble.append(line)
                i += 1
                continue
            if stripped.startswith("#") or stripped.startswith("import ") or stripped.startswith("from "):
                preamble.append(line)
                i += 1
                continue
            if stripped.startswith("@"):
                preamble.append(line)
                i += 1
                continue
            if (
                line.lstrip() == line
                and (line.startswith("def ") or line.startswith("class "))
            ):
                break
            if line.lstrip() == line and "=" in stripped:
                preamble.append(line)
            i += 1

        blocks = []
        found_target = False
        idx = def_indices[0]
        while idx < len(lines):
            line = lines[idx]
            if line.lstrip() == line and (line.startswith("def ") or line.startswith("class ")):
                start = idx
                idx += 1
                while idx < len(lines):
                    nxt = lines[idx]
                    if nxt.lstrip() == nxt and (nxt.startswith("def ") or nxt.startswith("class ")):
                        break
                    idx += 1
                block = lines[start:idx]
                blocks.append(block)
                if function_name and line.startswith(f"def {function_name}("):
                    found_target = True
                    break
                if not function_name:
                    break
                continue
            idx += 1

        if blocks:
            if function_name and not found_target:
                blocks = [blocks[0]]
            code_lines = preamble + [ln for blk in blocks for ln in blk]
            code_text = "\n".join(code_lines).rstrip()
            if helper_code:
                helper_code = helper_code.rstrip()
                if helper_code:
                    code_text = helper_code + "\n\n" + code_text
            return code_text

    body = cleaned.strip("\n")
    if body:
        indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines())
    else:
        indented = "    pass"
    prompt_for_body = prompt.rstrip()
    if not prompt_has_target_def and function_signature:
        signature = function_signature.rstrip()
        if signature:
            prompt_for_body = f"{prompt_for_body}\n{signature}"
    full_code = prompt_for_body.rstrip() + "\n" + indented
    if helper_code:
        helper_code = helper_code.rstrip()
        if helper_code:
            full_code = helper_code + "\n\n" + full_code
    return full_code


def _run_code_in_subprocess(code: str, unit_tests: list, timeout: int = 3) -> tuple:
    """Execute code and unit tests in an isolated subprocess.

    This function runs generated code in a completely separate process to protect
    against sys.exit(), os._exit(), segfaults, and other process-terminating operations.

    Args:
        code: The generated code to execute
        unit_tests: List of unit test statements to run
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_type: str or None)
    """
    import subprocess
    import sys
    import textwrap

    # Create a test script that will be run in a subprocess
    test_script = textwrap.dedent(f'''
import sys
import signal

# Override sys.exit to prevent it from killing the process
_original_exit = sys.exit
def _safe_exit(code=0):
    raise SystemExit(code)
sys.exit = _safe_exit

# Also override os._exit
import os
_original_os_exit = os._exit
def _safe_os_exit(code=0):
    raise SystemExit(code)
os._exit = _safe_os_exit

# Disable SIGINT and SIGTERM from killing the process unexpectedly
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)

# Create a restricted namespace
namespace = {{
    '__builtins__': __builtins__,
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'range': range,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}}

# Execute the generated code
try:
    exec({repr(code)}, namespace)
except SyntaxError as e:
    print(f"ERROR:syntax:{{type(e).__name__}}:{{e}}")
    sys.stdout.flush()
    _original_exit(0)
except SystemExit:
    # Catch sys.exit() calls (e.g., from argparse)
    print("ERROR:syntax")
    sys.stdout.flush()
    _original_exit(0)
except Exception as e:
    print(f"ERROR:syntax:{{type(e).__name__}}:{{e}}")
    sys.stdout.flush()
    _original_exit(0)

# Execute unit tests
unit_tests = {repr(unit_tests)}
try:
    for test in unit_tests:
        stmt = str(test).strip()
        if not stmt:
            continue
        exec(stmt, namespace)
except AssertionError:
    print("ERROR:test_failure")
    sys.stdout.flush()
    _original_exit(0)
except SystemExit:
    # Catch sys.exit() calls in tests
    print("ERROR:test_failure")
    sys.stdout.flush()
    _original_exit(0)
except Exception as e:
    print("ERROR:test_failure")
    sys.stdout.flush()
    _original_exit(0)

# Success
print("SUCCESS")
sys.stdout.flush()
''')

    try:
        # Run the script in a subprocess with timeout
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Prevent the subprocess from inheriting file descriptors
            close_fds=True,
            # Run in a new process group to isolate signals
            start_new_session=True,
        )

        # Parse the output
        output = result.stdout.strip()
        if "SUCCESS" in output:
            return True, None
        elif "ERROR:syntax" in output:
            return False, "syntax"
        elif "ERROR:test_failure" in output:
            return False, "test_failure"
        else:
            # Unexpected output or process crashed
            return False, "syntax"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "syntax"
    

class RandomFourierFeatures:
    """
    Random Fourier Features (RFF) for kernel approximation.
    This creates a more discriminative similarity metric by mapping embeddings
    to a higher-dimensional random feature space.

    Reference: "Random Features for Large-Scale Kernel Machines" (Rahimi & Recht, 2007)
    """
    def __init__(self, input_dim, num_features=1024, gamma=1.0, device='cuda'):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_features: Number of random features (output dimension)
            gamma: Bandwidth parameter for RBF kernel (higher = more discriminative)
            device: Device to store tensors on
        """
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.device = device

        # Sample random weights from Gaussian distribution
        # W ~ N(0, 2*gamma*I)
        self.W = torch.randn(input_dim, num_features, device=device) * np.sqrt(2 * gamma)

        # Sample random bias from uniform [0, 2π]
        self.b = torch.rand(num_features, device=device) * 2 * np.pi

    def transform(self, X):
        """
        Transform input embeddings to random Fourier features.

        Args:
            X: Input tensor of shape (batch_size, input_dim)

        Returns:
            Z: Transformed features of shape (batch_size, num_features)
        """
        # Ensure X is on the correct device
        if X.device != self.device:
            X = X.to(self.device)

        # Ensure X has the same dtype as self.W to avoid matmul dtype mismatch
        if X.dtype != self.W.dtype:
            X = X.to(self.W.dtype)

        # Z = sqrt(2/D) * cos(W^T * X + b)
        projection = torch.matmul(X, self.W) + self.b
        Z = torch.sqrt(torch.tensor(2.0 / self.num_features, device=self.device)) * torch.cos(projection)

        return Z


class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        eval_perplexity_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        top_p: float = 0.95,
        max_tokens: int = 512,
        temperature: float = 0.6,
        humaneval_dataloader=None,
        mbpp_dataloader=None,
        gsm8k_dataloader=None,
        math_dataloader=None,
        mmlu_dataloader=None,
        arc_dataloader=None,
        arc_easy_dataloader=None,
        obqa_dataloader=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_perplexity_dataloader = eval_perplexity_dataloader
        self.humaneval_dataloader = humaneval_dataloader
        self.mbpp_dataloader = mbpp_dataloader
        self.gsm8k_dataloader = gsm8k_dataloader
        self.math_dataloader = math_dataloader
        self.mmlu_dataloader = mmlu_dataloader
        self.arc_dataloader = arc_dataloader
        self.arc_easy_dataloader = arc_easy_dataloader
        self.obqa_dataloader = obqa_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.loss_fn = SFTLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        # Initialize vLLM engine once for evaluation
        self.vllm_engine = None
        # Don't initialize vLLM at startup - we'll create it fresh during evaluation
        if self.strategy.is_rank_0():
            self._initialize_vllm()
            self._initialize_translation_metrics()


    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = float("inf") #num_update_steps_per_epoch  # Evaluate once per epoch
        steps_to_save = None
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
            save_log_scale_count = getattr(args, "save_log_scale_count", -1)
            if save_log_scale_count != -1:
                total_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
                if save_log_scale_count > 0 and total_steps > 0:
                    logspace = np.logspace(-2.1, 0, save_log_scale_count) * total_steps
                    steps_to_save = sorted({max(1, min(total_steps, int(n))) for n in logspace})
                else:
                    steps_to_save = []

        args.steps_to_save = steps_to_save

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        # Perform initial evaluation at step 0 (before training starts)
        if step == 1 and start_epoch == 0 and (not math.isinf(args.eval_steps)):
            has_eval_data = self.eval_dataloader is not None and len(self.eval_dataloader) > 0
            has_swallow_code_eval = "swallow_code" in self.args.eval_dataset and (
                (self.humaneval_dataloader is not None and len(self.humaneval_dataloader) > 0)
                or (self.mbpp_dataloader is not None and len(self.mbpp_dataloader) > 0)
            )
            has_setting3_math = "sjelassi/setting3_math" in self.args.eval_dataset
            if has_eval_data or has_swallow_code_eval or has_setting3_math:
                logger.info("🔍 Running initial evaluation at step 0...")
                if has_setting3_math:
                    if self.gsm8k_dataloader is not None:
                        logger.info(
                            f"🔍 Running GSM8K eval on {len(self.gsm8k_dataloader)} batches"
                        )
                        self.evaluate_gsm8k_math(self.gsm8k_dataloader, 0)
                    if self.math_dataloader is not None:
                        logger.info(
                            f"🔍 Running MATH eval on {len(self.math_dataloader)} batches"
                        )
                        self.evaluate_gsm8k_math(self.math_dataloader, 0)
                elif "gsm8k" in self.args.eval_dataset or "math" in self.args.eval_dataset:
                    self.evaluate_gsm8k_math(self.eval_dataloader, 0)
                elif "fineweb" in self.args.eval_dataset or "finepdf" in self.args.eval_dataset:
                    # if self.mmlu_dataloader is not None:
                    logger.info(f"🔍 Running MMLU eval on {len(self.mmlu_dataloader)} batches")
                    self.evaluate_downstream_mmlu(self.mmlu_dataloader, 0)
                    # if self.arc_dataloader is not None:
                    logger.info(f"🔍 Running ARC eval on {len(self.arc_dataloader)} batches")
                    self.evaluate_downstream_arc(self.arc_dataloader, 0, metric_prefix="arc_challenge")
                    # if self.arc_easy_dataloader is not None:
                    logger.info(f"🔍 Running ARC-Easy eval on {len(self.arc_easy_dataloader)} batches")
                    self.evaluate_downstream_arc(self.arc_easy_dataloader, 0, metric_prefix="arc_easy")
                # if self.obqa_dataloader is not None:
                    logger.info(f"🔍 Running OBQA eval on {len(self.obqa_dataloader)} batches")
                    self.evaluate_downstream_obqa(self.obqa_dataloader, 0)
                elif "swallow_code" in self.args.eval_dataset:
                    if self.humaneval_dataloader is not None:
                        logger.info(
                            f"🔍 Running HumanEval eval on {len(self.humaneval_dataloader)} batches"
                        )
                        self.evaluate_downstream_humaneval(self.humaneval_dataloader, 0)
                    if self.mbpp_dataloader is not None:
                        logger.info(f"🔍 Running MBPP eval on {len(self.mbpp_dataloader)} batches")
                        self.evaluate_downstream_mbpp(self.mbpp_dataloader, 0)
                elif "opencode" in self.args.eval_dataset or "omi_code" in self.args.eval_dataset:
                    logger.info("🔍 Skipping evaluation for code dataset (use perplexity eval only)")
                else:
                    self.evaluate_translation(self.eval_dataloader, 0)
            # fewewfwf
            if self.eval_perplexity_dataloader is not None and len(self.eval_perplexity_dataloader) > 0:
                logger.info("🔍 Running initial perplexity evaluation at step 0...")
                self.evaluate_perplexity(self.eval_perplexity_dataloader, 0)
        
        # ewffewewfew

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for inputs, attention_masks, loss_masks in self.train_dataloader:
            # for inputs  in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                # attention_mask = torch.ones_like(inputs) 
                # loss_mask = torch.ones_like(inputs)
                per_token_log_probs, output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )
            

                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                gpt_loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when eval data is available, avoid zero division in eval.
            has_eval_data = self.eval_dataloader is not None and len(self.eval_dataloader) > 0
            has_swallow_code_eval = "swallow_code" in self.args.eval_dataset and (
                (self.humaneval_dataloader is not None and len(self.humaneval_dataloader) > 0)
                or (self.mbpp_dataloader is not None and len(self.mbpp_dataloader) > 0)
            )
            has_setting3_math = "sjelassi/setting3_math" in self.args.eval_dataset
            if has_eval_data or has_swallow_code_eval or has_setting3_math:
                logger.info(f"🔍 Running evaluation at step {global_step}...")
                if has_setting3_math:
                    if self.gsm8k_dataloader is not None:
                        logger.info(
                            f"🔍 Running GSM8K eval on {len(self.gsm8k_dataloader)} batches"
                        )
                        self.evaluate_gsm8k_math(self.gsm8k_dataloader, global_step)
                    if self.math_dataloader is not None:
                        logger.info(
                            f"🔍 Running MATH eval on {len(self.math_dataloader)} batches"
                        )
                        self.evaluate_gsm8k_math(self.math_dataloader, global_step)
                elif "gsm8k" in self.args.eval_dataset or "math" in self.args.eval_dataset:
                    self.evaluate_gsm8k_math(self.eval_dataloader, global_step)
                elif "fineweb" in self.args.eval_dataset or "finepdf" in self.args.eval_dataset:
                    # if self.mmlu_dataloader is not None:
                    logger.info(f"🔍 Running MMLU eval on {len(self.mmlu_dataloader)} batches")
                    self.evaluate_downstream_mmlu(self.mmlu_dataloader, global_step)
                    # if self.arc_dataloader is not None:
                    logger.info(f"🔍 Running ARC eval on {len(self.arc_dataloader)} batches")
                    self.evaluate_downstream_arc(self.arc_dataloader, global_step, metric_prefix="arc_challenge")
                    if self.arc_easy_dataloader is not None:
                        logger.info(f"🔍 Running ARC-Easy eval on {len(self.arc_easy_dataloader)} batches")
                        self.evaluate_downstream_arc(self.arc_easy_dataloader, global_step, metric_prefix="arc_easy")
                    # if self.obqa_dataloader is not None:
                    logger.info(f"🔍 Running OBQA eval on {len(self.obqa_dataloader)} batches")
                    self.evaluate_downstream_obqa(self.obqa_dataloader, global_step)
                elif "swallow_code" in self.args.eval_dataset:
                    if self.humaneval_dataloader is not None:
                        logger.info(
                            f"🔍 Running HumanEval eval on {len(self.humaneval_dataloader)} batches"
                        )
                        self.evaluate_downstream_humaneval(self.humaneval_dataloader, global_step)
                    if self.mbpp_dataloader is not None:
                        logger.info(f"🔍 Running MBPP eval on {len(self.mbpp_dataloader)} batches")
                        self.evaluate_downstream_mbpp(self.mbpp_dataloader, global_step)
                elif "opencode" in self.args.eval_dataset or "omi_code" in self.args.eval_dataset:
                    logger.info(f"🔍 Skipping evaluation for code dataset at step {global_step} (use perplexity eval only)")
                else:
                    self.evaluate_translation(self.eval_dataloader, global_step)
            if self.eval_perplexity_dataloader is not None and len(self.eval_perplexity_dataloader) > 0:
                logger.info(f"🔍 Running perplexity evaluation at step {global_step}...")
                self.evaluate_perplexity(self.eval_perplexity_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        should_save = False
        if getattr(args, "steps_to_save", None) is not None:
            if global_step in args.steps_to_save:
                should_save = True
        elif global_step % args.save_steps == 0:
            should_save = True

        if should_save:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    

    def evaluate_gsm8k_math(self, eval_dataloader, global_step):
        """Evaluate model performance on eval dataset with accuracy metrics.
        
        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            temperature: Temperature for sampling
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """

        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self._broadcast_to_vllm()
 
        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            all_datasources = []  # Track datasource for each prompt

            for datasources, prompts, labels in eval_dataloader:
                for prompt, datasource, label in zip(prompts, datasources, labels):
                    # Apply few-shot prefix based on datasource
                    if datasource == "gsm8k":
                        # 8-shot GSM8K format: "Question: {prompt}\nAnswer:"
                        fewshot_prompt = GSM8K_FEWSHOT_PREFIX + prompt + "\nAnswer:"
                    elif datasource == "math":
                        # 4-shot MATH format: "Problem: {prompt}\nSolution:"
                        fewshot_prompt = MATH_FEWSHOT_PREFIX + prompt + "\nSolution:"
                    else:
                        fewshot_prompt = prompt

                    all_prompts.append(fewshot_prompt)
                    all_labels.append(label)
                    all_datasources.append(datasource)

            # Generate samples
            # Get max_model_len from args (used when vLLM engine was created)
            max_model_len = getattr(self.args, 'max_seq_len', 2048)

            # For few-shot CoT evaluation, we need:
            # - 8-shot GSM8K prompts: ~700-800 tokens
            # - 4-shot MATH prompts: ~1100-1200 tokens
            # - Generation tokens for CoT reasoning: ~512-800 tokens
            # Recommended: max_seq_len >= 2048
            if max_model_len < 2048:
                logger.warning(f"⚠️ max_seq_len={max_model_len} may be too small for few-shot GSM8K/MATH evaluation. "
                             f"Recommended: --max_seq_len 2048 or higher.")

            # Prioritize fitting prompts: MATH 4-shot needs ~1200 tokens
            # Reserve at least 1300 for prompts, rest for generation (capped at 2048)
            max_prompt_len = max(1, min(1300, max_model_len - 512))  # At least 1 token for prompts
            max_tokens = min(2048, max_model_len - max_prompt_len)  # Cap at 2048

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop=[self.tokenizer.eos_token, "\nQuestion:", "\nProblem:"],  # Stop at next example
                n=1,
            )

            logger.info(f"GSM8K/MATH eval config: max_model_len={max_model_len}, max_tokens={max_tokens}, max_prompt_len={max_prompt_len}")

            # Truncate prompts that are too long instead of skipping to keep eval set intact
            filtered_prompts = []
            filtered_labels = []
            filtered_datasources = []
            truncated_by_source = {"gsm8k": 0, "math": 0, "other": 0}

            for prompt, label, datasource in zip(all_prompts, all_labels, all_datasources):
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_tokens) > max_prompt_len:
                    prompt_tokens = prompt_tokens[-max_prompt_len:]
                    prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                    key = datasource if datasource in truncated_by_source else "other"
                    truncated_by_source[key] += 1
                filtered_prompts.append(prompt)
                filtered_labels.append(label)
                filtered_datasources.append(datasource)

            total_truncated = sum(truncated_by_source.values())
            if total_truncated > 0:
                logger.warning(
                    f"⚠️ Truncated {total_truncated} prompts to {max_prompt_len} tokens "
                    f"(gsm8k: {truncated_by_source['gsm8k']}, math: {truncated_by_source['math']})"
                )

            # Update references to use filtered lists
            all_prompts = filtered_prompts
            all_labels = filtered_labels
            all_datasources = filtered_datasources

            logger.info(f"GSM8K/MATH few-shot evaluation: {len(all_prompts)} prompts (after filtering)")
            if all_prompts:
                logger.info(f"Sample prompt (first 500 chars): {all_prompts[0][:500]}...")

            if not all_prompts:
                logger.warning("No prompts remaining after length filtering; skipping GSM8K/MATH evaluation. "
                             "Increase --max_seq_len to at least 2048.")
                return
            logger.info("GSM8K/MATH evaluation examples: %d", len(all_prompts))

            # Generate responses using vLLM
            logger.info(f"Generating {len(all_prompts)} responses with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)


            all_responses = [result.outputs[0].text for result in results]
            # Evaluate responses using verification
            rewards_list = []
            mismatch_samples = []
            for i, (response, label) in enumerate(zip(all_responses, all_labels)):
                try:
                    # Verify if the generated response is correct
                    is_correct = verify_llm_answer(response, str(label))
                    rewards_list.append(1.0 if is_correct else 0.0)
                    if not is_correct and len(mismatch_samples) < 5:
                        mismatch_samples.append(
                            {
                                "idx": i,
                                "datasource": all_datasources[i],
                                "label": str(label),
                                "response": response,
                            }
                        )
                except Exception:
                    # If verification fails, treat as incorrect
                    rewards_list.append(0.0)
                    if len(mismatch_samples) < 5:
                        mismatch_samples.append(
                            {
                                "idx": i,
                                "datasource": all_datasources[i],
                                "label": str(label),
                                "response": response,
                                "error": "verify_exception",
                            }
                        )

            # Collect local statistics for each data source
            global_metrics = {}

            # Process rewards for each prompt
            for i in range(len(all_prompts)):
                datasource = all_datasources[i]

                if datasource not in global_metrics:
                    global_metrics[datasource] = {"pass1": 0, "count": 0}

                global_metrics[datasource]["pass1"] += rewards_list[i]
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                accuracy = metrics["pass1"] / metrics["count"] if metrics["count"] > 0 else 0.0
                logs[f"eval_{datasource}_accuracy"] = accuracy
                logger.info(f"📊 {datasource}: {accuracy:.2%} ({int(metrics['pass1'])}/{metrics['count']})")
            if mismatch_samples:
                logger.info("🔎 Sample GSM8K/MATH mismatches (up to 5):")
                for sample in mismatch_samples:
                    logger.info(
                        "idx=%s datasource=%s label=%s response=%s",
                        sample.get("idx"),
                        sample.get("datasource"),
                        sample.get("label"),
                        (sample.get("response") or "")[:500],
                    )

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ GSM8K/MATH evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")


    def evaluate_downstream_humaneval(self, eval_dataloader, global_step):
        """Evaluate HumanEval via unit tests."""
        start_time = time.time()
        logger.info(f"⏰ HumanEval evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self._broadcast_to_vllm()

        n_samples_per_prompt = max(1, getattr(self.args, "eval_n_samples_per_prompt", 1))
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            stop=[self.tokenizer.eos_token],
            n=n_samples_per_prompt,
        )

        with torch.no_grad():
            all_prompts = []
            all_unit_tests = []
            all_entry_points = []

            for batch in eval_dataloader:
                if isinstance(batch, dict):
                    batch_items = [batch]
                elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
                    batch_items = batch
                else:
                    raise ValueError(f"Unsupported batch format for HumanEval: {type(batch)}")

                for item in batch_items:
                    prompt = item.get("prompt", "")
                    test_code = item.get("unit_test", "")
                    entry_point = item.get("entry_point")
                    if entry_point:
                        unit_tests = [test_code, f"check({entry_point})"]
                    else:
                        unit_tests = [test_code]
                    all_prompts.append(prompt)
                    all_unit_tests.append(unit_tests)
                    all_entry_points.append(entry_point)

            if not all_prompts:
                logger.warning("No prompts collected for HumanEval; skipping.")
                return
            logger.info("HumanEval evaluation examples: %d", len(all_prompts))

            logger.info(f"🚀 Generating {len(all_prompts)} HumanEval solutions with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)

            rewards_per_prompt = []
            error_counts = Counter()

            for idx, (result, unit_tests, prompt, entry_point) in enumerate(zip(
                results, all_unit_tests, all_prompts, all_entry_points
            )):
                outputs = [output.text for output in result.outputs] if result.outputs else [""]
                if len(outputs) < n_samples_per_prompt:
                    outputs.extend([""] * (n_samples_per_prompt - len(outputs)))
                outputs = outputs[:n_samples_per_prompt]

                if idx < 20:
                    logger.info(
                        "📝 HumanEval sample %d\nPrompt:\n%s\nOutput[0]:\n%s",
                        idx,
                        prompt[:2000],
                        outputs[0][:2000],
                    )

                prompt_rewards = []
                for response in outputs:
                    if idx < 20:
                        logger.info("🧾 HumanEval raw generation:\n%s", (response or "")[:2000])
                    cleaned_response = _sanitize_generated_code(response or "")
                    full_code = prompt.rstrip() + "\n" + cleaned_response
                    is_correct, error_type = self._execute_and_test_code(full_code, unit_tests)
                    reward = 1.0 if is_correct else 0.0
                    prompt_rewards.append(reward)
                    if not is_correct:
                        error_counts[error_type or "unknown"] += 1
                        if idx < 3:
                            logger.info(
                                "🧪 HumanEval combined code (sample %d, error=%s)\n%s",
                                idx,
                                error_type,
                                full_code[:2000],
                            )
                rewards_per_prompt.append(prompt_rewards)

            rewards = torch.tensor(rewards_per_prompt)
            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_attempts = rewards.numel()
            if total_attempts > 0:
                syntax_pct = error_counts.get("syntax", 0) / total_attempts
                fail_pct = error_counts.get("test_failure", 0) / total_attempts
                timeout_pct = error_counts.get("timeout", 0) / total_attempts
            else:
                syntax_pct = fail_pct = timeout_pct = 0.0

            logs = {
                "reward_humaneval_passk": passk,
                "reward_humaneval_pass1": pass1,
                "err_humaneval_syntax_pct": syntax_pct,
                "err_humaneval_fail_pct": fail_pct,
                "err_humaneval_timeout_pct": timeout_pct,
            }

            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ HumanEval evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def evaluate_downstream_mbpp(self, eval_dataloader, global_step):
        """Evaluate MBPP via unit tests."""
        start_time = time.time()
        logger.info(f"⏰ MBPP evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self._broadcast_to_vllm()

        prompt_style = getattr(self.args, "mbpp_prompt_style", "bigcode")
        logger.info("MBPP prompt style: %s", prompt_style)

        stop_sequences = [self.tokenizer.eos_token]
        if prompt_style == "bigcode":
            # Mirror BigCode harness stop words to prevent extra defs/tests.
            stop_sequences.extend(
                [
                    "\n\ndef ",
                    "\nclass",
                    "\nassert",
                    '\n"""',
                    "\nprint",
                    "\nif",
                    "\n<|/",
                    "\n```",
                ]
            )
        stop_sequences = [s for s in stop_sequences if s]

        n_samples_per_prompt = max(1, getattr(self.args, "eval_n_samples_per_prompt", 1))
        sampling_params = SamplingParams(
            # Use deterministic decoding for pass@1 evaluation
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            stop=stop_sequences,
            n=n_samples_per_prompt,
        )
        logger.info(
            "MBPP sampling params: temperature=%s top_p=%s max_tokens=%s n=%s",
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.max_tokens,
            sampling_params.n,
        )

        with torch.no_grad():
            all_prompts = []
            all_unit_tests = []
            all_helper_codes = []
            all_function_names = []
            all_function_signatures = []

            for batch in eval_dataloader:
                if isinstance(batch, dict):
                    batch_items = [batch]
                elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
                    batch_items = batch
                else:
                    raise ValueError(f"Unsupported batch format for MBPP: {type(batch)}")

                for item in batch_items:
                    prompt = item.get("prompt", "")
                    if MBPP_FEWSHOT_PREFIX not in prompt:
                        prompt = MBPP_FEWSHOT_PREFIX + "\n\n" + prompt
                    all_prompts.append(prompt)
                    all_unit_tests.append(item.get("unit_test", item.get("unit_tests", [])))
                    code_context = item.get("code_context", {}) or {}
                    all_helper_codes.append(code_context.get("helper_code", ""))
                    all_function_names.append(code_context.get("function_name"))
                    all_function_signatures.append(code_context.get("function_signature"))

            if not all_prompts:
                logger.warning("No prompts collected for MBPP; skipping.")
                return
            logger.info("MBPP evaluation examples: %d", len(all_prompts))

            logger.info(f"🚀 Generating {len(all_prompts)} MBPP solutions with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)

            rewards_per_prompt = []
            error_counts = Counter()
            response_stats = Counter()

            for idx, (result, unit_tests, prompt, helper_code, function_name, function_signature) in enumerate(zip(
                results,
                all_unit_tests,
                all_prompts,
                all_helper_codes,
                all_function_names,
                all_function_signatures,
            )):
                outputs = [output.text for output in result.outputs] if result.outputs else [""]
                if len(outputs) < n_samples_per_prompt:
                    outputs.extend([""] * (n_samples_per_prompt - len(outputs)))
                outputs = outputs[:n_samples_per_prompt]

                logger.info("🧾 MBPP sample %d", idx)
                logger.info("MBPP function_name: %s", function_name)
                logger.info("MBPP helper_code:\n%s", helper_code)
                logger.info("MBPP function_signature: %s", function_signature)
                logger.info("MBPP unit_tests (%s):\n%s", type(unit_tests).__name__, unit_tests)
                logger.info("MBPP prompt:\n%s", prompt)
                logger.info("MBPP raw outputs (%d):\n%s", len(outputs), outputs)

                prompt_rewards = []
                for response in outputs:
                    logger.info("🧾 MBPP raw generation:\n%s", response or "")
                    keep_leading_def = True if prompt_style == "bigcode" else None
                    raw_response = response or ""
                    if not raw_response.strip():
                        response_stats["resp_empty"] += 1
                    raw_lstrip = raw_response.lstrip()
                    if raw_lstrip.startswith(('"""', "'''")):
                        response_stats["resp_starts_docstring"] += 1
                    if re.search(r"^\s*def\s+\w+", raw_response, re.M):
                        response_stats["resp_has_def"] += 1
                    if re.search(r"^\s*class\s+\w+", raw_response, re.M):
                        response_stats["resp_has_class"] += 1
                    cleaned_preview = _sanitize_generated_code(
                        raw_response, keep_leading_def=keep_leading_def
                    )
                    if not cleaned_preview.strip():
                        response_stats["cleaned_empty"] += 1
                    if re.search(r"^\s*def\s+\w+", cleaned_preview, re.M):
                        response_stats["cleaned_has_def"] += 1
                    else:
                        response_stats["cleaned_no_def"] += 1
                    full_code = _build_mbpp_code(
                        prompt,
                        raw_response,
                        function_name,
                        helper_code,
                        function_signature=function_signature,
                        keep_leading_def=keep_leading_def,
                    )
                    logger.info(
                        "🧹 MBPP prepared code (len=%d):\n%s",
                        len(full_code),
                        full_code,
                    )
                    is_correct, error_type = self._execute_and_test_code(full_code, unit_tests)
                    reward = 1.0 if is_correct else 0.0
                    logger.info(
                        "✅ MBPP execution result: correct=%s error_type=%s",
                        is_correct,
                        error_type,
                    )
                    prompt_rewards.append(reward)
                    if not is_correct:
                        error_counts[error_type or "unknown"] += 1
                        if error_type == "test_failure" and idx < 5:
                            preview_tests = []
                            if isinstance(unit_tests, (list, tuple)):
                                preview_tests = [str(t) for t in unit_tests[:2]]
                            else:
                                preview_tests = [str(unit_tests)]
                            logger.info(
                                "🧪 MBPP failure (sample %d): func=%s tests=%d preview=%s",
                                idx,
                                function_name,
                                len(unit_tests) if isinstance(unit_tests, (list, tuple)) else 1,
                                preview_tests,
                            )
                        if idx < 3:
                            logger.info(
                                "🧪 MBPP combined code (sample %d, error=%s)\n%s",
                                idx,
                                error_type,
                                full_code[:2000],
                            )
                rewards_per_prompt.append(prompt_rewards)

            rewards = torch.tensor(rewards_per_prompt)
            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_attempts = rewards.numel()
            if total_attempts > 0:
                syntax_pct = error_counts.get("syntax", 0) / total_attempts
                fail_pct = error_counts.get("test_failure", 0) / total_attempts
                timeout_pct = error_counts.get("timeout", 0) / total_attempts
            else:
                syntax_pct = fail_pct = timeout_pct = 0.0

            logs = {
                "reward_mbpp_passk": passk,
                "reward_mbpp_pass1": pass1,
                "err_mbpp_syntax_pct": syntax_pct,
                "err_mbpp_fail_pct": fail_pct,
                "err_mbpp_timeout_pct": timeout_pct,
            }
            logger.info("MBPP response stats: %s", dict(response_stats))

            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ MBPP evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def evaluate_downstream_opencode(self, eval_dataloader, global_step, metric_prefix="opencode"):
        """Evaluate model performance on downstream code datasets via unit tests."""

        start_time = time.time()
        logger.info(f"⏰ OpenCode evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Ensure vLLM is synchronized with the latest model weights
        self._broadcast_to_vllm()

        n_samples_per_prompt = max(1, getattr(self.args, "eval_n_samples_per_prompt", 1))
        sampling_params = SamplingParams(
            temperature=0.6, #if self.temperature == 0 else self.temperature,
            top_p=0.95,#1.0 if self.temperature == 0 else self.top_p,
            max_tokens=512,
            stop=[self.tokenizer.eos_token],
            n=n_samples_per_prompt,
        )

        with torch.no_grad():
            all_prompts = []
            all_unit_tests = []

            logger.info("📦 Collecting evaluation data for code tasks...")
            for batch in eval_dataloader:
                if isinstance(batch, dict):
                    batch_items = [batch]
                elif isinstance(batch, (list, tuple)):
                    if len(batch) > 0 and isinstance(batch[0], dict):
                        batch_items = batch
                    elif len(batch) == 4:
                        # Backward compatibility for (datasources, prompts, labels, unit_tests)
                        _, prompts, _, unit_tests = batch
                        all_prompts.extend(prompts)
                        all_unit_tests.extend(unit_tests)
                        continue
                    else:
                        raise ValueError(f"Unsupported batch format for code evaluation: {type(batch)}")
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                all_prompts.extend(item.get("prompt", "") for item in batch_items)
                all_unit_tests.extend(
                    item.get("unit_test", item.get("unit_tests", [])) for item in batch_items
                )

            if not all_prompts:
                logger.warning("No prompts collected for code evaluation; skipping OpenCode metrics.")
                return
            logger.info("OpenCode evaluation examples: %d", len(all_prompts))

            logger.info(f"🚀 Generating {len(all_prompts)} code solutions with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)
           
            
            

            rewards_per_prompt = []
            error_counts = Counter()

            logger.info("🧪 Executing generated code against unit tests...")
            for idx, (result, unit_tests) in enumerate(zip(results, all_unit_tests)):
                outputs = [output.text for output in result.outputs] if result.outputs else [""]

                logger.info(f"ALL RESPONSES: {outputs}; LENALLREP: {len(outputs)}")

                
                if len(outputs) < n_samples_per_prompt:
                    outputs.extend([""] * (n_samples_per_prompt - len(outputs)))
                outputs = outputs[:n_samples_per_prompt]

                prompt_rewards = []
                for response in outputs:
                    code = self._extract_code_from_response(response)
                    is_correct, error_type = self._execute_and_test_code(code, unit_tests)
                    reward = 1.0 if is_correct else 0.0
                    prompt_rewards.append(reward)
                    if not is_correct:
                        error_counts[error_type or "unknown"] += 1
                rewards_per_prompt.append(prompt_rewards)

            rewards = torch.tensor(rewards_per_prompt)
            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_attempts = rewards.numel()
            if total_attempts > 0:
                syntax_pct = error_counts.get("syntax", 0) / total_attempts
                fail_pct = error_counts.get("test_failure", 0) / total_attempts
                timeout_pct = error_counts.get("timeout", 0) / total_attempts
            else:
                syntax_pct = fail_pct = timeout_pct = 0.0

            logs = {
                f"reward_{metric_prefix}_passk": passk,
                f"reward_{metric_prefix}_pass1": pass1,
                f"err_{metric_prefix}_syntax_pct": syntax_pct,
                f"err_{metric_prefix}_fail_pct": fail_pct,
                f"err_{metric_prefix}_timeout_pct": timeout_pct,
            }

            if self._wandb is not None:
                logs_with_step = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs_with_step)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ OpenCode evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def evaluate_downstream_mmlu(self, eval_dataloader, global_step):
        """Evaluate MMLU with Brier score using option perplexity."""
        start_time = time.time()
        logger.info(f"⏰ MMLU evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.model.eval()
        choices = ["A", "B", "C", "D"]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = getattr(self.args, "max_len", None)
        prompt_batch_size = max(1, getattr(self.args, "eval_mc_batch_size", 8))

        def normalize_label(label):
            if label is None:
                return None
            label_str = str(label).strip()
            if not label_str:
                return None
            upper = label_str.upper()
            for choice in choices:
                if f"{choice}" in upper:
                    return choice
            if upper.isdigit():
                idx = int(upper)
                if 0 <= idx < len(choices):
                    return choices[idx]
            return upper[0] if upper else None

        def build_choice_inputs(prompt, choice):
            sep = "" if prompt.endswith((" ", "\n")) else " "
            full_text = f"{prompt}{sep}{choice}"

            encoding = None
            offsets = None
            if getattr(self.tokenizer, "is_fast", False):
                try:
                    encoding = self.tokenizer(
                        full_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    offsets = encoding.get("offset_mapping")
                except Exception:
                    encoding = None
                    offsets = None

            if encoding is None:
                encoding = self.tokenizer(full_text, add_special_tokens=False)

            input_ids = encoding["input_ids"]
            if not input_ids:
                return None, None

            if offsets is not None:
                # Use offsets to robustly locate the option span (avoids leading-space trap).
                option_start = len(prompt + sep)
                option_end = option_start + len(choice)
                token_mask = [0] * len(input_ids)
                for idx, (start, end) in enumerate(offsets):
                    if end <= option_start or start >= option_end:
                        continue
                    token_mask[idx] = 1
            else:
                # Fallback: assume prompt tokenization is a prefix of the full text.
                prompt_ids = self.tokenizer(prompt + sep, add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_ids)
                if input_ids[:prompt_len] != prompt_ids:
                    logger.warning(
                        "MMLU eval: prompt tokenization is not a prefix of the full text; "
                        "option masking may be misaligned. Consider using a fast tokenizer."
                    )
                token_mask = [0] * len(input_ids)
                for idx in range(prompt_len, len(input_ids)):
                    token_mask[idx] = 1

            if max_len and len(input_ids) > max_len:
                option_indices = [idx for idx, flag in enumerate(token_mask) if flag]
                if not option_indices:
                    return None, None
                option_start_idx = option_indices[0]
                option_end_idx = option_indices[-1] + 1
                if (option_end_idx - option_start_idx) >= max_len:
                    start_idx = option_end_idx - max_len
                else:
                    start_idx = max(0, option_end_idx - max_len)
                end_idx = start_idx + max_len
                input_ids = input_ids[start_idx:end_idx]
                token_mask = token_mask[start_idx:end_idx]

            if len(input_ids) < 2:
                return None, None

            loss_mask = torch.tensor(token_mask[1:], dtype=torch.float32)
            if loss_mask.sum().item() == 0:
                return None, None
            return input_ids, loss_mask

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            if not all_prompts:
                logger.warning("No prompts collected for MMLU evaluation; skipping.")
                return
            logger.info("MMLU evaluation examples: %d", len(all_prompts))

            total_brier = 0.0
            total_logprob = 0.0
            total_count = 0
            global_metrics = {}

            for start in range(0, len(all_prompts), prompt_batch_size):
                batch_prompts = all_prompts[start : start + prompt_batch_size]
                batch_labels = all_labels[start : start + prompt_batch_size]

                sequences = []
                attention_masks = []
                loss_masks = []

                for prompt in batch_prompts:
                    for choice in choices:
                        input_ids, loss_mask = build_choice_inputs(prompt, choice)
                        if input_ids is None:
                            continue
                        seq_len = len(input_ids)
                        sequences.append(torch.tensor(input_ids, dtype=torch.long))
                        attention_masks.append(torch.ones(seq_len, dtype=torch.long))
                        loss_masks.append(loss_mask)

                if not sequences:
                    continue

                input_ids = zero_pad_sequences(sequences, "right", pad_token_id, stack=True)
                attention_mask = zero_pad_sequences(attention_masks, "right", 0, stack=True)
                loss_mask = zero_pad_sequences(loss_masks, "right", 0, stack=True)

                input_ids = input_ids.to(torch.cuda.current_device())
                attention_mask = attention_mask.to(torch.cuda.current_device())
                loss_mask = loss_mask.to(torch.cuda.current_device())

                per_token_logps = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                denom = loss_mask.sum(dim=1).clamp(min=1.0)
                losses = -(per_token_logps * loss_mask).sum(dim=1) / denom
                losses = losses.view(len(batch_prompts), len(choices)).cpu()
                probs = torch.softmax(-losses, dim=1)

                for idx, (prompt, label) in enumerate(zip(batch_prompts, batch_labels)):
                    normalized = normalize_label(label)
                    if normalized not in choices:
                        continue
                    correct_idx = choices.index(normalized)
                    target = torch.zeros(len(choices))
                    target[correct_idx] = 1.0
                    brier = torch.mean((probs[idx] - target) ** 2).item()
                    correct_logprob = (-losses[idx][correct_idx]).item()
                    total_count += 1
                    total_brier += brier
                    total_logprob += correct_logprob

                    datasource = prompt_to_datasource.get(prompt, "default")
                    if datasource not in global_metrics:
                        global_metrics[datasource] = {
                            "brier_sum": 0.0,
                            "logprob_sum": 0.0,
                            "count": 0,
                        }
                    global_metrics[datasource]["count"] += 1
                    global_metrics[datasource]["brier_sum"] += brier
                    global_metrics[datasource]["logprob_sum"] += correct_logprob

            if total_count == 0:
                logger.warning("No valid labels for MMLU evaluation; skipping metrics.")
                return

            logs = {
                "eval_mmlu_brier": total_brier / total_count,
                "eval_mmlu_avg_logprob": total_logprob / total_count,
            }
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_brier"] = metrics["brier_sum"] / metrics["count"]
                logs[f"eval_{datasource}_avg_logprob"] = (
                    metrics["logprob_sum"] / metrics["count"]
                )

            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        self.model.train()
        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ MMLU evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def evaluate_downstream_arc(self, eval_dataloader, global_step, metric_prefix="arc"):
        """Evaluate ARC accuracy using option perplexity."""
        start_time = time.time()
        logger.info(f"⏰ ARC evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.model.eval()
        choices = ["A", "B", "C", "D"]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = getattr(self.args, "max_len", None)
        prompt_batch_size = max(1, getattr(self.args, "eval_mc_batch_size", 8))

        def normalize_label(label):
            if label is None:
                return None
            label_str = str(label).strip()
            if not label_str:
                return None
            upper = label_str.upper()
            for choice in choices:
                if f"{choice}" in upper:
                    return choice
            if upper.isdigit():
                idx = int(upper)
                if 0 <= idx < len(choices):
                    return choices[idx]
            return upper[0] if upper else None

        def build_choice_inputs(prompt, choice):
            sep = "" if prompt.endswith((" ", "\n")) else " "
            full_text = f"{prompt}{sep}{choice}"

            encoding = None
            offsets = None
            if getattr(self.tokenizer, "is_fast", False):
                try:
                    encoding = self.tokenizer(
                        full_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    offsets = encoding.get("offset_mapping")
                except Exception:
                    encoding = None
                    offsets = None

            if encoding is None:
                encoding = self.tokenizer(full_text, add_special_tokens=False)

            input_ids = encoding["input_ids"]
            if not input_ids:
                return None, None

            if offsets is not None:
                # Use offsets to robustly locate the option span (avoids leading-space trap).
                option_start = len(prompt + sep)
                option_end = option_start + len(choice)
                token_mask = [0] * len(input_ids)
                for idx, (start, end) in enumerate(offsets):
                    if end <= option_start or start >= option_end:
                        continue
                    token_mask[idx] = 1
            else:
                # Fallback: assume prompt tokenization is a prefix of the full text.
                prompt_ids = self.tokenizer(prompt + sep, add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_ids)
                if input_ids[:prompt_len] != prompt_ids:
                    logger.warning(
                        "ARC eval: prompt tokenization is not a prefix of the full text; "
                        "option masking may be misaligned. Consider using a fast tokenizer."
                    )
                token_mask = [0] * len(input_ids)
                for idx in range(prompt_len, len(input_ids)):
                    token_mask[idx] = 1

            if max_len and len(input_ids) > max_len:
                option_indices = [idx for idx, flag in enumerate(token_mask) if flag]
                if not option_indices:
                    return None, None
                option_start_idx = option_indices[0]
                option_end_idx = option_indices[-1] + 1
                if (option_end_idx - option_start_idx) >= max_len:
                    start_idx = option_end_idx - max_len
                else:
                    start_idx = max(0, option_end_idx - max_len)
                end_idx = start_idx + max_len
                input_ids = input_ids[start_idx:end_idx]
                token_mask = token_mask[start_idx:end_idx]

            if len(input_ids) < 2:
                return None, None

            loss_mask = torch.tensor(token_mask[1:], dtype=torch.float32)
            if loss_mask.sum().item() == 0:
                return None, None
            return input_ids, loss_mask

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            if not all_prompts:
                logger.warning("No prompts collected for ARC evaluation; skipping.")
                return
            logger.info("ARC evaluation examples (%s): %d", metric_prefix, len(all_prompts))

            total_correct = 0
            total_count = 0
            global_metrics = {}

            for start in range(0, len(all_prompts), prompt_batch_size):
                batch_prompts = all_prompts[start : start + prompt_batch_size]
                batch_labels = all_labels[start : start + prompt_batch_size]

                sequences = []
                attention_masks = []
                loss_masks = []

                for prompt in batch_prompts:
                    for choice in choices:
                        input_ids, loss_mask = build_choice_inputs(prompt, choice)
                        if input_ids is None:
                            continue
                        seq_len = len(input_ids)
                        sequences.append(torch.tensor(input_ids, dtype=torch.long))
                        attention_masks.append(torch.ones(seq_len, dtype=torch.long))
                        loss_masks.append(loss_mask)

                if not sequences:
                    continue

                input_ids = zero_pad_sequences(sequences, "right", pad_token_id, stack=True)
                attention_mask = zero_pad_sequences(attention_masks, "right", 0, stack=True)
                loss_mask = zero_pad_sequences(loss_masks, "right", 0, stack=True)

                input_ids = input_ids.to(torch.cuda.current_device())
                attention_mask = attention_mask.to(torch.cuda.current_device())
                loss_mask = loss_mask.to(torch.cuda.current_device())

                per_token_logps = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                denom = loss_mask.sum(dim=1).clamp(min=1.0)
                losses = -(per_token_logps * loss_mask).sum(dim=1) / denom
                losses = losses.view(len(batch_prompts), len(choices)).cpu()

                for idx, (prompt, label) in enumerate(zip(batch_prompts, batch_labels)):
                    normalized = normalize_label(label)
                    if normalized not in choices:
                        continue
                    best_choice = choices[int(torch.argmin(losses[idx]).item())]
                    is_correct = best_choice == normalized
                    total_count += 1
                    total_correct += 1 if is_correct else 0

                    datasource = prompt_to_datasource.get(prompt, "default")
                    if datasource not in global_metrics:
                        global_metrics[datasource] = {"correct": 0, "count": 0}
                    global_metrics[datasource]["count"] += 1
                    global_metrics[datasource]["correct"] += 1 if is_correct else 0

            if total_count == 0:
                logger.warning("No valid labels for ARC evaluation; skipping metrics.")
                return

            logs = {f"eval_{metric_prefix}_acc": total_correct / total_count}
            # Skip per-datasource logging to avoid duplicate ARC metrics.

            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        self.model.train()
        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(
            f"✨ ARC ({metric_prefix}) evaluation completed in {time_str}, "
            f"global_step {global_step}, eval_metrics: {logs}"
        )

    def evaluate_downstream_obqa(self, eval_dataloader, global_step):
        """Evaluate OpenBookQA accuracy using option perplexity."""
        start_time = time.time()
        logger.info(f"⏰ OpenBookQA evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.model.eval()
        choices = ["A", "B", "C", "D"]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = getattr(self.args, "max_len", None)
        prompt_batch_size = max(1, getattr(self.args, "eval_mc_batch_size", 8))

        def normalize_label(label):
            if label is None:
                return None
            label_str = str(label).strip()
            if not label_str:
                return None
            upper = label_str.upper()
            for choice in choices:
                if f"{choice}" in upper:
                    return choice
            if upper.isdigit():
                idx = int(upper)
                if 0 <= idx < len(choices):
                    return choices[idx]
            return upper[0] if upper else None

        def build_choice_inputs(prompt, choice):
            sep = "" if prompt.endswith((" ", "\n")) else " "
            full_text = f"{prompt}{sep}{choice}"

            encoding = None
            offsets = None
            if getattr(self.tokenizer, "is_fast", False):
                try:
                    encoding = self.tokenizer(
                        full_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    offsets = encoding.get("offset_mapping")
                except Exception:
                    encoding = None
                    offsets = None

            if encoding is None:
                encoding = self.tokenizer(full_text, add_special_tokens=False)

            input_ids = encoding["input_ids"]
            if not input_ids:
                return None, None

            if offsets is not None:
                # Use offsets to robustly locate the option span (avoids leading-space trap).
                option_start = len(prompt + sep)
                option_end = option_start + len(choice)
                token_mask = [0] * len(input_ids)
                for idx, (start, end) in enumerate(offsets):
                    if end <= option_start or start >= option_end:
                        continue
                    token_mask[idx] = 1
            else:
                # Fallback: assume prompt tokenization is a prefix of the full text.
                prompt_ids = self.tokenizer(prompt + sep, add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_ids)
                if input_ids[:prompt_len] != prompt_ids:
                    logger.warning(
                        "OpenBookQA eval: prompt tokenization is not a prefix of the full text; "
                        "option masking may be misaligned. Consider using a fast tokenizer."
                    )
                token_mask = [0] * len(input_ids)
                for idx in range(prompt_len, len(input_ids)):
                    token_mask[idx] = 1

            if max_len and len(input_ids) > max_len:
                option_indices = [idx for idx, flag in enumerate(token_mask) if flag]
                if not option_indices:
                    return None, None
                option_start_idx = option_indices[0]
                option_end_idx = option_indices[-1] + 1
                if (option_end_idx - option_start_idx) >= max_len:
                    start_idx = option_end_idx - max_len
                else:
                    start_idx = max(0, option_end_idx - max_len)
                end_idx = start_idx + max_len
                input_ids = input_ids[start_idx:end_idx]
                token_mask = token_mask[start_idx:end_idx]

            if len(input_ids) < 2:
                return None, None

            loss_mask = torch.tensor(token_mask[1:], dtype=torch.float32)
            if loss_mask.sum().item() == 0:
                return None, None
            return input_ids, loss_mask

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            if not all_prompts:
                logger.warning("No prompts collected for OpenBookQA evaluation; skipping.")
                return
            logger.info("OpenBookQA evaluation examples: %d", len(all_prompts))

            total_correct = 0
            total_count = 0
            global_metrics = {}

            for start in range(0, len(all_prompts), prompt_batch_size):
                batch_prompts = all_prompts[start : start + prompt_batch_size]
                batch_labels = all_labels[start : start + prompt_batch_size]

                sequences = []
                attention_masks = []
                loss_masks = []

                for prompt in batch_prompts:
                    for choice in choices:
                        input_ids, loss_mask = build_choice_inputs(prompt, choice)
                        if input_ids is None:
                            continue
                        seq_len = len(input_ids)
                        sequences.append(torch.tensor(input_ids, dtype=torch.long))
                        attention_masks.append(torch.ones(seq_len, dtype=torch.long))
                        loss_masks.append(loss_mask)

                if not sequences:
                    continue

                input_ids = zero_pad_sequences(sequences, "right", pad_token_id, stack=True)
                attention_mask = zero_pad_sequences(attention_masks, "right", 0, stack=True)
                loss_mask = zero_pad_sequences(loss_masks, "right", 0, stack=True)

                input_ids = input_ids.to(torch.cuda.current_device())
                attention_mask = attention_mask.to(torch.cuda.current_device())
                loss_mask = loss_mask.to(torch.cuda.current_device())

                per_token_logps = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                denom = loss_mask.sum(dim=1).clamp(min=1.0)
                losses = -(per_token_logps * loss_mask).sum(dim=1) / denom
                losses = losses.view(len(batch_prompts), len(choices)).cpu()

                for idx, (prompt, label) in enumerate(zip(batch_prompts, batch_labels)):
                    normalized = normalize_label(label)
                    if normalized not in choices:
                        continue
                    best_choice = choices[int(torch.argmin(losses[idx]).item())]
                    is_correct = best_choice == normalized
                    total_count += 1
                    total_correct += 1 if is_correct else 0

                    datasource = prompt_to_datasource.get(prompt, "default")
                    if datasource not in global_metrics:
                        global_metrics[datasource] = {"correct": 0, "count": 0}
                    global_metrics[datasource]["count"] += 1
                    global_metrics[datasource]["correct"] += 1 if is_correct else 0

            if total_count == 0:
                logger.warning("No valid labels for OpenBookQA evaluation; skipping metrics.")
                return

            logs = {"eval_obqa_acc": total_correct / total_count}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_acc"] = metrics["correct"] / metrics["count"]

            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        self.model.train()
        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ OpenBookQA evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

       
    def _extract_code_from_response(self, response):
        """Extract Python code from a response that may contain markdown code blocks.

        Handles multiple formats:
        - ```python\ncode\n```
        - ```\ncode\n```
        - Plain code without markdown
        - Code followed by test statements (strips test code)
        - Code preceded by "Solution:" prefix

        Note: Indentation will be fixed by _fix_code_indentation() after combining with prompt.

        Args:
            response: The generated response text

        Returns:
            str: The extracted Python code (function implementation only, may have indentation issues)
        """
        import re

        if not response:
            return ''

        normalized = response.replace('\r', '')

        # Extract code that comes after "Solution:" if present (handles "Solution:\n..." format anywhere in response)
        solution_pattern = r'Solution\s*:\s*\n?'
        solution_match = re.search(solution_pattern, normalized, flags=re.IGNORECASE)
        if solution_match:
            normalized = normalized[solution_match.end():]

        # Match a fenced code block even if the closing ``` is missing.
        code_block_pattern = r'```(?:[a-zA-Z0-9_+.\-]+)?\s*\n(.*?)(?:```|$)'
        match = re.search(code_block_pattern, normalized, re.DOTALL)

        if match:
            code = match.group(1).strip()
        else:
            # If we saw an opening fence but couldn't match (e.g., malformed markdown),
            # fall back to slicing from the first fence onward.
            if '```' in normalized:
                start = normalized.find('```')
                remainder = normalized[start + 3:]
                remainder = remainder.lstrip()

                # Drop a potential language spec on the first line.
                newline_idx = remainder.find('\n')
                if newline_idx != -1:
                    first_line = remainder[:newline_idx].strip()
                    if first_line and len(first_line) <= 32 and ' ' not in first_line:
                        remainder = remainder[newline_idx + 1 :]

                code = remainder.split('```', 1)[0].strip()
            else:
                # No code fences at all—treat the whole response as code.
                code = normalized.strip()

        # Remove test code that often appears after the function implementation
        # Look for common test patterns and truncate before them
        lines = code.split('\n')
        function_lines = []
        markdown_prefixes = (
            '###',
            '##',
            '**',
            '* ',
            '> ',
            'Implementation',
            'Solution',
            'Answer',
            'Here is',
            'Explanation',
        )

        def is_code_like(raw_line, stripped_line):
            if not stripped_line:
                return False
            if re.match(r"^(def|class)\s+\w+", stripped_line):
                return True
            if re.match(r"^(from|import)\s+\w+", stripped_line):
                return True
            if stripped_line.startswith("@"):
                return True
            if stripped_line.startswith(
                (
                    "return",
                    "if ",
                    "for ",
                    "while ",
                    "try:",
                    "with ",
                    "elif ",
                    "else:",
                    "except",
                    "pass",
                    "raise",
                )
            ):
                return True
            if raw_line.startswith((" ", "\t")):
                return True
            if re.search(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped_line):
                return True
            return False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not function_lines:
                if not stripped:
                    continue
                if stripped in {'`', '``', '```'}:
                    continue
                lowered = stripped.lower()
                if any(lowered.startswith(prefix.lower()) for prefix in markdown_prefixes):
                    continue
                if not is_code_like(line, stripped):
                    continue

            # Stop at common test indicators and example usage
            if (
                stripped.startswith('print(')
                or stripped.startswith('# Test')
                or stripped.startswith('# Example')
                or stripped.startswith('# ---')
                or stripped.startswith('if __name__')
                or (stripped.startswith('assert ') and i > 0)
                # Stop at module-level variable assignments that look like test setup
                # (variable = ClassName(...) at column 0)
                or (not line.startswith(' ') and not line.startswith('\t')
                    and re.match(r'^[a-z_][a-z0-9_]*\s*=\s*[A-Z]', stripped)
                    and function_lines)  # Only after we've collected some code
            ):
                break

            function_lines.append(line)

        # Remove trailing empty lines
        while function_lines and not function_lines[-1].strip():
            function_lines.pop()

        # Return with a leading newline to separate from docstring
        if function_lines:
            return '\n' + '\n'.join(function_lines)
        return ''

    def _combine_prompt_and_completion(
        self,
        prompt,
        completion,
        expected_function_name=None,
        helper_code=None,
        return_debug=False,
    ):
        """Combine a prompt (with def signature) and a model completion safely.

        If the completion already defines a top-level function, use it as-is.
        Otherwise, append the completion as the function body, indenting if needed.
        """
        import re

        def extract_body_from_def(text):
            lines = text.splitlines()
            def_idx = None
            def_indent = 0
            for idx, line in enumerate(lines):
                if re.match(r"^\s*def\s+\w+\b", line):
                    def_idx = idx
                    def_indent = len(line) - len(line.lstrip())
                    break
            if def_idx is None:
                return ""

            body_lines = []
            for line in lines[def_idx + 1 :]:
                if not line.strip():
                    body_lines.append("")
                    continue
                indent = len(line) - len(line.lstrip())
                if indent <= def_indent:
                    break
                body_lines.append(line)

            if not body_lines:
                return ""

            base_indent = None
            for line in body_lines:
                if line.strip():
                    base_indent = len(line) - len(line.lstrip())
                    break
            if base_indent is None:
                return ""

            trimmed = []
            for line in body_lines:
                if not line.strip():
                    trimmed.append("")
                else:
                    trimmed.append(line[base_indent:])
            return "\n".join(trimmed).strip("\n")

        def extract_import_preamble(text):
            preamble_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    break
                if not stripped:
                    preamble_lines.append(line)
                    continue
                if stripped.startswith("from __future__ import"):
                    preamble_lines.append(line)
                    continue
                if stripped.startswith("import ") or stripped.startswith("from "):
                    preamble_lines.append(line)
                    continue
            preamble = "\n".join(preamble_lines).strip("\n")
            return preamble

        def prompt_has_body(text):
            lines = text.splitlines()
            def_idx = None
            for idx, line in enumerate(lines):
                if line.strip().startswith("def "):
                    def_idx = idx
                    break
            if def_idx is None:
                return False
            for line in lines[def_idx + 1 :]:
                if line.strip() and line.startswith((" ", "\t")):
                    return True
            return False

        def normalize_body(text):
            lines = text.splitlines()
            # Trim leading/trailing empty lines
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                return ""

            expanded = [line.expandtabs(4) for line in lines]
            indents = [
                len(line) - len(line.lstrip()) for line in expanded if line.strip()
            ]
            base_indent = min(indents) if indents else 0
            relative = [
                line[base_indent:] if len(line) >= base_indent else line.lstrip()
                for line in expanded
            ]

            normalized = []
            for line in relative:
                stripped = line.rstrip()
                if not stripped.strip():
                    normalized.append("")
                    continue
                normalized.append("    " + stripped)

            return "\n".join(normalized)

        debug_info = {}
        completion = completion or ""
        debug_info["raw_completion"] = completion
        stripped_completion = completion.lstrip("\n")
        if stripped_completion:
            if expected_function_name and re.search(
                rf"^\s*def\s+{re.escape(expected_function_name)}\b", stripped_completion, re.M
            ):
                preamble = extract_import_preamble(prompt)
                if preamble:
                    combined = preamble + "\n\n" + stripped_completion
                else:
                    combined = stripped_completion
            else:
                if re.search(r"^\s*def\s+\w+\b", stripped_completion, re.M):
                    stripped_completion = extract_body_from_def(stripped_completion)

                if stripped_completion:
                    debug_info["extracted_code"] = stripped_completion
                    normalized_body = normalize_body(stripped_completion)
                    debug_info["normalized_body"] = normalized_body
                    combined = prompt.rstrip() + "\n" + normalized_body
                else:
                    combined = prompt.rstrip()
        else:
            combined = prompt.rstrip()

        if helper_code:
            helper_code = helper_code.rstrip()
            if helper_code:
                combined = helper_code + "\n\n" + combined
        if combined == prompt.rstrip() and not prompt_has_body(prompt):
            combined = combined + "\n    pass"
        if return_debug:
            return combined, debug_info
        return combined

    def _execute_and_test_code(self, code, unit_tests, timeout=3):
        """Execute generated code and run provided unit tests in an isolated subprocess.

        This method runs code in a separate process to protect against:
        - sys.exit() calls (e.g., from argparse errors)
        - os._exit() calls
        - Segmentation faults
        - Any other process-terminating operations

        Args:
            code: The generated code to execute
            unit_tests: Unit tests to run (list or string)
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success: bool, error_type: str or None)
        """
        # Parse unit_tests if it's a string
        if isinstance(unit_tests, str):
            import ast
            try:
                unit_tests = ast.literal_eval(unit_tests)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Failed to parse unit_tests: {exc}") from exc

        # Use the subprocess-based execution for safety
        return _run_code_in_subprocess(code, unit_tests, timeout)
      

    def evaluate_translation(self, eval_dataloader, global_step):
        """Evaluate model performance on eval dataset with translation quality metrics.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            temperature: Temperature for sampling
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """

        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self._broadcast_to_vllm()

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            logger.info("Translation evaluation examples: %d", len(all_prompts))
            # Generate samples
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=0.0,#self.temperature,  # Use the passed temperature parameter
                top_p=1.0,#self.top_p,
                max_tokens=400,#self.max_tokens,
                stop=[self.tokenizer.eos_token],
                n=1,
            )
            logger.info(f"ALL PROMPTS: {all_prompts[:5]}; LENALLPR: {len(all_prompts)}")


            # Generate responses using vLLM
            logger.info(f"Generating {len(all_prompts)} responses with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)


            all_responses = [result.outputs[0].text for result in results]
            logger.info(f"ALL RESPONSES: {all_responses[:5]}; LENALLREP: {len(all_responses)}")

            # Extract sources from prompts for COMET
            all_sources = [prompt.split(': ', 1)[1].split('\n')[0].strip() for prompt in all_prompts]

            all_responses = [resp.split('\n')[0].strip() for resp in all_responses]
            # ewrewewewr



            # Evaluate translations using batched scoring for efficiency
            logger.info("Computing translation quality metrics in batches...")
            labels= [str(label) for label in all_labels]
            # for i in range(len(all_responses)):
            #     print(f"SOURCES{i}: {[all_sources[i]]};  HYPS{i}: {[all_responses[i]]}; LBL{i}: {labels[i]}")
            # print(f"SOURCES1: {[all_sources[1]]};  HYPS1: {[all_responses[1]]}; LBL1: {labels[1]}")

            # Check if we should compute gamma for median criterion bandwidth selection
            # Pass --compute_gamma flag to enable this (will compute and exit)
            # compute_gamma = True
            # print(f"ALL RESPONSES: {len(all_responses)}")
            # qwrwqqw
            batch_scores = self.score_translations_batched(
                hypotheses=all_responses,
                references=labels,
                sources=all_sources,
                batch_size=128,  # Adjust batch size based on GPU memory
                # compute_gamma=compute_gamma  # If True, computes gamma and exits
            )
            # wt4t4t4

            all_bleu_scores = batch_scores['bleu']
            all_comet_scores = batch_scores['comet']
            all_semantic_sim_scores = batch_scores['semantic_sim']

            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            n_samples_per_prompt = 1
            bleu_scores_tensor = torch.tensor(all_bleu_scores).reshape(-1, n_samples_per_prompt)
            comet_scores_tensor = torch.tensor(all_comet_scores).reshape(-1, n_samples_per_prompt)
            semantic_sim_scores_tensor = torch.tensor(all_semantic_sim_scores).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"combined": 0, "bleu": 0, "comet": 0, "semantic_sim": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {
                        "bleu": 0,
                        "comet": 0,
                        "semantic_sim": 0,
                        "count": 0
                    }

                # Get scores for this chunk
                bleu_score = bleu_scores_tensor[i].mean().float().item()
                comet_score = comet_scores_tensor[i].mean().float().item()
                semantic_sim_score = semantic_sim_scores_tensor[i].mean().float().item()

                # Accumulate scores
                global_metrics[datasource]["bleu"] += bleu_score
                global_metrics[datasource]["comet"] += comet_score
                global_metrics[datasource]["semantic_sim"] += semantic_sim_score
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                count = metrics["count"]
                logs[f"eval_{datasource}_bleu"] = metrics["bleu"] / count
                logs[f"eval_{datasource}_comet"] = metrics["comet"] / count
                logs[f"eval_{datasource}_semantic_sim"] = metrics["semantic_sim"] / count

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    
    def evaluate_perplexity(self, eval_perplexity_dataloader, steps):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_tokens = 0
            step_bar = tqdm(
                range(eval_perplexity_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for inputs, attention_masks, loss_masks in eval_perplexity_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                per_token_log_probs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                # Calculate loss for this batch
                loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
                
                # Count the number of tokens we're computing loss over
                num_tokens = loss_mask[:, :-1].sum().item()
                
                # Accumulate weighted loss (loss is already averaged over tokens in the batch)
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Calculate running average
                if total_tokens > 0:
                    avg_loss = total_loss / total_tokens
                    perplexity = math.exp(avg_loss)
                else:
                    raise ValueError("total_tokens = 0 in eval perplexity computation")
                    
                bar_dict = {"eval/gpt_loss": avg_loss, "eval/perplexity": perplexity}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state


    def evaluate_wikilarge(self, eval_dataloader, global_step):
        """Evaluate model performance on eval dataset with translation quality metrics.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            temperature: Temperature for sampling
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """

        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self._broadcast_to_vllm()

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            logger.info("WikiLarge evaluation examples: %d", len(all_prompts))
            # Generate samples
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=0.0,#self.temperature,  # Use the passed temperature parameter
                top_p=1.0,#self.top_p,
                max_tokens=400,#self.max_tokens,
                stop=[self.tokenizer.eos_token],
                n=1,
            )
            logger.info(f"ALL PROMPTS: {all_prompts[:5]}; LENALLPR: {len(all_prompts)}")


            # Generate responses using vLLM
            logger.info(f"Generating {len(all_prompts)} responses with vLLM...")
            results = self.vllm_engine.generate(all_prompts, sampling_params)


            all_responses = [result.outputs[0].text for result in results]
            logger.info(f"ALL RESPONSES: {all_responses[:5]}; LENALLREP: {len(all_responses)}")

            # Extract sources from prompts for COMET
            all_sources = [prompt.split(': ', 1)[1].split('\n')[0].strip() for prompt in all_prompts]

            all_responses = [resp.split('\n')[0].strip() for resp in all_responses]
            # ewrewewewr



            # Evaluate translations using batched scoring for efficiency
            logger.info("Computing translation quality metrics in batches...")
            labels= [str(label) for label in all_labels]
            # for i in range(len(all_responses)):
            #     print(f"SOURCES{i}: {[all_sources[i]]};  HYPS{i}: {[all_responses[i]]}; LBL{i}: {labels[i]}")
            # print(f"SOURCES1: {[all_sources[1]]};  HYPS1: {[all_responses[1]]}; LBL1: {labels[1]}")

            # Check if we should compute gamma for median criterion bandwidth selection
            # Pass --compute_gamma flag to enable this (will compute and exit)
            # compute_gamma = True
            # print(f"ALL RESPONSES: {len(all_responses)}")
            # qwrwqqw
            batch_scores = self.score_translations_batched(
                hypotheses=all_responses,
                references=labels,
                sources=all_sources,
                batch_size=128,  # Adjust batch size based on GPU memory
                # compute_gamma=compute_gamma  # If True, computes gamma and exits
            )
            # wt4t4t4

            all_bleu_scores = batch_scores['bleu']
            all_comet_scores = batch_scores['comet']
            all_semantic_sim_scores = batch_scores['semantic_sim']

            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            n_samples_per_prompt = 1
            bleu_scores_tensor = torch.tensor(all_bleu_scores).reshape(-1, n_samples_per_prompt)
            comet_scores_tensor = torch.tensor(all_comet_scores).reshape(-1, n_samples_per_prompt)
            semantic_sim_scores_tensor = torch.tensor(all_semantic_sim_scores).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"combined": 0, "bleu": 0, "comet": 0, "semantic_sim": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {
                        "bleu": 0,
                        "comet": 0,
                        "semantic_sim": 0,
                        "count": 0
                    }

                # Get scores for this chunk
                bleu_score = bleu_scores_tensor[i].mean().float().item()
                comet_score = comet_scores_tensor[i].mean().float().item()
                semantic_sim_score = semantic_sim_scores_tensor[i].mean().float().item()

                # Accumulate scores
                global_metrics[datasource]["bleu"] += bleu_score
                global_metrics[datasource]["comet"] += comet_score
                global_metrics[datasource]["semantic_sim"] += semantic_sim_score
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                count = metrics["count"]
                logs[f"eval_{datasource}_bleu"] = metrics["bleu"] / count
                logs[f"eval_{datasource}_comet"] = metrics["comet"] / count
                logs[f"eval_{datasource}_semantic_sim"] = metrics["semantic_sim"] / count

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")


    

    def _initialize_vllm(self):
        """Initialize vLLM engine once with the pretrained model."""
        try:
            from vllm import LLM
            
            logger.info("🚀 Initializing vLLM engine for fast evaluation...")
            
            # Initialize vLLM with the pretrained model
            self.vllm_engine = LLM(
                model=getattr(self.args, 'pretrain', None),
                tensor_parallel_size=getattr(self.args, 'vllm_tensor_parallel_size', 1),
                trust_remote_code=True,
                dtype="bfloat16" if getattr(self.args, 'bf16', False) else "float16",
                gpu_memory_utilization=getattr(self.args, 'vllm_gpu_memory_utilization', 0.7),
                max_model_len=getattr(self.args, 'max_seq_len', 512),
                enforce_eager=True,  # Disable CUDA graphs for weight updates
            )
            logger.info("✅ vLLM engine initialized successfully")
            
        except ImportError:
            logger.error("vLLM not installed. Please install vLLM for evaluation.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def _broadcast_to_vllm(self):
        """Create or update vLLM engine with current model weights."""
        logger.info("🔄 Setting up vLLM engine with current model weights...")
        
        # Debug: Check what model we're working with
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        logger.info(f"  Model type: {type(model_to_save)}")
        logger.info(f"  Model class name: {model_to_save.__class__.__name__}")
        
        # The simplest and most reliable approach: always create fresh vLLM with current weights
        try:
            import tempfile

            path_tmp_dir = os.path.join(os.getcwd(), "/n/holylfs06/LABS/sham_lab/Users/mkwun/ebm_openrlhf/wd")
            # path_tmp_dir = "/n/holystore01/LABS/barak_lab/Users/sjelassi/ebm_openrlhf/tmp"
            
            os.makedirs(path_tmp_dir,exist_ok=True)

            
            with tempfile.TemporaryDirectory(dir=path_tmp_dir) as tmpdir:
                # Save current model to temp directory
                self.strategy.save_model(self.model, self.tokenizer, tmpdir)
                logger.info(f"  Saved model to temporary directory: {tmpdir}")
                
                # List files in the temp directory for debugging
                saved_files = os.listdir(tmpdir)
                logger.info(f"  Files saved in temp directory: {saved_files}")
                
                # Check if config.json exists and has correct model type
                config_path = os.path.join(tmpdir, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        logger.info(f"  Model config: architectures={config.get('architectures', 'N/A')}, model_type={config.get('model_type', 'N/A')}")
                
                # Clean up any existing vLLM engine
                if self.vllm_engine is not None:
                    del self.vllm_engine
                    torch.cuda.empty_cache()
                
                logger.info(f"  Creating vLLM engine from {tmpdir}")
                self.vllm_engine = LLM(
                    model=tmpdir,
                    tensor_parallel_size=getattr(self.args, 'vllm_tensor_parallel_size', 1),
                    trust_remote_code=True,
                    dtype="bfloat16" if getattr(self.args, 'bf16', False) else "float16",
                    gpu_memory_utilization=getattr(self.args, 'vllm_gpu_memory_utilization', 0.7),
                    max_model_len=getattr(self.args, 'max_seq_len', 512),
                    enforce_eager=True,
                    disable_log_stats=True,  # Reduce logging noise
                )
                
                logger.info("✅ vLLM engine created with current model weights")
                
        except Exception as e:
            logger.error(f"Failed to update vLLM weights: {e}")
            raise

    def _initialize_translation_metrics(self):
        """Initialize translation evaluation metrics (COMET, semantic similarity model)."""
        self.comet_model = None
        self.semantic_sim_model = None
        self.semantic_sim_tokenizer = None

        model_path = download_model("Unbabel/wmt20-comet-da")
        self.comet_model = load_from_checkpoint(model_path)

        # Use the same encoding approach as launcher.py and ebft_trainer.py
        embedding_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        logger.info(f"🚀 Loading text embedding model: {embedding_model_name}")
        self.semantic_sim_model, self.semantic_sim_tokenizer = get_llm_for_text_embedding(
            embedding_model_name,
            bf16=True
        )
        self.semantic_sim_model.eval()
        self.semantic_sim_model = self.semantic_sim_model.cuda()

        # Initialize RFF transformer for more discriminative similarity
        # Get embedding dimension from the model config
        cfg = AutoConfig.from_pretrained(embedding_model_name)
        embedding_dim = getattr(cfg, "hidden_size", None)
        logger.info(f"🎲 Initializing RFF with embedding_dim={embedding_dim}")

        # Initialize RFF with:
        # - num_features: higher = more discriminative but slower (1024 is a good balance)
        # - gamma: MUCH higher for more discrimination (10.0 is very discriminative)
        # - Higher gamma = narrower RBF kernel = only very similar things get high scores
        self.rff_transform = RandomFourierFeatures(
            input_dim=embedding_dim,
            num_features=1024,
            gamma=0.033902,  # Increased from 1.0 to 10.0 for much higher discrimination
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
 
        

        # # Initialize COMET model if available
        # if COMET_AVAILABLE:
        #     try:
        #         logger.info("🚀 Initializing COMET model for translation evaluation...")
        #         # Use the default COMET model (wmt20-comet-da)
        #         model_path = download_model("Unbabel/wmt20-comet-da")
        #         self.comet_model = load_from_checkpoint(model_path)
        #         logger.info("✅ COMET model initialized successfully")
        #     except Exception as e:
        #         logger.warning(f"Failed to initialize COMET model: {e}")
        #         self.comet_model = None
        # else:
        #     logger.warning("COMET not available. Install with: pip install unbabel-comet")

        # Initialize semantic similarity model if available
        # if SEMANTIC_SIM_AVAILABLE:
        #     try:
        #         logger.info("🚀 Initializing semantic similarity model...")
        #         # Use a multilingual model for semantic similarity
        #         self.semantic_sim_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        #         logger.info("✅ Semantic similarity model initialized successfully")
        #     except Exception as e:
        #         logger.warning(f"Failed to initialize semantic similarity model: {e}")
        #         self.semantic_sim_model = None
        # else:
        #     logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

    def _encode_texts(self, texts, batch_size=64):
        """
        Encode texts using the same approach as launcher.py EBFTRewardModelActor.
        Uses CLS token embeddings (first token) from the transformer model.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding

        Returns:
            torch.Tensor: Embeddings of shape (num_texts, hidden_size)
        """
        device = torch.cuda.current_device()
        all_embeddings = []

        # Replace empty sequences with a single space to ensure valid tokenization
        texts = [text if text and text.strip() else " " for text in texts]

        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize sequences
                input_tokens = self.semantic_sim_tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512  # BERT's max length
                )

                # Move to device while explicitly preserving integer dtypes for index tensors
                input_tokens = {
                    k: v.to(
                        device=device,
                        dtype=torch.long if k in ['input_ids', 'attention_mask', 'token_type_ids'] else v.dtype
                    )
                    for k, v in input_tokens.items()
                }

                # Forward pass - gets CLS token embeddings
                batch_embeddings = self.semantic_sim_model(**input_tokens)
                all_embeddings.append(batch_embeddings.cpu())

        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def compute_median_criterion_and_exit(self, hypotheses, references):
        """
        Compute median criterion for bandwidth selection and exit.

        For each context:
          - Compute embeddings for ground truth and all generated sequences
          - Compute median distance among those embeddings
        Then average across all contexts and compute gamma.

        Args:
            hypotheses: List of hypotheses (one per context)
            references: List of references (one per context)
        """
        import sys

        logger.info("=" * 80)
        logger.info("COMPUTING MEDIAN CRITERION FOR BANDWIDTH SELECTION")
        logger.info("=" * 80)

        num_contexts = len(references)
        logger.info(f"Number of contexts: {num_contexts}")

        median_distances_per_context = []

        for i in range(num_contexts):
            # For this context, gather all sequences: [reference, hypothesis]
            # If you have multiple hypotheses per context, you'd extend this
            context_sentences = [references[i], hypotheses[i]]

            # Encode
            embeddings = self.semantic_sim_model.encode(
                context_sentences,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Compute pairwise distances for this context
            norms_sq = (embeddings ** 2).sum(dim=1, keepdim=True)
            distances_sq = norms_sq + norms_sq.T - 2 * torch.mm(embeddings, embeddings.T)
            distances_sq = torch.clamp(distances_sq, min=0)

            # Get upper triangular (exclude diagonal)
            mask = torch.triu(torch.ones_like(distances_sq), diagonal=1).bool()
            distances_sq_upper = distances_sq[mask]

            if len(distances_sq_upper) > 0:
                # Compute median distance for this context
                median_dist = torch.sqrt(torch.median(distances_sq_upper)).item()
                median_distances_per_context.append(median_dist)

        # Average median distances across all contexts
        avg_median_distance = np.mean(median_distances_per_context)

        # Compute gamma = 1 / (2 * avg_median_distance^2)
        gamma = 1.0 / (2.0 * (avg_median_distance ** 2))

        logger.info("=" * 80)
        logger.info(f"Average median distance across contexts: {avg_median_distance:.6f}")
        logger.info(f"Computed gamma (median criterion): {gamma:.6f}")
        logger.info("")
        logger.info(f"RECOMMENDED GAMMA VALUE: {gamma:.6f}")
        logger.info("")
        logger.info("You can now hardcode this gamma value in _initialize_translation_metrics():")
        logger.info(f"    gamma = {gamma:.6f}")
        logger.info("=" * 80)

        # Exit the run
        logger.info("Exiting run after computing gamma...")
        sys.exit(0)

    def score_translations_batched(self, hypotheses, references, sources=None, batch_size=32): #, compute_gamma=True):
        """
        Score multiple translations using batched inference for efficiency.

        Args:
            hypotheses: List of generated translations
            references: List of reference translations
            sources: List of source texts (optional, required for COMET)
            batch_size: Batch size for COMET and semantic similarity models
            compute_gamma: If True, compute median criterion gamma and exit

        Returns:
            dict: Dictionary containing lists of scores for each metric
        """
        # If requested, compute gamma using median criterion and exit
        # if compute_gamma:
        #     self.compute_median_criterion_and_exit(hypotheses, references)

        num_samples = len(hypotheses)
        bleu_scores = []
        comet_scores = []
        semantic_sim_scores = []

        # Calculate BLEU scores (fast enough to do sequentially)
        logger.info(f"Computing BLEU scores for {num_samples} samples...")
        for hypothesis, reference in zip(hypotheses, references):
            bleu_score = sentence_bleu(hypothesis, [reference]).score / 100.0
            bleu_scores.append(bleu_score)
        logger.info(f"✅ BLEU scores computed")

        # wqrrwqrwq

        # Calculate COMET scores in batches
        # COMET is slower than semantic similarity, use smaller batch size
        logger.info(f"Computing COMET scores for {num_samples} samples...")
        comet_data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        # Use smaller batch size for COMET (it's much slower than semantic similarity)
        comet_batch_size = min(32, batch_size)  # Cap at 32 for COMET
        logger.info(f"Using COMET batch_size={comet_batch_size}")
        comet_output = self.comet_model.predict(comet_data, batch_size=comet_batch_size, gpus=1)
        comet_scores = comet_output.scores
        logger.info(f"✅ COMET scores computed")
        # ewfwefew
        # else:
        #     comet_scores = [0.0] * num_samples

        # Calculate semantic similarity scores in batches
        logger.info(f"Computing semantic similarity scores...")
        # Encode all hypotheses and references using the same approach as launcher.py
        logger.info(f"Encoding {num_samples} hypotheses...")
        hyp_embeddings = self._encode_texts(hypotheses, batch_size=batch_size)
        logger.info(f"Encoding {num_samples} references...")
        ref_embeddings = self._encode_texts(references, batch_size=batch_size)
        # Apply RFF transformation for more discriminative similarity
        logger.info(f"Applying RFF transformation...")
        # print(f"BEFORE  RFF: {hyp_embeddings.shape}; {ref_embeddings.shape}")
        # print(f"BEFORE  NORMRFF: HYP {torch.norm(hyp_embeddings,dim=1)};  REF{torch.norm(ref_embeddings, dim=1)}")
        hyp_embeddings_rff = self.rff_transform.transform(hyp_embeddings)
        ref_embeddings_rff = self.rff_transform.transform(ref_embeddings)
        # hyp_embeddings_rff = hyp_embeddings
        # ref_embeddings_rff = ref_embeddings
        # print(f"AFTER RFF: {hyp_embeddings_rff.shape}; {ref_embeddings_rff.shape}")
        # print(f"AFTER NORMRFF: HYP {torch.norm(hyp_embeddings_rff,dim=1)}; REF {torch.norm(ref_embeddings_rff, dim=1)}")
        # bferregwreg

        norm_gt_embed = torch.norm(ref_embeddings_rff, dim=1)
        norm_gen_embed = torch.norm(hyp_embeddings_rff, dim=1) 

        # print(f"NORM GT EMEBD: {torch.mean(norm_gt_embed)} +- {torch.var(norm_gt_embed)}")
        # print(f"NORM GEN EMEBD: {torch.mean(norm_gen_embed)} +- {torch.var(norm_gen_embed)}") 

        # Calculate cosine similarity in RFF space
        logger.info(f"Computing cosine similarities...")
        semantic_sim_scores = F.cosine_similarity(hyp_embeddings_rff, ref_embeddings_rff, dim=1)#.tolist()
        # print(f"AFTER COSINESIM: {semantic_sim_scores}; CSIM SHAPE: {semantic_sim_scores.shape}")
        logger.info(f"✅ Semantic similarity scores computed")

        semantic_sim_scores = semantic_sim_scores.tolist()
        # print(f"LIST COSINESIM: {len(semantic_sim_scores)}")

        return {
            'bleu': bleu_scores,
            'comet': comet_scores,
            'semantic_sim': semantic_sim_scores
        }

    # def score_translation(self, hypothesis, reference, source=None):
    #     """
    #     Score a translation using multiple metrics: BLEU, COMET, and semantic similarity.

    #     Args:
    #         hypothesis: Generated translation
    #         reference: Reference translation
    #         source: Source text (optional, required for COMET)

    #     Returns:
    #         dict: Dictionary containing scores for each metric
    #     """
    #     scores = {}

    #     # Calculate BLEU score
    #     bleu_score = sentence_bleu(hypothesis, [reference]).score / 100.0  # Normalize to [0, 1]
    #     scores['bleu'] = bleu_score

    #     #     try:

    #     #     except Exception as e:
    #     #         logger.warning(f"Failed to calculate BLEU score: {e}")
    #     #         scores['bleu'] = 0.0
    #     # else:
    #     #     scores['bleu'] = 0.0

    #     data = [{
    #                 "src": source,
    #                 "mt": hypothesis,
    #                 "ref": reference
    #             }]
    #     comet_output = self.comet_model.predict(data, batch_size=1, gpus=1)
    #     scores['comet'] = comet_output.scores[0]

    #     # Calculate COMET score (requires source text)
    #     # if COMET_AVAILABLE and self.comet_model is not None and source is not None:
    #     #     try:
    #     #         # COMET expects a list of dictionaries with src, mt, and ref keys
    #     #         data = [{
    #     #             "src": source,
    #     #             "mt": hypothesis,
    #     #             "ref": reference
    #     #         }]
    #     #         comet_output = self.comet_model.predict(data, batch_size=1, gpus=1)
    #     #         scores['comet'] = comet_output.scores[0]
    #     #     except Exception as e:
    #     #         logger.warning(f"Failed to calculate COMET score: {e}")
    #     #         scores['comet'] = 0.0
    #     # else:
    #     #     scores['comet'] = 0.0

    #     # Calculate semantic similarity
    #     # Encode both sentences
    #     emb1 = self.semantic_sim_model.encode(hypothesis, convert_to_tensor=True)
    #     emb2 = self.semantic_sim_model.encode(reference, convert_to_tensor=True)

    #     # Calculate cosine similarity
    #     similarity = util.cos_sim(emb1, emb2).item()
    #     scores['semantic_sim'] = similarity

    #     # if SEMANTIC_SIM_AVAILABLE and self.semantic_sim_model is not None:
    #     #     try:
    #     #         # Encode both sentences
    #     #         emb1 = self.semantic_sim_model.encode(hypothesis, convert_to_tensor=True)
    #     #         emb2 = self.semantic_sim_model.encode(reference, convert_to_tensor=True)

    #     #         # Calculate cosine similarity
    #     #         similarity = util.cos_sim(emb1, emb2).item()
    #     #         scores['semantic_sim'] = similarity
    #     #     except Exception as e:
    #     #         logger.warning(f"Failed to calculate semantic similarity: {e}")
    #     #         scores['semantic_sim'] = 0.0
    #     # else:
    #     #     scores['semantic_sim'] = 0.0

    #     # Calculate combined score (weighted average)
    #     # You can adjust these weights based on your preferences
    #     # weights = {'bleu': 0.3, 'comet': 0.4, 'semantic_sim': 0.3}
    #     # combined_score = sum(scores[metric] * weights[metric] for metric in weights if metric in scores)
    #     # scores['combined'] = combined_score

    #     return scores

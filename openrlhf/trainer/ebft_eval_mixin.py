"""
EBFTEvalMixin — task-specific downstream evaluation methods.

Extracted from ebft_trainer.py to keep that file focused on the core
training loop. BaseEBFTTrainer inherits from this mixin.

Covers:
  - GSM8K / MATH evaluation
  - HumanEval evaluation
  - OpenCode evaluation
  - Machine-translation evaluation (sacreBLEU + COMET-22)
  - Translation metric initialization
  - Batched translation scoring utility

Module-level helpers (code execution sandbox):
  - TimeoutException
  - time_limit()
  - _run_code_in_subprocess()
"""

import math
import signal
import time
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from typing import List

import ray
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Code-execution sandbox (used by coding eval methods)
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time using signal.alarm."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Code execution timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
except SyntaxError:
    print("ERROR:syntax")
    sys.stdout.flush()
    _original_exit(0)
except SystemExit:
    print("ERROR:syntax")
    sys.stdout.flush()
    _original_exit(0)
except Exception as e:
    print("ERROR:runtime")
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
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            close_fds=True,
            start_new_session=True,
        )

        output = result.stdout.strip()
        if "SUCCESS" in output:
            return True, None
        elif "ERROR:syntax" in output:
            return False, "syntax"
        elif "ERROR:test_failure" in output:
            return False, "test_failure"
        elif "ERROR:runtime" in output:
            return False, "runtime"
        else:
            return False, "syntax"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "syntax"


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class EBFTEvalMixin:
    """
    Mixin providing task-specific downstream evaluation methods for BaseEBFTTrainer.

    Expects the following attributes to be present on the host class (all set by
    BaseEBFTTrainer.__init__):
        self.strategy, self.args, self.tokenizer, self.generate_kwargs,
        self.samples_generator, self._wandb, self._tensorboard,
        self.reward_model_groups
    """

    # ------------------------------------------------------------------
    # GSM8K / MATH evaluation
    # ------------------------------------------------------------------

    def evaluate_downstream_gsm8k_math(self, eval_downstream_dataloader, global_step, generate_max_len, temperature, n_samples_per_prompt):
        """Evaluate model performance on GSM8K / MATH."""
        from openrlhf.utils.math_verifier import get_llm_answer
        from math_verify import verify

        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            for dict_data in eval_downstream_dataloader:
                prompts = [d["prompt"] for d in dict_data]
                labels = [d["label"] for d in dict_data]
                all_prompts.extend(prompts)
                all_labels.extend(labels)

            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["generate_max_len"] = generate_max_len
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95 if temperature > 0.0 else 1.0
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt

            eval_samples_list = self.samples_generator.generate_samples_for_downstream(
                all_prompts, all_labels, **generate_kwargs
            )

            logger.info(f"DOWNTEMP: {temperature}; DOWNLEN: {generate_max_len}; DOWNSAMP: {n_samples_per_prompt}")

            rewards_list = []
            response_type_list = []
            for samples in eval_samples_list:
                gen_seq_list = [gen.split(self.tokenizer.eos_token)[0] if self.tokenizer.eos_token in gen else gen for gen in samples.generated_sequences]
                for (pr, lbl, gen) in zip(samples.prompt_strings, samples.label_strings, gen_seq_list):
                    lbl, _ = get_llm_answer(lbl)
                    try:
                        prediction, response_type = get_llm_answer(gen)
                        acc = verify(lbl, prediction) * 1.0
                    except:
                        acc = 0.0
                        response_type = "text"
                    logger.info(f"Step-{global_step}: PROMPT: {[pr]}; GEN: {[gen]} PRED: {prediction}; LBL: {lbl}; ACC: {acc}")
                    rewards_list.append(acc)
                    response_type_list.append(response_type)

            rewards = torch.tensor(rewards_list).view(-1, n_samples_per_prompt)

            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_responses = len(response_type_list)
            llm_code_pct = sum(1 for rt in response_type_list if rt == 'llm-code') / total_responses
            tinygsm_code_pct = sum(1 for rt in response_type_list if rt == 'tinygsm-code') / total_responses
            text_pct = sum(1 for rt in response_type_list if rt == 'text') / total_responses

            logs = {
                "reward_down_passk": passk,
                "reward_down_pass1": pass1,
                "response_type_llm_code_pct": llm_code_pct,
                "response_type_tinygsm_code_pct": tinygsm_code_pct,
                "response_type_text_pct": text_pct,
            }

            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        time_str = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    # ------------------------------------------------------------------
    # HumanEval evaluation
    # ------------------------------------------------------------------

    def evaluate_downstream_humaneval(self, eval_downstream_dataloader, global_step, generate_max_len, temperature, n_samples_per_prompt):
        """Evaluate model performance on HumanEval via unit tests."""
        import re

        start_time = time.time()
        logger.info(f"⏰ HumanEval evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            all_unit_tests = []
            all_entry_points = []

            for dict_data in eval_downstream_dataloader:
                prompts = [d["prompt"] for d in dict_data]
                labels = [d.get("label", "") for d in dict_data]
                tests = [d.get("unit_test", "") for d in dict_data]
                entry_points = [d.get("entry_point") for d in dict_data]

                all_prompts.extend(prompts)
                all_labels.extend(labels)
                all_entry_points.extend(entry_points)
                for test_code, entry_point in zip(tests, entry_points):
                    if entry_point:
                        unit_tests = [test_code, f"check({entry_point})"]
                    else:
                        unit_tests = [test_code]
                    all_unit_tests.append(unit_tests)

            if not all_prompts:
                logger.warning("No prompts collected for HumanEval; skipping.")
                return

            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["generate_max_len"] = generate_max_len
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95 if temperature > 0.0 else 1.0
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt

            eval_samples_list = self.samples_generator.generate_samples_for_downstream(
                all_prompts, all_labels, all_unit_tests=all_unit_tests, all_entry_points=all_entry_points, **generate_kwargs
            )

            def _combine_prompt_and_completion(prompt: str, completion: str) -> str:
                if not completion:
                    return prompt
                prompt_no_trailing_newlines = prompt.rstrip("\n")
                if completion.startswith("\n"):
                    return prompt_no_trailing_newlines + completion
                if prompt_no_trailing_newlines.endswith("    "):
                    return prompt_no_trailing_newlines + completion
                return prompt_no_trailing_newlines + "\n" + completion

            def _extract_prompt_preamble(prompt: str) -> str:
                if not prompt:
                    return ""
                lines = prompt.splitlines()
                preamble = []
                for line in lines:
                    stripped = line.lstrip()
                    if re.match(r"^(def|class)\s+\w+", stripped):
                        break
                    preamble.append(line)
                return "\n".join(preamble).rstrip()

            def _indent_completion_if_needed(completion: str) -> str:
                if not completion:
                    return completion
                lines = completion.splitlines()
                first_nonempty = None
                for line in lines:
                    if line.strip():
                        first_nonempty = line
                        break
                if first_nonempty is None:
                    return completion
                if first_nonempty.startswith((" ", "\t")):
                    return completion
                indent = "    "
                return "\n".join([indent + ln if ln.strip() else ln for ln in lines])

            rewards_list = []
            error_counts = Counter()
            log_limit = 5
            logged = 0
            for samples in eval_samples_list:
                raw_gen_list = samples.generated_sequences
                gen_seq_list = [self._extract_code_from_response_humaneval(code) for code in raw_gen_list]
                for (pr, ut, raw_gen, gen) in zip(samples.prompt_strings, samples.unit_tests, raw_gen_list, gen_seq_list):
                    if logged < log_limit:
                        logger.info("HumanEval raw generation %d:\n%s", logged, raw_gen or "")
                        logger.info("HumanEval processed generation %d:\n%s", logged, gen or "")
                    stripped = (gen or "").lstrip()
                    first_line = ""
                    for line in stripped.splitlines():
                        if line.strip():
                            first_line = line.strip()
                            break
                    if re.match(r"^(def|class)\s+\w+", first_line):
                        preamble = _extract_prompt_preamble(pr)
                        if preamble:
                            full_code = preamble + "\n" + stripped
                        else:
                            full_code = stripped
                    else:
                        indented = _indent_completion_if_needed(gen or "")
                        full_code = _combine_prompt_and_completion(pr, indented)
                    if logged < log_limit:
                        logger.info("HumanEval executed code %d:\n%s", logged, full_code or "")
                    logged += 1
                    is_correct, error_type = self._execute_and_test_code_humaneval(full_code, ut)
                    rewards_list.append(float(is_correct))
                    if not is_correct:
                        error_counts[error_type or "unknown"] += 1
            if logged >= log_limit:
                logger.info("HumanEval logging truncated to first %d generations.", log_limit)

            rewards = torch.tensor(rewards_list).view(-1, n_samples_per_prompt)
            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_attempts = len(rewards_list)
            if total_attempts > 0:
                syntax_pct = error_counts.get("syntax", 0) / total_attempts
                fail_pct = error_counts.get("test_failure", 0) / total_attempts
                timeout_pct = error_counts.get("timeout", 0) / total_attempts
                runtime_pct = error_counts.get("runtime", 0) / total_attempts
            else:
                syntax_pct = fail_pct = timeout_pct = runtime_pct = 0.0

            logs = {
                "reward_humaneval_passk": passk,
                "reward_humaneval_pass1": pass1,
                "err_humaneval_syntax_pct": syntax_pct,
                "err_humaneval_fail_pct": fail_pct,
                "err_humaneval_timeout_pct": timeout_pct,
                "err_humaneval_runtime_pct": runtime_pct,
            }

            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        time_str = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        logger.info(f"✨ HumanEval evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    # ------------------------------------------------------------------
    # OpenCode evaluation
    # ------------------------------------------------------------------

    def evaluate_downstream_opencode(self, eval_downstream_dataloader, global_step, generate_max_len, temperature, n_samples_per_prompt):
        """Evaluate model performance on OpenCode via unit tests."""
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            all_prompts = []
            all_labels = []
            all_unit_tests = []
            for dict_data in eval_downstream_dataloader:
                prompts = [d["prompt"] for d in dict_data]
                labels = [d["label"] for d in dict_data]
                unit_tests = [d["unit_test"] for d in dict_data]
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                all_unit_tests.extend(unit_tests)

            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["generate_max_len"] = generate_max_len
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95 if temperature > 0.0 else 1.0
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            eval_samples_list = self.samples_generator.generate_samples_for_downstream(
                all_prompts, all_labels, all_unit_tests=all_unit_tests, **generate_kwargs
            )

            rewards_list = []
            error_counts = Counter()
            for samples in eval_samples_list:
                gen_seq_list = [self._extract_code_from_response(code) for code in samples.generated_sequences]
                for (pr, ut, gen) in zip(samples.prompt_strings, samples.unit_tests, gen_seq_list):
                    is_correct, error_type = self._execute_and_test_code(gen, ut)
                    rewards_list.append(float(is_correct))
                    if not is_correct:
                        error_counts[error_type or "unknown"] += 1

            rewards = torch.tensor(rewards_list).view(-1, n_samples_per_prompt)

            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
            pass1 = rewards.mean().item()

            total_attempts = len(rewards_list)
            if total_attempts > 0:
                syntax_pct = error_counts.get("syntax", 0) / total_attempts
                fail_pct = error_counts.get("test_failure", 0) / total_attempts
                timeout_pct = error_counts.get("timeout", 0) / total_attempts
            else:
                syntax_pct = fail_pct = timeout_pct = 0.0

            logs = {
                "reward_opencode_passk": passk,
                "reward_opencode_pass1": pass1,
                "err_opencode_syntax_pct": syntax_pct,
                "err_opencode_fail_pct": fail_pct,
                "err_opencode_timeout_pct": timeout_pct,
            }

            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        time_str = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    # ------------------------------------------------------------------
    # Code extraction helpers
    # ------------------------------------------------------------------

    def _extract_code_from_response(self, response):
        """Extract Python code from a response that may contain markdown code blocks."""
        import re

        if not response:
            return ''

        normalized = response.replace('\r', '')

        solution_pattern = r'Solution\s*:\s*\n?'
        solution_match = re.search(solution_pattern, normalized, flags=re.IGNORECASE)
        if solution_match:
            normalized = normalized[solution_match.end():]

        code_block_pattern = r'```(?:[a-zA-Z0-9_+.\-]+)?\s*\n(.*?)(?:```|$)'
        match = re.search(code_block_pattern, normalized, re.DOTALL)

        if match:
            code = match.group(1).strip()
        else:
            if '```' in normalized:
                start = normalized.find('```')
                remainder = normalized[start + 3:]
                remainder = remainder.lstrip()

                newline_idx = remainder.find('\n')
                if newline_idx != -1:
                    first_line = remainder[:newline_idx].strip()
                    if first_line and len(first_line) <= 32 and ' ' not in first_line:
                        remainder = remainder[newline_idx + 1:]

                code = remainder.split('```', 1)[0].strip()
            else:
                code = normalized.strip()

        lines = code.split('\n')
        function_lines = []
        markdown_prefixes = ('###', '##', '**', '* ', '> ', 'Implementation', 'Solution', 'Answer')

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

            if (
                stripped.startswith('print(')
                or stripped.startswith('# Test')
                or stripped.startswith('# Example')
                or stripped.startswith('# ---')
                or stripped.startswith('if __name__')
                or (stripped.startswith('assert ') and i > 0)
                or (not line.startswith(' ') and not line.startswith('\t')
                    and re.match(r'^[a-z_][a-z0-9_]*\s*=\s*[A-Z]', stripped)
                    and function_lines)
            ):
                break

            function_lines.append(line)

        while function_lines and not function_lines[-1].strip():
            function_lines.pop()

        if function_lines:
            return '\n' + '\n'.join(function_lines)
        return ''

    def _extract_code_from_response_humaneval(self, response):
        """Extract code for HumanEval while preserving leading indentation."""
        import re

        if not response:
            return ""

        normalized = response.replace("\r", "")

        solution_pattern = r"Solution\s*:\s*\n?"
        solution_match = re.search(solution_pattern, normalized, flags=re.IGNORECASE)
        if solution_match:
            normalized = normalized[solution_match.end():]

        code_block_pattern = r"```(?:[a-zA-Z0-9_+.\-]+)?\s*\n(.*?)(?:```|$)"
        match = re.search(code_block_pattern, normalized, re.DOTALL)

        if match:
            code = match.group(1).rstrip()
        else:
            if "```" in normalized:
                start = normalized.find("```")
                remainder = normalized[start + 3:]
                remainder = remainder.lstrip("\n")

                newline_idx = remainder.find("\n")
                if newline_idx != -1:
                    first_line = remainder[:newline_idx].strip()
                    if first_line and len(first_line) <= 32 and " " not in first_line:
                        remainder = remainder[newline_idx + 1:]

                code = remainder.split("```", 1)[0].rstrip()
            else:
                code = normalized.strip("\n")

        lines = code.split("\n")
        function_lines = []
        markdown_prefixes = ("###", "##", "**", "* ", "> ", "Implementation", "Solution", "Answer")

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not function_lines:
                if not stripped:
                    continue
                if stripped in {"`", "``", "```"}:
                    continue
                lowered = stripped.lower()
                if any(lowered.startswith(prefix.lower()) for prefix in markdown_prefixes):
                    continue

            if (
                stripped.startswith("print(")
                or stripped.startswith("# Test")
                or stripped.startswith("# Example")
                or stripped.startswith("# ---")
                or stripped.startswith("if __name__")
                or (stripped.startswith("assert ") and i > 0)
                or (
                    not line.startswith(" ")
                    and not line.startswith("\t")
                    and re.match(r"^[a-z_][a-z0-9_]*\s*=\s*[A-Z]", stripped)
                    and function_lines
                )
            ):
                break

            function_lines.append(line)

        while function_lines and not function_lines[-1].strip():
            function_lines.pop()

        if function_lines:
            return "\n" + "\n".join(function_lines)
        return ""

    def _execute_and_test_code_humaneval(self, code, unit_tests, timeout=3):
        """Execute HumanEval code and tests in an isolated subprocess."""
        return self._execute_and_test_code(code, unit_tests, timeout=timeout)

    def _execute_and_test_code(self, code, unit_tests, timeout=3):
        """Execute generated code and run provided unit tests in an isolated subprocess."""
        if isinstance(unit_tests, str):
            import ast
            try:
                unit_tests = ast.literal_eval(unit_tests)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Failed to parse unit_tests: {exc}") from exc

        return _run_code_in_subprocess(code, unit_tests, timeout)

    # ------------------------------------------------------------------
    # Machine translation evaluation
    # ------------------------------------------------------------------

    def evaluate_downstream_translation(self, eval_downstream_dataloader, global_step, generate_max_len, temperature, n_samples_per_prompt):
        """Evaluate model performance on machine translation downstream datasets.

        Logs sacreBLEU + COMET-22 per direction and macro-averaged aggregates.
        If n_samples_per_prompt > 1, also logs oracle best-of-k metrics.
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        def _sanitize_metric_key(text: str) -> str:
            import re
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()

        def _extract_source_text(prompt: str) -> str:
            if not prompt:
                return ""
            p = str(prompt).replace("\r", "")
            try:
                after_colon = p.split(":", 1)[1]
                return after_colon.split("\n", 1)[0].strip(" \t")
            except Exception:
                return p.strip()

        _lang_name_map = {
            "en": "English",
            "cs": "Czech",
            "de": "German",
            "is": "Icelandic",
            "ru": "Russian",
            "zh": "Chinese",
        }

        def _clean_translation(text: str, direction: str = None) -> str:
            import re
            if text is None:
                return ""
            s = str(text).replace("\r", "")
            if getattr(self.tokenizer, "eos_token", None) and self.tokenizer.eos_token in s:
                s = s.split(self.tokenizer.eos_token, 1)[0]
            lines = [ln.strip() for ln in s.split("\n")]
            lines = [ln for ln in lines if ln]
            if not lines:
                return ""
            first = lines[0]
            if re.match(r"^[A-Za-z][A-Za-z\\s\\-]{0,32}:\\s*$", first) and len(lines) > 1:
                first = lines[1]
            prefixes = ["Translation:", "Answer:", "Output:"]
            if direction and "-" in str(direction):
                try:
                    tgt_code = str(direction).split("-", 1)[1].strip().lower()
                except Exception:
                    tgt_code = ""
                tgt_name = _lang_name_map.get(tgt_code)
                if tgt_name:
                    prefixes.extend([f"{tgt_name}:", f"{tgt_name} :"])
                if tgt_code:
                    prefixes.extend([f"{tgt_code}:", f"{tgt_code.upper()}:"])
            for pfx in prefixes:
                if first.lower().startswith(pfx.lower()):
                    first = first[len(pfx):].strip()
                    break
            first = first.strip(" \t\"'")
            return first

        def _mean_finite(values):
            finite = []
            for v in values:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                finite.append(fv)
            return sum(finite) / len(finite) if finite else float("nan")

        def _corpus_sacrebleu(hypotheses, references, tokenize: str) -> float:
            try:
                from sacrebleu.metrics import BLEU
                bleu = BLEU(tokenize=tokenize)
                return bleu.corpus_score(hypotheses, [references]).score / 100.0
            except Exception:
                from sacrebleu import corpus_bleu
                return corpus_bleu(hypotheses, [references], tokenize=tokenize).score / 100.0

        def _compute_comet_scores(srcs, hyps, refs, batch_size: int = 32):
            if getattr(self, "comet_model", None) is None:
                self._initialize_translation_metrics()
            if getattr(self, "comet_model", None) is None:
                return [float("nan")] * len(hyps)
            comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            gpus = 1 if torch.cuda.is_available() else 0
            try:
                output = self.comet_model.predict(comet_data, batch_size=batch_size, gpus=gpus)
            except TypeError:
                output = self.comet_model.predict(comet_data, batch_size=batch_size)
            if hasattr(output, "scores"):
                scores = output.scores
            elif isinstance(output, dict) and "scores" in output:
                scores = output["scores"]
            elif isinstance(output, (list, tuple)):
                scores = output[0] if (len(output) > 0 and isinstance(output[0], (list, tuple))) else output
            else:
                raise RuntimeError(f"Unexpected COMET predict output type: {type(output)}")
            return [float(s) for s in scores]

        logs = {}
        with torch.no_grad():
            all_prompts: List[str] = []
            all_labels: List[str] = []
            all_dirs: List[str] = []

            for batch in eval_downstream_dataloader:
                if isinstance(batch, list) and (len(batch) == 0 or isinstance(batch[0], dict)):
                    all_prompts.extend([d["prompt"] for d in batch])
                    all_labels.extend([d["label"] for d in batch])
                    all_dirs.extend([d.get("datasource", d.get("source", "default")) for d in batch])
                else:
                    datasources, prompts, labels = batch[:3]
                    all_dirs.extend(list(datasources))
                    all_prompts.extend(list(prompts))
                    all_labels.extend(list(labels))

            all_labels = [str(x) for x in all_labels]
            all_sources = [_extract_source_text(p) for p in all_prompts]

            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["generate_max_len"] = generate_max_len
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95 if temperature > 0.0 else 1.0
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt

            logger.info(
                f"MT eval: temp={temperature}; max_len={generate_max_len}; n_samples_per_prompt={n_samples_per_prompt}; "
                f"num_prompts={len(all_prompts)}"
            )

            eval_samples_list = self.samples_generator.generate_samples_for_downstream(
                all_prompts, all_labels, **generate_kwargs
            )
            flat_hyps_raw = sum([s.generated_sequences for s in eval_samples_list], [])

            num_prompts = len(all_prompts)
            k = int(n_samples_per_prompt)
            expected = num_prompts * k
            if len(flat_hyps_raw) != expected:
                logger.warning(
                    f"MT eval: expected {expected} generations (num_prompts={num_prompts} * k={k}), "
                    f"but got {len(flat_hyps_raw)}. Truncating."
                )
                expected = min(expected, len(flat_hyps_raw))
                num_prompts = expected // k

            rep_dirs = [d for d in all_dirs[:num_prompts] for _ in range(k)]
            rep_srcs = [s for s in all_sources[:num_prompts] for _ in range(k)]
            rep_refs = [r for r in all_labels[:num_prompts] for _ in range(k)]
            raw_hyps = flat_hyps_raw[:len(rep_refs)]
            hyps = [_clean_translation(h, direction=d) for h, d in zip(raw_hyps, rep_dirs)]

            comet_batch_size = min(32, getattr(self.args, "eval_comet_batch_size", 32))
            comet_scores = _compute_comet_scores(rep_srcs, hyps, rep_refs, batch_size=comet_batch_size)

            first_indices = [i * k for i in range(num_prompts)]
            hyps_1 = [hyps[idx] for idx in first_indices]
            refs_1 = all_labels[:num_prompts]
            dirs_1 = all_dirs[:num_prompts]
            comet_1 = [comet_scores[idx] for idx in first_indices]

            hyps_best = None
            comet_best = None
            if k > 1:
                hyps_best = []
                comet_best = []
                for i in range(num_prompts):
                    base = i * k
                    group = comet_scores[base:base + k]
                    best_j = 0
                    best_val = float("-inf")
                    for j, v in enumerate(group):
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if math.isnan(fv):
                            continue
                        if fv > best_val:
                            best_val = fv
                            best_j = j
                    best_idx = base + best_j
                    hyps_best.append(hyps[best_idx])
                    comet_best.append(comet_scores[best_idx])

            def _tokenize_for_direction(direction: str) -> str:
                try:
                    tgt = str(direction).split("-", 1)[1].strip().lower()
                except Exception:
                    tgt = ""
                return "zh" if tgt == "zh" else "13a"

            by_dir = {}
            for d, h, r, c in zip(dirs_1, hyps_1, refs_1, comet_1):
                entry = by_dir.setdefault(d, {"hyps": [], "refs": [], "comet": []})
                entry["hyps"].append(h)
                entry["refs"].append(r)
                entry["comet"].append(c)

            log_per_direction = False

            dir_metrics = {}
            for direction, data in sorted(by_dir.items(), key=lambda kv: str(kv[0])):
                tok = _tokenize_for_direction(direction)
                bleu = _corpus_sacrebleu(data["hyps"], data["refs"], tokenize=tok)
                comet_mean = _mean_finite(data["comet"])
                dir_metrics[direction] = {"bleu": bleu, "comet": comet_mean}

                if log_per_direction:
                    key = _sanitize_metric_key(direction)
                    logs[f"bleu_{key}"] = bleu
                    logs[f"comet_{key}"] = comet_mean

            en_to_xx = [m for d, m in dir_metrics.items() if str(d).startswith("en-") and str(d) != "en-en"]
            xx_to_en = [m for d, m in dir_metrics.items() if str(d).endswith("-en") and not str(d).startswith("en-")]

            if en_to_xx:
                logs["bleu_en_to_xx_avg"] = sum(m["bleu"] for m in en_to_xx) / len(en_to_xx)
                logs["comet_en_to_xx_avg"] = _mean_finite([m["comet"] for m in en_to_xx])
            if xx_to_en:
                logs["bleu_xx_to_en_avg"] = sum(m["bleu"] for m in xx_to_en) / len(xx_to_en)
                logs["comet_xx_to_en_avg"] = _mean_finite([m["comet"] for m in xx_to_en])

            if dir_metrics:
                logs["bleu_avg"] = sum(m["bleu"] for m in dir_metrics.values()) / len(dir_metrics)
                logs["comet_avg"] = _mean_finite([m["comet"] for m in dir_metrics.values()])

            if k > 1 and hyps_best is not None and comet_best is not None:
                by_dir_best = {}
                for d, h, r, c in zip(dirs_1, hyps_best, refs_1, comet_best):
                    entry = by_dir_best.setdefault(d, {"hyps": [], "refs": [], "comet": []})
                    entry["hyps"].append(h)
                    entry["refs"].append(r)
                    entry["comet"].append(c)

                dir_metrics_best = {}
                for direction, data in sorted(by_dir_best.items(), key=lambda kv: str(kv[0])):
                    tok = _tokenize_for_direction(direction)
                    bleu = _corpus_sacrebleu(data["hyps"], data["refs"], tokenize=tok)
                    comet_mean = _mean_finite(data["comet"])
                    dir_metrics_best[direction] = {"bleu": bleu, "comet": comet_mean}

                    if log_per_direction:
                        key = _sanitize_metric_key(direction)
                        logs[f"bleu_bestofk_{key}"] = bleu
                        logs[f"comet_bestofk_{key}"] = comet_mean

                en_to_xx_best = [m for d, m in dir_metrics_best.items() if str(d).startswith("en-") and str(d) != "en-en"]
                xx_to_en_best = [m for d, m in dir_metrics_best.items() if str(d).endswith("-en") and not str(d).startswith("en-")]

                if en_to_xx_best:
                    logs["bleu_bestofk_en_to_xx_avg"] = sum(m["bleu"] for m in en_to_xx_best) / len(en_to_xx_best)
                    logs["comet_bestofk_en_to_xx_avg"] = _mean_finite([m["comet"] for m in en_to_xx_best])
                if xx_to_en_best:
                    logs["bleu_bestofk_xx_to_en_avg"] = sum(m["bleu"] for m in xx_to_en_best) / len(xx_to_en_best)
                    logs["comet_bestofk_xx_to_en_avg"] = _mean_finite([m["comet"] for m in xx_to_en_best])

                if dir_metrics_best:
                    logs["bleu_bestofk_avg"] = sum(m["bleu"] for m in dir_metrics_best.values()) / len(dir_metrics_best)
                    logs["comet_bestofk_avg"] = _mean_finite([m["comet"] for m in dir_metrics_best.values()])

            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k_: v_ for k_, v_ in {**logs, "global_step": global_step}.items()})
            elif self._tensorboard is not None:
                for k_, v_ in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k_}", v_, global_step)

        end_time = time.time()
        time_str = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    # ------------------------------------------------------------------
    # Translation metric initialization
    # ------------------------------------------------------------------

    def _initialize_translation_metrics(self):
        """Initialize translation evaluation metrics (COMET-22)."""
        if getattr(self, "comet_model", None) is not None:
            return

        self.comet_model = None
        self.semantic_sim_model = None

        try:
            from comet import download_model, load_from_checkpoint
            model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
            try:
                model.eval()
            except Exception:
                pass
            self.comet_model = model
            logger.info("✅ Loaded COMET-22 model: Unbabel/wmt22-comet-da")
        except Exception as e:
            logger.warning(f"Failed to initialize COMET-22 (Unbabel/wmt22-comet-da): {e}")
            self.comet_model = None

    # ------------------------------------------------------------------
    # Batched translation scoring utility
    # ------------------------------------------------------------------

    def score_translations_batched(self, hypotheses, references, sources=None, batch_size=32):
        """Score multiple translations using batched inference for efficiency."""
        from sacrebleu import sentence_bleu

        num_samples = len(hypotheses)
        bleu_scores = []

        logger.info(f"Computing BLEU scores for {num_samples} samples...")
        for hypothesis, reference in zip(hypotheses, references):
            bleu_score = sentence_bleu(hypothesis, [reference]).score / 100.0
            bleu_scores.append(bleu_score)
        logger.info(f"✅ BLEU scores computed")

        all_scores = {"bleu": bleu_scores}

        logger.info(f"Computing semantic similarity scores...")
        logger.info(f"Encoding {num_samples} hypotheses and references using reward model...")

        combined_sequences = hypotheses + references
        reward_groups = self.reward_model_groups
        for i, reward_model_group in enumerate(reward_groups):
            group_label = self._reward_model_group_label(reward_model_group, f"reward_model_{i}")
            rm_actors = reward_model_group._actor_handlers
            if not rm_actors:
                raise RuntimeError("No reward model actors available.")

            if not self._is_comet_group(reward_model_group):
                batches = [
                    (bid, combined_sequences[start:start + batch_size])
                    for bid, start in enumerate(range(0, len(combined_sequences), batch_size))
                ]

                inflight_refs = []
                for k, (bid, batch) in enumerate(batches):
                    a = rm_actors[k % len(rm_actors)]
                    ref = a.forward.remote(input_sequences=batch)
                    inflight_refs.append((bid, ref))

                id_by_ref = {ref: bid for (bid, ref) in inflight_refs}
                pending = [ref for (_, ref) in inflight_refs]
                results_by_bid = {}

                while pending:
                    ready, pending = ray.wait(pending, num_returns=1)
                    r = ready[0]
                    bid = id_by_ref[r]
                    batch_out = ray.get(r)
                    results_by_bid[bid] = batch_out

                ordered = [results_by_bid[i] for i in range(len(batches))]
                all_embeddings = torch.cat(ordered, dim=0)

                gen_embedding = all_embeddings[:num_samples]
                gt_embedding = all_embeddings[num_samples:]

                logger.info(f"✅ Embeddings computed from reward model (hyp: {gen_embedding.shape}, ref: {gt_embedding.shape})")

                gen_embedding = gen_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
                gt_embedding = gt_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(-2)

                from openrlhf.utils.embedding_utils import get_alignment_rewards
                logger.info(f"Computing cosine similarities...")
                gt_rewards_tensor = get_alignment_rewards(gen_embedding, gt_embedding)
                semantic_sim_scores = gt_rewards_tensor.squeeze(0).squeeze(0).squeeze(-1).cpu().tolist()
            else:
                if sources is None:
                    raise ValueError(
                        "COMET reward models require source inputs. Pass `sources` to score_translations_batched."
                    )
                if len(sources) != num_samples:
                    raise ValueError(
                        f"COMET scoring expects {num_samples} sources, but received {len(sources)}."
                    )
                source_sequences = list(sources)
                batches = [
                    (
                        bid,
                        hypotheses[start:start + batch_size],
                        references[start:start + batch_size],
                        source_sequences[start:start + batch_size],
                    )
                    for bid, start in enumerate(range(0, num_samples, batch_size))
                ]
                inflight_refs = []
                for k, (bid, hyps_batch, refs_batch, prompts_batch) in enumerate(batches):
                    a = rm_actors[k % len(rm_actors)]
                    ref = a.forward.remote(input_sequences=hyps_batch, gt_sequences=refs_batch, ct_sequences=prompts_batch)
                    inflight_refs.append((bid, ref))
                id_by_ref = {ref: bid for (bid, ref) in inflight_refs}
                pending = [ref for (_, ref) in inflight_refs]
                results_by_bid = {}

                while pending:
                    done, pending = ray.wait(pending, num_returns=1)
                    for ref in done:
                        bid = id_by_ref[ref]
                        result = ray.get(ref)
                        results_by_bid[bid] = result
                ordered = [results_by_bid[i] for i in range(len(batches))]
                if len(ordered) > 0 and isinstance(ordered[0], torch.Tensor):
                    semantic_sim_scores = torch.cat(ordered).tolist()
                else:
                    semantic_sim_scores = sum(ordered, [])

            all_scores[f'semantic_sim_model_{i}'] = semantic_sim_scores
            logger.info(f"✅ Semantic similarity scores computed using reward model group '{group_label}' (index {i})")

        return all_scores

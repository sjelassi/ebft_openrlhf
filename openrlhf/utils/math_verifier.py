from math_verify import parse,verify
from argparse import ArgumentParser
from tqdm import tqdm
import jsonlines
import os
import re
import numpy as np
import torch
import re
import os
import signal
import json
import hashlib
from pathlib import Path
import math
 
import torch


def hamming_distance(l1, l2):
    """
    Compute Hamming distance between two lists of equal length.
    """
    if len(l1) != len(l2):
        raise ValueError("Hamming distance is only defined for lists of equal length.")
    return sum(a != b for a, b in zip(l1, l2))


def normalized_edit(a, b):
    d = hamming_distance(a, b)
    L = len(b)
    return d / L

 



def reward_func(queries, prompts, answers):
    scores = []
    assert isinstance(queries, list)
    assert isinstance(prompts, list)
    assert isinstance(answers, list)
    for q, p, a in zip(queries, prompts, answers):
        assert (q[:len(p)] == p)


        pred = q[len(p):]
        target   = a
        print(f"QUERIES: {q};\n PROMPTS: {p};\n ANSWER: {a};\n BOOL: {q[:len(p)] == p}; PRED: {len(pred)}; TGT: {len(target)}",flush=True)
        assert len(pred) == len(target)

        if len(pred) > 1:
            reward = 1 - normalized_edit(pred, target)
        else:
            reward = int(pred == target)

    
        scores.append(reward)
        print(f"Pred: {pred}, Target: {target}, Reward: {reward:.3f}", flush=True)

    return {"rewards": torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)}
  

    # # CORRECT REWARD: Check if prediction matches target
        # # Exact match for same length sequences
        # correct = int(pred == target)
        # reward = 1.0 if correct else 0.0
        

# def levenshtein_distance(a, b):
#     """Levenshtein distance between two lists of token_ids."""
#     m, n = len(a), len(b)
#     # 1D rolling DP to avoid O(mn) memory
#     prev = list(range(n + 1))
#     for i in range(1, m + 1):
#         cur = [i] + [0] * n
#         ai = a[i - 1]
#         for j in range(1, n + 1):
#             cost = 0 if ai == b[j - 1] else 1
#             cur[j] = min(
#                 prev[j] + 1,       # deletion
#                 cur[j - 1] + 1,    # insertion
#                 prev[j - 1] + cost # substitution
#             )
#         prev = cur
#     return prev[n]

# def normalized(x, denom):
#     return x / max(1, denom)

# def reward_func(queries, prompts, answers):
#     """
#     queries : list[list[int]]  -> prompt || generated_suffix
#     prompts : list[list[int]]  -> prompt tokens (should equal queries[:len(prompt)])
#     answers : list[list[int]]  -> FULL gold target continuation (y*), not just one token

#     Returns a scalar reward per completion, designed for 1-token rollouts:
#       R = (ED(prefix, y*) - ED(prefix + gen_token, y*)) / |y*|
#     If more than one token was generated, average ΔED across tokens (still scalar).
#     """
#     assert isinstance(queries, list) and isinstance(prompts, list) and isinstance(answers, list)

#     rewards = []
#     for q, p, y_star in zip(queries, prompts, answers):
#         # sanity: prompt must be a prefix of query
#         if q[:len(p)] != p:
#             # be permissive but informative
#             # trim/repair if some logging wrapper added BOS, etc.
#             raise ValueError("query does not start with prompt; check tokenization/alignment")

#         gen = q[len(p):]                 # model-generated suffix (could be 1+ tokens)
#         prefix = p                       # gold prefix you fed to the model
#         L = len(y_star)
#         print(f"QUERIES: {q};\n PROMPTS: {p};\n ANSWER: {y_star};\n BOOL: {q[:len(p)] == p}; PRED: {len(gen)}; TGT: {len(y_star)}",flush=True)
#         assert len(gen) == len(y_star)
#         # ΔED for each *additional* generated token along the same prefix path
#         # (prefix stays constant; we extend it by gen[:t])
#         ed_prev = levenshtein_distance(prefix, y_star)

#         deltas = []
#         cur_prefix = list(prefix)
#         for tkn in gen:
#             cur_prefix.append(tkn)
#             ed_after = levenshtein_distance(cur_prefix, y_star)
#             delta = ed_prev - ed_after              # positive if we got closer
#             deltas.append(normalized(delta, L))
#             ed_prev = ed_after
#         # If this was a true 1-token rollout, this is just one number.
#         # If multiple tokens slipped in, average per-step improvements.
#         R = float(sum(deltas) / len(deltas))

#         rewards.append(R)

#     return {"rewards": torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)}







def handler(signum, frame):
    raise TimeoutError("Execution timed out!")

def execute_function(code: str, timeout=3): 
    try:
        # Set the alarm handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)  # Start the alarm
        local_namespace = {}
        exec(code, {}, local_namespace)
        return str(local_namespace["simple_math_problem"]())
    except TimeoutError as e:
        return None
    except Exception:
        return None
    finally:
        # Always disable the alarm after execution
        signal.alarm(0)

def execute_tinygsm_code(text):
    code = text.split('def')[-1]
    code = 'def' + code
    try:
        return execute_function(code)
    except:
        return None

def execute_llm_code(text):
    try:
        # Extract code inside <llm-code> tags
        code_match = re.search(r'<llm-code>(.*?)</llm-code>', text, re.DOTALL)
        if not code_match:
            return None
        
        code = code_match.group(1).strip()
        
        # Create a dictionary for execution context
        exec_globals = {}
        
        # Split the code into lines and execute it
        lines = code.split("\n")
        last_expr = lines[-1]  # The last line of code
        timeout = 3
        
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)  # Start the alarm
            exec(code, exec_globals)
        except TimeoutError as e:

            return None
        except Exception:
            return None
        finally:
            # Always disable the alarm after execution
            signal.alarm(0)
        
        return str(eval(last_expr, exec_globals))
    except:
        return None
    
def execute_code(text):
    if '<llm-code>' in text:
        code_out = execute_llm_code(text)
        return code_out
    else:
        return execute_tinygsm_code(text)

def parse_text_answer(text):
    answer = parse(text)

def get_llm_answer(text):
    response_type = 'text'
    if '<llm-code>' in text:
        code_out = execute_llm_code(text)
        response_type = 'llm-code'
        if code_out is not None:
            return parse(code_out), 'llm-code'
    if 'def' in text:
        code_out = execute_tinygsm_code(text)
        response_type = 'tinygsm-code'
        if code_out is not None:
            return parse(code_out), 'tinygsm-code'
    
    return parse(text), response_type


def verify_llm_answer(llm_text, answer_text):
    llm_answer, _ = get_llm_answer(llm_text)
    # Wrap answer in \boxed{} for proper parsing (math_verify requires this)
    correct_answer = parse(f"\\boxed{{{answer_text}}}")
    return verify(llm_answer, correct_answer)


if __name__ == "__main__":
    gsm8k_answer = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. #### 18"
    tinygsm_style_wrong = "\n\ndef simple_math_problem() -> int:\n    '''\n    Janet's ducks lay 16 eggs per day.\n    She eats three for breakfast every morning and bakes muffins for her friends every day with four.\n    She sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n    How much in dollars does she make every day at the farmers' market?\n    '''\n    eggs_per_day = 16\n    breakfast_per_day = 3\n    muffins_per_day = 4\n    fresh_duck_eggs = 4\n    total_eggs = breakfast_per_day * 2 * 3\n    total_muffins = muffins_per_day * 2 * 3\n    total_dollars = total_eggs + total_muffins\n    dollars_per_day = total_dollars / fresh_duck_eggs\n    result = dollars_per_day\n    return result\n"
    tinygsm_style = "\n\ndef simple_math_problem() -> int:\n    '''\n    Janet\u2019s ducks lay 16 eggs per day.\n    She eats three for breakfast every morning and bakes muffins for her friends every day with four.\n    She sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n    How much in dollars does she make every day at the farmers' market?\n    '''\n    eggs_per_day = 16\n    breakfast_eggs = 3\n    muffin_eggs = 4\n    remaining_eggs = eggs_per_day - breakfast_eggs - muffin_eggs\n    price_per_egg = 2\n    total_money = remaining_eggs * price_per_egg\n    result = total_money\n    return result\n"
    llm_code_style = "\n\nLet's solve this problem using Python code.\n<llm-code>\neggs_per_day = 16\neggs_per_day_for_breakfast = 3\neggs_per_day_for_muffins = 4\ndaily_earnings = eggs_per_day - eggs_per_day_for_breakfast - eggs_per_day_for_muffins\ndaily_earnings * 2\n</llm-code>\n<llm-code-output>\n96\n</llm-code-output>\nThus the farmers' market earns \\boxed{96} dollars every day."
    text_style_wrong = "\n\nTo find out how much Janet makes every day, we need to calculate the total number of eggs she uses and the total amount of money she makes from selling fresh eggs.\n\nJanet uses 3 eggs for breakfast every morning, so she uses 3 eggs for breakfast every day.\nShe eats 3 eggs for breakfast every morning, so she eats 3 eggs for breakfast every day.\nShe sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n\nThe total number of eggs Janet uses is 3 (for breakfast) + 3 (for breakfast) = 6 eggs.\n\nThe total amount of money Janet makes from selling fresh eggs is 6 eggs * $2/egg = $12.\n\nSince Janet uses 3 eggs for breakfast every day, the amount of money she makes from selling fresh eggs is 3 eggs * $2/egg = $6.\n\nThe amount of money Janet makes from selling fresh eggs is the total amount of money she makes from selling eggs minus the amount of money she makes from selling fresh eggs.\n\nSo, the amount of money Janet makes every day is $12 - $6 = $6.\n\nThus, Janet makes \\boxed{6} dollars every day at the farmers' market."
    text_style = "\n\nTo find out how much Janet makes every day, we need to calculate the total number of eggs she uses and the total amount of money she makes from selling fresh eggs.\n\nJanet uses 3 eggs for breakfast every morning, so she uses 3 eggs for breakfast every day.\nShe eats 3 eggs for breakfast every morning, so she eats 3 eggs for breakfast every day.\nShe sells the remainder at the farmers' market daily for $2 per fresh duck egg.\n\nThe total number of eggs Janet uses is 3 (for breakfast) + 3 (for breakfast) = 6 eggs.\n\nThe total amount of money Janet makes from selling fresh eggs is 6 eggs * $2/egg = $12.\n\nSince Janet uses 3 eggs for breakfast every day, the amount of money she makes from selling fresh eggs is 3 eggs * $2/egg = $6.\n\nThe amount of money Janet makes from selling fresh eggs is the total amount of money she makes from selling eggs minus the amount of money she makes from selling fresh eggs.\n\nSo, the amount of money Janet makes every day is $12 - $6 = $6.\n\nThus, Janet makes \\boxed{18} dollars every day at the farmers' market."
    assert verify_llm_answer(gsm8k_answer, gsm8k_answer) == True
    assert verify_llm_answer(llm_code_style, gsm8k_answer) == True
    assert verify_llm_answer(text_style, gsm8k_answer) == True
    assert verify_llm_answer(tinygsm_style, gsm8k_answer) == True
    assert verify_llm_answer(tinygsm_style_wrong, gsm8k_answer) == False
    assert verify_llm_answer(text_style_wrong, gsm8k_answer) == False
    print('All good!')

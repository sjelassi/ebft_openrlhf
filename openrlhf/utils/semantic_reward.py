import torch

def prepare_tensors_for_semantic_reward(prompts_list, full_sequences_list, prompt_length, stride, num_blocks, n_samples_per_prompt, context_length, gen_len, tokenizer):
    """
    Returns a list of decoded strings for semantic reward calculation, across all rollout_batch_size * n_samples_per_prompt examples.
    """

    prompts_tensor = torch.stack(prompts_list) #(rollout_batch_size // micro_rollout_batch_size * n_samples_per_prompt, micro_rollout_batch_size, prompt_len)
    # print(f'prompts_tensor shape: {prompts_tensor.shape}')

    #TO DEBUG
    # prompts_tensor_averaged = prompts_tensor.float().mean(dim=2)

    prompts_tensor = prompts_tensor.reshape(prompts_tensor.shape[0] // n_samples_per_prompt, prompts_tensor.shape[1], n_samples_per_prompt, prompts_tensor.shape[2])

    # prompts_tensor_reshaped_averaged = prompts_tensor.float().mean(dim=3)

    # print(f'prompts_tensor_averaged.shape: {prompts_tensor_averaged.shape}, prompts_tensor_reshaped_averaged.shape: {prompts_tensor_reshaped_averaged.shape}')
    # print(f'prompts_tensor_averaged: {prompts_tensor_averaged}')
    # print(f'prompts_tensor_reshaped_averaged: {prompts_tensor_reshaped_averaged}')

    full_tensor = torch.stack(full_sequences_list) #(rollout_batch_size // micro_rollout_batch_size * n_samples_per_prompt, micro_rollout_batch_size, prompt_len)
    full_tensor = full_tensor.reshape(full_tensor.shape[0] // n_samples_per_prompt, full_tensor.shape[1], n_samples_per_prompt, full_tensor.shape[2])

    # print(f'partial_ct_len: {partial_ct_len}, stride: {stride}')
    assert stride >= partial_ct_len #extend to general case after
    starting_idx = stride-partial_ct_len
    ct_tensor = prompts_tensor[:,:,:,starting_idx:-gen_len].unfold(3, partial_ct_len, stride) # (rollout_batch_size // micro_rollout_batch_size, n_samples_per_prompt, micro_rollout_batch_size, num_blocks, partial_ct_len)

    gt_tensor = prompts_tensor[:,:,0:1,starting_idx:].unfold(3, partial_ct_len+gen_len, stride)

    gen_tensor = full_tensor[:,:,:,prompt_length:] # (rollout_batch_size // micro_rollout_batch_size, n_samples_per_prompt, micro_rollout_batch_size, num_blocks * gen_len)

    # last dim looks like A1, B1, C1, A2, B2, C2
    gen_tensor = gen_tensor.reshape(gen_tensor.shape[0], gen_tensor.shape[1], gen_tensor.shape[2],  gen_len, num_blocks) # last two dims look like [[A1, B1, C1], [A2, B2, C2]...]
    gen_tensor = gen_tensor.transpose(-1, -2) # last two dims now look like [[A1, A2, ...],[B1,B2,B3]...]

    # this is shape ..., num_blocks, gen_len.
    print(f'ct_tensor shape: {ct_tensor.shape}, gen_tensor shape: {gen_tensor.shape}')
    input_tensor = torch.cat([ct_tensor, gen_tensor], dim = -1) #(rollout_batch_size // micro_rollout_batch_size, n_samples_per_prompt, micro_rollout_batch_size, num_blocks, partial_ct_len + gen_len)

    return decode_tensor(input_tensor.reshape(-1, input_tensor.shape[-1]), tokenizer), decode_tensor(gt_tensor.reshape(-1, gt_tensor.shape[-1]), tokenizer)


def decode_tensor(input_tensor, tokenizer):
    return tokenizer.batch_decode(input_tensor, skip_special_tokens=True)



    



    
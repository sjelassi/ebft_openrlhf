from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_humaneval_data(data, apply_chat_template=None):
    """Preprocess HumanEval data.

    Args:
        data: Raw data dictionary with keys: 'prompt', 'test', 'entry_point'
        apply_chat_template: Function to apply chat template

    Returns:
        tuple: (prompt, test, entry_point)
    """
    # Get the prompt (the function signature + docstring)
    prompt = data["prompt"]

    if apply_chat_template:
        chat = [{"role": "user", "content": prompt}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Get test code and entry point
    test = data["test"]
    solution = data["canonical_solution"]
    entry_point = data["entry_point"]

    return prompt, test, solution, entry_point


class HumanEvalDataset(Dataset):
    """
    Dataset for HumanEval code generation evaluation.

    The HumanEval dataset has a specific structure:
    - prompt: Function signature + docstring
    - test: Test code with check(candidate) function
    - entry_point: Name of the function to test
    - canonical_solution: Reference solution (not used for evaluation)

    Args:
        dataset: HumanEval dataset (e.g., from 'openai/openai_humaneval')
        tokenizer: tokenizer for the model
        strategy: training strategy
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.solutions = []
        self.tests = []
        self.entry_points = []
        self.datasources = []

        for data in tqdm(dataset, desc="Preprocessing HumanEval data", disable=not self.strategy.is_rank_0()):
            prompt, test, solution, entry_point = preprocess_humaneval_data(
                data, apply_chat_template
            )
            self.prompts.append(prompt)
            self.solutions.append(solution)
            self.tests.append(test)
            self.entry_points.append(entry_point)
            # Use task_id as datasource if available, otherwise use 'humaneval'
            self.datasources.append(data.get("task_id", data.get("source", "humaneval")))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        """Returns (datasource, prompt, test, entry_point)"""
        # return self.datasources[idx], self.prompts[idx], self.tests[idx], self.entry_points[idx]
        return {"prompt": self.prompts[idx], "label": self.solutions[idx], "unit_test": self.tests[idx], "entry_point": self.entry_points[idx]}

    def collate_fn(self, batch):
        """Collate function for batching code prompts.

        Simply returns the batch as-is since items are tuples of
        (datasource, prompt, label, unit_tests) that don't need special handling.
        """
        return batch
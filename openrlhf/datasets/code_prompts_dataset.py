from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_code_data(data, input_key="input", label_key=None, unit_tests_key="unit_tests", apply_chat_template=None):
    """Preprocess code data including unit tests.

    Args:
        data: Raw data dictionary
        input_key: Key for input/question in data dict
        label_key: Key for label/answer in data dict
        unit_tests_key: Key for unit tests in data dict
        apply_chat_template: Function to apply chat template

    Returns:
        tuple: (prompt, label, unit_tests)
    """
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]

    # Get label and unit tests
    label = "" if label_key is None else data[label_key]
    unit_tests = data.get(unit_tests_key, [])

    return prompt, label, unit_tests


class CodePromptDataset(Dataset):
    """
    Dataset for code generation evaluation with unit tests.

    Similar to PromptDataset but also includes unit_tests for code execution.

    Args:
        dataset: dataset for code generation model
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
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        unit_tests_key = getattr(self.strategy.args, "unit_tests_key", "unit_tests")
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.unit_tests = []
        self.datasources = []

        for data in tqdm(dataset, desc="Preprocessing code data", disable=not self.strategy.is_rank_0()):
            prompt, label, unit_tests = preprocess_code_data(
                data, input_key, label_key, unit_tests_key, apply_chat_template
            )
            self.prompts.append(prompt)
            self.labels.append(label)
            self.unit_tests.append(unit_tests)
            self.datasources.append(data.get("source", data.get("datasource", "default")))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        """Returns (datasource, prompt, label, unit_tests)"""
        return {"prompt": self.prompts[idx], "label": self.labels[idx], "unit_test": self.unit_tests[idx]}
    #self.datasources[idx], self.prompts[idx], self.labels[idx], self.unit_tests[idx]

    def collate_fn(self, batch):
        """Collate function for batching code prompts.

        Simply returns the batch as-is since items are tuples of
        (datasource, prompt, label, unit_tests) that don't need special handling.
        """
        return batch

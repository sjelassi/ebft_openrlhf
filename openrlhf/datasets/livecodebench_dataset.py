import json
from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_livecodebench_data(data, apply_chat_template=None):
    """Preprocess LiveCodeBench data.

    Args:
        data: Raw data dictionary with keys:
            'question_content' (input),
            'starter_code' (for functional tests),
            'public_test_cases' (JSON string with test cases),
            'metadata' (contains func_name for functional tests)
        apply_chat_template: Function to apply chat template

    Returns:
        tuple: (prompt, starter_code, test_cases, metadata, testtype)
    """
    # Get starter code (for functional tests, empty for stdin tests)
    starter_code = data.get("starter_code", "")

    # Parse the test cases from JSON string
    test_cases_str = data.get("public_test_cases", "[]")
    try:
        test_cases = json.loads(test_cases_str) if isinstance(test_cases_str, str) else test_cases_str
    except json.JSONDecodeError:
        test_cases = []

    # Get metadata (contains func_name for functional tests)
    metadata_str = data.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
    except json.JSONDecodeError:
        metadata = {}

    # Determine test type from the first test case
    testtype = "stdin"  # default
    if test_cases and len(test_cases) > 0:
        testtype = test_cases[0].get("testtype", "stdin")

    # Get the question content
    question_content = data["question_content"]

    # Construct prompt based on test type
    if testtype == "functional" and starter_code:
        # For functional tests: include the starter code (function signature)
        # This helps the model understand it needs to complete a class method
        prompt = question_content + "\n\n" + starter_code
    else:
        # For stdin tests: keep the question as-is
        # The question typically already contains example inputs/outputs
        # which implicitly show the program needs to read stdin and print to stdout
        prompt = question_content

    if apply_chat_template:
        chat = [{"role": "user", "content": prompt}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return prompt, starter_code, test_cases, metadata, testtype


class LiveCodeBenchDataset(Dataset):
    """
    Dataset for LiveCodeBench code generation evaluation.

    The LiveCodeBench dataset has a specific structure:
    - question_content: Problem description
    - starter_code: Function signature (for functional tests, empty for stdin tests)
    - public_test_cases: JSON string with list of test cases
    - metadata: JSON string with func_name (for functional tests)

    Two types of test cases:
    1. testtype == "stdin": Standard input/output tests
       - input: stdin input string
       - output: expected stdout output
       - empty starter_code and metadata

    2. testtype == "functional": Function-based tests
       - input: function arguments (as string)
       - output: expected return value
       - starter_code: class Solution with function signature
       - metadata: contains func_name

    Args:
        dataset: LiveCodeBench dataset (e.g., from 'sam-paech/livecodebench-code_generation_lite')
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
        self.starter_codes = []
        self.test_cases = []
        self.metadatas = []
        self.testtypes = []
        self.datasources = []

        for data in tqdm(dataset, desc="Preprocessing LiveCodeBench data", disable=not self.strategy.is_rank_0()):
            prompt, starter_code, test_cases, metadata, testtype = preprocess_livecodebench_data(
                data, apply_chat_template
            )
            self.prompts.append(prompt)
            self.starter_codes.append(starter_code)
            self.test_cases.append(test_cases)
            self.metadatas.append(metadata)
            self.testtypes.append(testtype)
            # Use task_id or question_id as datasource if available
            self.datasources.append(data.get("question_id", data.get("task_id", data.get("source", "livecodebench"))))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        """Returns (datasource, prompt, starter_code, test_cases, metadata, testtype)"""
        return (
            self.datasources[idx],
            self.prompts[idx],
            self.starter_codes[idx],
            self.test_cases[idx],
            self.metadatas[idx],
            self.testtypes[idx]
        )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to properly handle batching.

        Args:
            batch: List of tuples (datasource, prompt, starter_code, test_cases, metadata, testtype)

        Returns:
            Tuple of lists: (datasources, prompts, starter_codes, test_cases_list, metadatas, testtypes)
        """
        datasources, prompts, starter_codes, test_cases_list, metadatas, testtypes = zip(*batch)
        # Keep everything as lists/tuples without any transformation
        return list(datasources), list(prompts), list(starter_codes), list(test_cases_list), list(metadatas), list(testtypes)

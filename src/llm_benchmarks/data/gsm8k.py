from datasets import load_dataset
from typing import Literal


class GSM8KDataset:
    """
    A basic API wrapper for the gsm8k dataset to be accessed like a standard
    Python iterable.

    Args:
        split (Literal['train', 'test']): The dataset split to load.
                                          Defaults to 'train'.
        config (Literal['main', 'socratic']): The dataset configuration to use.
                                              Defaults to 'main'.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        config: Literal["main", "socratic"] = "main",
        shuffle: bool = False,
    ):
        if split not in ["train", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Please choose 'train' or 'test'."
            )

        if config not in ["main", "socratic"]:
            raise ValueError(
                f"Invalid config '{config}'. Please choose 'main' or 'socratic'."
            )

        try:
            self.dataset = load_dataset("gsm8k", name=config, split=split)
            if shuffle:
                self.dataset = self.dataset.shuffle(seed=42)
        except Exception as e:
            print("Failed to load dataset from Hugging Face.")
            print(
                "Please ensure you have an internet connection and 'datasets' is installed."
            )
            raise e

    def __len__(self):
        """Returns the total number of examples in the loaded split."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Retrieves an example from the dataset at the specified index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict: A dictionary with 'question' and 'answer' keys.
        """
        if not 0 <= idx < len(self.dataset):
            raise IndexError(
                f"Index {idx} is out of range for a dataset of size {len(self.dataset)}."
            )

        return self.dataset[idx]

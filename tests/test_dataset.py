# test_dataset.py

import pytest
from unittest.mock import MagicMock

from llm_benchmarks.data import GSM8KDataset

# --- Unit Tests (Fast, No Network Needed) ---


def test_invalid_split_raises_value_error():
    """
    Ensures that providing an invalid `split` raises a ValueError.
    """
    with pytest.raises(ValueError, match="Invalid split 'validation'"):
        GSM8KDataset(split="validation")


def test_invalid_config_raises_value_error():
    """
    Ensures that providing an invalid `config` raises a ValueError.
    """
    with pytest.raises(ValueError, match="Invalid config 'other'"):
        GSM8KDataset(config="other")


# --- Test Data and Fixtures for Mocking ---
# Define a simple, fake dataset that our mock function will return.
FAKE_DATA = [
    {"question": "Mock Question 1", "answer": "Mock Answer 1"},
    {"question": "Mock Question 2", "answer": "Mock Answer 2"},
]


# Mock the load_dataset function to avoid network calls
@pytest.fixture
def mock_load_dataset(mocker):
    """
    This fixture mocks the `load_dataset` function within the `llm_benchmarks.data` module.
    Any test using this fixture will not call the real `load_dataset`.
    """
    # Create a fake dataset object that mimics a real Hugging Face Dataset
    mock_dataset_obj = MagicMock()
    mock_dataset_obj.__len__.return_value = len(FAKE_DATA)
    mock_dataset_obj.__getitem__.side_effect = lambda idx: FAKE_DATA[idx]

    # The path to patch is crucial: it's where the function is *looked up*.
    # Since your GSM8KDataset class is in `llm_benchmarks.data`, that's the path we use.
    return mocker.patch(
        "llm_benchmarks.data.gsm8k.load_dataset", return_value=mock_dataset_obj
    )


class TestGSM8KDatasetMocked:
    """
    Test suite for mocked scenarios. These tests are fast and run in isolation
    without any network calls.
    """

    def test_initialization_defaults(self, mock_load_dataset):
        """
        Checks if the class calls `load_dataset` with the correct default arguments.
        """
        GSM8KDataset()
        mock_load_dataset.assert_called_once_with("gsm8k", name="main", split="train")

    def test_initialization_with_custom_args(self, mock_load_dataset):
        """
        Checks if the class calls `load_dataset` with specified custom arguments.
        """
        GSM8KDataset(split="test", config="socratic")
        mock_load_dataset.assert_called_once_with(
            "gsm8k", name="socratic", split="test"
        )

    def test_len_method(self, mock_load_dataset):
        """
        Tests if the `__len__` method returns the correct length of the mocked dataset.
        """
        dataset = GSM8KDataset()
        assert len(dataset) == len(FAKE_DATA)  # Should be 2

    def test_getitem_method_valid_index(self, mock_load_dataset):
        """
        Tests if the `__getitem__` method returns the correct item for a valid index.
        """
        dataset = GSM8KDataset()
        assert dataset[0] == FAKE_DATA[0]
        assert dataset[1]["question"] == "Mock Question 2"

    def test_getitem_raises_index_error_out_of_bounds(self, mock_load_dataset):
        """
        Ensures __getitem__ raises an IndexError for an index that is too high.
        """
        dataset = GSM8KDataset()
        with pytest.raises(IndexError, match="Index 10 is out of range"):
            _ = dataset[10]

    def test_getitem_raises_index_error_for_negative_index(self, mock_load_dataset):
        """
        Ensures __getitem__ raises an IndexError for a negative index.
        """
        dataset = GSM8KDataset()
        with pytest.raises(IndexError, match="Index -1 is out of range"):
            _ = dataset[-1]


# --- Integration Tests (Slower, Requires Network on First Run) ---


@pytest.mark.integration
class TestGSM8KDatasetIntegration:
    """
    Test suite for real data loading and interaction.
    These tests will download the actual gsm8k dataset.
    """

    def test_loading_defaults_and_len(self):
        """
        Tests loading the default ('train') split and confirms its known length.
        """
        # The first time this runs, it will download the data
        dataset = GSM8KDataset()

        # The gsm8k 'main' config has a known number of training examples
        assert len(dataset) == 7473

    def test_loading_test_split_and_len(self):
        """
        Tests loading the 'test' split and confirms its known length.
        """
        dataset = GSM8KDataset(split="test")

        # The gsm8k 'main' config has a known number of test examples
        assert len(dataset) == 1319

    def test_getitem_structure_and_type(self):
        """
        Tests if a retrieved item has the correct structure (dict with specific keys)
        and the correct data types for its values (strings).
        """
        dataset = GSM8KDataset(split="test")

        # Get an arbitrary item from the dataset
        item = dataset[100]  # Use an index that is safely within bounds

        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)
        assert len(item["question"]) > 0  # Ensure it's not an empty string

    def test_getitem_raises_index_error_out_of_bounds(self):
        """
        Ensures __getitem__ raises an IndexError when using the real dataset length.
        """
        dataset = GSM8KDataset(split="test")  # Known length is 1319

        with pytest.raises(IndexError):
            # Try to access an item that is far beyond the dataset's size
            _ = dataset[9999]

    def test_dataset_shuffle_integration(self):
        """
        Tests the shuffle functionality of the GSM8KDataset.
        """
        split = "test"
        config = "main"
        num_items_to_check = 3

        # 1. Define expected first few items (questions) when not shuffled
        expected_initial_questions = [
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
            "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
        ]

        # 2. Instantiate with shuffle=False (or default)
        dataset_no_shuffle = GSM8KDataset(split=split, config=config, shuffle=False)
        assert len(dataset_no_shuffle) >= num_items_to_check, "Dataset too small for test"

        initial_items = [dataset_no_shuffle[i]["question"] for i in range(num_items_to_check)]

        assert initial_items == expected_initial_questions, \
            f"Initial items {initial_items} did not match expected {expected_initial_questions}"

        # 3. Instantiate with shuffle=True
        dataset_shuffled = GSM8KDataset(split=split, config=config, shuffle=True)
        assert len(dataset_shuffled) >= num_items_to_check, "Shuffled dataset too small"

        shuffled_items = [dataset_shuffled[i]["question"] for i in range(num_items_to_check)]

        # 4. Assert that the order is different
        # This assertion assumes that shuffling (with seed=42) will change the order
        # of the first `num_items_to_check` items. This is highly probable for any reasonably sized dataset.
        assert initial_items != shuffled_items, \
            f"Shuffled items {shuffled_items} were the same as initial items {initial_items}. Shuffling might not have occurred."

        # 5. Test that shuffling again produces a NEW different order
        dataset_shuffled_again = GSM8KDataset(split=split, config=config, shuffle=True)
        assert len(dataset_shuffled_again) >= num_items_to_check, "Second shuffled dataset too small"
        shuffled_items_again = [dataset_shuffled_again[i]["question"] for i in range(num_items_to_check)]

        # Assert that two separate shuffles produce different results.
        # There's a very small theoretical chance of collision for small N, but highly unlikely here.
        assert shuffled_items != shuffled_items_again, \
            f"Two separate shuffles produced the same question order: {shuffled_items}."

        # 6. Instantiate again with shuffle=False to ensure non-persistence of shuffle state
        dataset_unshuffled_after_shuffle = GSM8KDataset(split=split, config=config, shuffle=False)
        assert len(dataset_unshuffled_after_shuffle) >= num_items_to_check, "Dataset (unshuffled after shuffle) too small"

        unshuffled_questions_after_shuffle = [dataset_unshuffled_after_shuffle[i]["question"] for i in range(num_items_to_check)]

        # Assert that the items match the original predefined list (expected_initial_questions)
        # This ensures that shuffle=True in the previous steps didn't somehow
        # permanently alter the underlying data source for subsequent non-shuffled loads.
        assert unshuffled_questions_after_shuffle == expected_initial_questions, \
            f"Items after re-instantiating with shuffle=False {unshuffled_questions_after_shuffle} " \
            f"did not match original expected questions {expected_initial_questions}."

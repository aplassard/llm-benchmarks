# test_dataset.py

import pytest
from unittest.mock import MagicMock

from my_dataset import GSM8KDataset 

# Define a simple, fake dataset that our mock function will return.
# This mimics the structure of a Hugging Face Dataset object.
FAKE_DATA = [
    {'question': 'Question 1', 'answer': 'Answer 1'},
    {'question': 'Question 2', 'answer': 'Answer 2'},
]

@pytest.fixture
def mock_load_dataset(mocker):
    """
    This pytest fixture mocks the `load_dataset` function.
    It will be used by tests to avoid actual network calls.
    """
    # Create a mock object that behaves like a Hugging Face Dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = len(FAKE_DATA)
    mock_dataset.__getitem__.side_effect = lambda idx: FAKE_DATA[idx]

    # Replace the real `load_dataset` with our mock that returns the fake data
    return mocker.patch('my_dataset.load_dataset', return_value=mock_dataset)


# --- Test Group 1: Initialization and Validation ---

def test_initialization_defaults(mock_load_dataset):
    """
    Tests if the class initializes with default arguments ('train', 'main')
    and calls `load_dataset` correctly.
    """
    GSM8KDataset()  # Initialize with defaults
    mock_load_dataset.assert_called_once_with("gsm8k", name="main", split="train")

def test_initialization_with_custom_args(mock_load_dataset):
    """
    Tests if the class initializes with custom arguments and calls
    `load_dataset` correctly.
    """
    GSM8KDataset(split='test', config='socratic')
    mock_load_dataset.assert_called_once_with("gsm8k", name="socratic", split="test")

def test_invalid_split_raises_value_error():
    """
    Tests that a ValueError is raised for an invalid `split` argument.
    This test does NOT need the mock, as it should fail before `load_dataset` is called.
    """
    with pytest.raises(ValueError, match="Invalid split 'validation'"):
        GSM8KDataset(split='validation')

def test_invalid_config_raises_value_error():
    """
    Tests that a ValueError is raised for an invalid `config` argument.
    """
    with pytest.raises(ValueError, match="Invalid config 'my_config'"):
        GSM8KDataset(config='my_config')


# --- Test Group 2: Core Class Functionality ---

def test_len_method(mock_load_dataset):
    """
    Tests if the __len__ method returns the correct length based on the mock data.
    """
    dataset = GSM8KDataset()
    assert len(dataset) == len(FAKE_DATA)  # Should be 2

def test_getitem_method_valid_index(mock_load_dataset):
    """
    Tests if the __getitem__ method returns the correct item.
    """
    dataset = GSM8KDataset()
    first_item = dataset[0]
    second_item = dataset[1]
    
    assert first_item == FAKE_DATA[0]
    assert second_item == FAKE_DATA[1]
    assert first_item['question'] == 'Question 1'

def test_getitem_raises_index_error_out_of_bounds(mock_load_dataset):
    """
    Tests that __getitem__ raises an IndexError for an out-of-bounds index.
    """
    dataset = GSM8KDataset()
    with pytest.raises(IndexError, match="Index 10 is out of range"):
        _ = dataset[10]

def test_getitem_raises_index_error_for_negative_index(mock_load_dataset):
    """
    Tests that __getitem__ raises an IndexError for a negative index,
    as per our class implementation.
    """
    dataset = GSM8KDataset()
    with pytest.raises(IndexError, match="Index -1 is out of range"):
        _ = dataset[-1]
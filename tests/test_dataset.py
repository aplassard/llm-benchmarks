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
        GSM8KDataset(split='validation')

def test_invalid_config_raises_value_error():
    """
    Ensures that providing an invalid `config` raises a ValueError.
    """
    with pytest.raises(ValueError, match="Invalid config 'other'"):
        GSM8KDataset(config='other')


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
        dataset = GSM8KDataset(split='test')

        # The gsm8k 'main' config has a known number of test examples
        assert len(dataset) == 1319
    
    def test_getitem_structure_and_type(self):
        """
        Tests if a retrieved item has the correct structure (dict with specific keys)
        and the correct data types for its values (strings).
        """
        dataset = GSM8KDataset(split='test')
        
        # Get an arbitrary item from the dataset
        item = dataset[100] # Use an index that is safely within bounds
        
        assert isinstance(item, dict)
        assert 'question' in item
        assert 'answer' in item
        assert isinstance(item['question'], str)
        assert isinstance(item['answer'], str)
        assert len(item['question']) > 0 # Ensure it's not an empty string

    def test_getitem_raises_index_error_out_of_bounds(self):
        """
        Ensures __getitem__ raises an IndexError when using the real dataset length.
        """
        dataset = GSM8KDataset(split='test') # Known length is 1319
        
        with pytest.raises(IndexError):
            # Try to access an item that is far beyond the dataset's size
            _ = dataset[9999]
import unittest

from dataset_reader import DatasetReader


class DataSetReaderTest(unittest.TestCase):
    def test_tensor_conversion(self):
        training_dataset, _ = DatasetReader.read('data/text.txt', 3, 1.0)
        chars_batch = training_dataset.sentences[0]
        tensor = training_dataset.to_tensor(chars_batch)
        self.assertEqual(chars_batch, training_dataset.to_chars(tensor))

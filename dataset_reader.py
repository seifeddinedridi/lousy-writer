from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetReader:

    @staticmethod
    def read(filepath, sequence_length, training_data_split=0.8):
        with open(filepath, 'r', encoding='utf8') as f:
            data = f.read()
        start_index = 0
        sentences = []
        while start_index + sequence_length <= len(data):
            test_dataset_size = min(len(data), start_index + sequence_length)
            sentences.append(data[start_index:test_dataset_size])
            start_index = test_dataset_size
        chars = sorted(set(data))  # all the possible characters
        ctoi = {c: index + 1 for index, c in enumerate(chars)}
        itoc = {index + 1: c for index, c in enumerate(chars)}
        print(f"Number of examples in the dataset: {len(sentences)}")
        permuted_indices = torch.randperm(len(sentences))
        # partition the input with a minimum threshold for the number of test samples
        test_dataset_size = min(500, int(len(sentences) * training_data_split))
        train_sentences = [sentences[i] for i in permuted_indices[:-test_dataset_size]]
        test_sentences = [sentences[i] for i in permuted_indices[-test_dataset_size:]]
        print(f"Number of examples in the training dataset: {len(sentences) - test_dataset_size}")
        print(f"Number of examples in the testing dataset: {test_dataset_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:")
        print(''.join(chars))
        # Create training and test dataset
        return CharacterDataset(chars, train_sentences, ctoi, itoc, sequence_length), \
            CharacterDataset(chars, test_sentences, ctoi, itoc, sequence_length)


class CharacterDataset(Dataset):
    def __init__(self, chars, sentences, ctoi, itoc, sequence_length):
        super(CharacterDataset, self).__init__()
        self.chars = chars
        self.sentences = sentences
        self.ctoi = ctoi
        self.itoc = itoc
        self.sequence_length = sequence_length

    def __getitem__(self, idx):
        batch = self.sentences[idx]
        in_tensor = self.to_tensor(batch[:-1])
        target = self.to_tensor(batch[1:], -1)
        return in_tensor, target

    def __len__(self):
        return len(self.sentences)

    def to_tensor(self, chars, padding_value=0):
        return torch.tensor([self.ctoi[c] if idx < len(chars) else padding_value for idx, c in enumerate(chars)])

    def to_chars(self, tensor):
        return ''.join([self.itoc[tensor[i].item()] for i in range(tensor.shape[0])])

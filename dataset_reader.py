import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetReader:

    @staticmethod
    def read(filepath, sequence_length, training_data_split=0.8):
        with open(filepath, 'r') as f:
            data = f.read()
        start_index = 0
        sentences = []
        while start_index < len(data):
            end_index = min(len(data), start_index + sequence_length)
            sentences.append(data[start_index:end_index])
            start_index = end_index
        chars = sorted(list(set(''.join(data))))  # all the possible characters
        ctoi = {c: index + 1 for index, c in enumerate(chars)}
        itoc = {index + 1: c for index, c in enumerate(chars)}
        print(f"Number of examples in the dataset: {len(sentences)}")
        end_index = math.floor(len(sentences) * training_data_split)
        print(f"Number of examples in the training dataset: {end_index}")
        print(f"Number of examples in the testing dataset: {len(sentences) - end_index}")
        print(f"Sequence length: {sequence_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:")
        print(''.join(chars))
        # Create training and test dataset
        return CharacterDataset(chars, sentences[0:end_index], ctoi, itoc, sequence_length), \
            CharacterDataset(chars, sentences[end_index: len(sentences)], ctoi, itoc, sequence_length)


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
        in_tensor = self.to_tensor(batch, starting_index=1)
        target = self.to_tensor(batch, -1.0, 0)
        return in_tensor, target

    def __len__(self):
        return len(self.sentences)

    def to_tensor(self, chars, padding_value=0.0, starting_index=0):
        tensor = torch.zeros(self.sequence_length + 1, dtype=torch.long)
        tensor[0:starting_index] = padding_value
        tensor[starting_index:len(chars) + starting_index] = torch.tensor([self.ctoi[c] for c in chars])
        tensor[len(chars) + starting_index:] = padding_value
        return tensor

    def to_chars(self, tensor):
        return ''.join([self.itoc[tensor[i].item()] for i in range(tensor.shape[0])])

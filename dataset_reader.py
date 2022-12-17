import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetReader:

    @staticmethod
    def read(filepath, sequence_length=10, training_data_split=0.8):
        with open(filepath, 'r') as f:
            data = f.read()
        start_index = 0
        sentences = []
        while start_index < len(data):
            sentences.append(data[start_index:min(len(data), start_index + sequence_length + 1)])
            start_index += sequence_length + 1
        chars = sorted(list(set(''.join(data))))  # all the possible characters
        ctoi = {c: index for index, c in enumerate(chars)}
        itoc = {index: c for index, c in enumerate(chars)}
        print(f"number of examples in the dataset: {len(sentences)}")
        print(f"Sequence length: {sequence_length}")
        print(f"number of unique characters in the vocabulary: {len(chars)}")
        print("vocabulary:")
        print(''.join(chars))
        # Create training and test dataset
        end_index = math.floor(len(sentences) * training_data_split)
        return CharacterDataset(chars, sentences[0:end_index], ctoi, itoc, sequence_length), \
            CharacterDataset(chars, sentences[end_index + 1: len(sentences)], ctoi, itoc, sequence_length)


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
        target = self.to_tensor(batch[1:], -1.0)
        return in_tensor, target

    def __len__(self):
        return len(self.sentences)

    def to_tensor(self, chars, padding_value=0.0):
        tensors = torch.zeros(self.sequence_length, dtype=torch.long)
        tensors[0:len(chars)] = torch.tensor([self.ctoi[c] for c in chars], dtype=torch.long)
        tensors[len(chars):] = padding_value
        return tensors

    def to_chars(self, tensor):
        return ''.join([self.itoc[tensor[i].item()] for i in range(tensor.shape[0])])

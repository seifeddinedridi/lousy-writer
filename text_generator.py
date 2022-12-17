import argparse
from datetime import datetime
from time import time

import torch as torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from dataset_reader import DatasetReader
from rnn import RNN, ModelConfig


def train_model(model, optimizer, max_epoch, training_dataset, testing_dataset):
    model.train()
    dataset_loader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset), batch_size=16, num_workers=0,
                                pin_memory=True)
    dataset_iter = iter(dataset_loader)
    start = time()
    for epoch in range(max_epoch):
        in_tensor, target = get_next_sample(dataset_iter, dataset_loader)
        # target has shape (batch_size, sequence_length)
        out_tensor = model(in_tensor)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        if epoch % 1000 == 0:
            print(f'Epoch={epoch + 1}/{max_epoch}, Loss={loss.item()}')
        end = time()
        if epoch >= max_epoch - 1 or end - start >= 15 * 60:  # 15 minutes
            start = end
            eval_model(model, testing_dataset)
            filepath = f'pretrained_model/rnn_{epoch}.pt'
            print(f'Saving training checkpoint as {filepath} at {datetime.now()}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, filepath)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_model(model, testing_dataset):
    model.train(False)
    dataset_loader = DataLoader(testing_dataset, sampler=RandomSampler(testing_dataset), batch_size=16,
                                num_workers=0)
    dataset_iter = iter(dataset_loader)
    average_loss = 0
    max_epoch = 100
    for epoch in range(max_epoch):
        in_tensor, target = get_next_sample(dataset_iter, dataset_loader)
        # target has shape (batch_size, sequence_length)
        out_tensor = model(in_tensor)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        average_loss += loss.item()
    print(f'Evaluation Loss={average_loss / max_epoch}')
    model.train()


def get_next_sample(dataset_iter, dataset_loader):
    try:
        in_tensor, target = next(dataset_iter)
    except StopIteration:
        dataset_iter = iter(dataset_loader)
        in_tensor, target = next(dataset_iter)
    return in_tensor, target


@torch.no_grad()
def generate_sample_text(model, training_dataset, sequence_length=1000):
    model.train(False)
    in_tensor = torch.zeros((1, 1), dtype=torch.long)
    in_tensor[0, 0] = training_dataset.ctoi['T']
    cumulated_tensor = torch.zeros((1, 1), dtype=torch.long)
    cumulated_tensor[0, 0] = in_tensor[0, 0]
    for idx in range(sequence_length):
        out_tensor = model(in_tensor)
        probs = F.softmax(out_tensor[:, -1, :] / 1.0, dim=-1)
        char_indices = torch.multinomial(probs, num_samples=1)
        cumulated_tensor = torch.cat((cumulated_tensor, char_indices), dim=1)
        in_tensor = char_indices.expand(1, -1)
    model.train()
    return training_dataset.to_chars(cumulated_tensor[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Train the recurrent neural network")
    parser.add_argument("--te", "--training-epoch", type=int, default=1e6, help="Training epoch")
    parser.add_argument("--eval", action='store_true',
                        help="Evaluate the recurrent neural network based on a predefined model at the "
                             "given checkpoint (if it exists)")
    parser.add_argument("--epoch",
                        type=int,
                        help="Evaluate the recurrent neural network based on a predefined model at the "
                             "given checkpoint (if it exists)")
    args = parser.parse_args()
    if args.eval and args.epoch is None:
        parser.print_help()
        exit(1)
    training_dataset, testing_dataset = DatasetReader.read('data/english_text.txt', sequence_length=100,
                                                           training_data_split=0.8)
    config = ModelConfig(in_features=len(training_dataset.chars), out_features=len(training_dataset.chars))
    model = RNN(config)
    if args.train:
        print(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-6)
        train_model(model, optimizer, args.te, training_dataset, testing_dataset)
        eval_model(model, testing_dataset)
    else:
        checkpoint = torch.load(f'pretrained_model/rnn_{args.epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        eval_model(model, testing_dataset)
        print(generate_sample_text(model, training_dataset, 5000))


if __name__ == '__main__':
    main()

import argparse
from datetime import datetime
from time import time

import torch as torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_reader import DatasetReader
from model import ModelConfig, MultiLayerRNN


def train_model(model, optimizer, max_epoch, training_dataset, testing_dataset, start_epoch=0):
    if start_epoch > 0:
        checkpoint = torch.load(f'pretrained_model/rnn_{start_epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    dataset_loader = DataLoader(training_dataset, batch_size=64,
                                num_workers=0,
                                pin_memory=True, shuffle=False)
    dataset_iter = iter(dataset_loader)
    best_loss = float('inf')
    for epoch in range(start_epoch, max_epoch):
        dataset_iter, (in_tensor, target) = get_next_sample(dataset_iter, dataset_loader)
        # (batch_size, sequence_length), (batch_size, sequence_length)
        out_tensor, _ = model(in_tensor)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        print(f'Epoch={epoch + 1}/{max_epoch}, Loss={loss.item()}')
        if epoch > 0 and epoch % 10 == 0:
            eval_loss = eval_model(model, testing_dataset)
            if eval_loss < best_loss:
                save_training_checkpoint(model, optimizer, loss, epoch, max_epoch)
                best_loss = eval_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_training_checkpoint(model, optimizer, loss, epoch, max_epoch):
    print(f'Epoch={epoch + 1}/{max_epoch}, Loss={loss.item()}')
    filepath = f'pretrained_model/rnn_{epoch}.pt'
    print(f'Saving training checkpoint as {filepath} at {datetime.now()}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


@torch.no_grad()
def eval_model(model, testing_dataset):
    model.train(False)
    dataset_loader = DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=0)
    dataset_iter = iter(dataset_loader)
    average_loss = 0
    max_epoch = 10
    for epoch in range(max_epoch):
        dataset_iter, (in_tensor, target) = get_next_sample(dataset_iter, dataset_loader)
        out_tensor, _ = model(in_tensor)  # (batch_size, sequence_length)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        average_loss += loss.item()
    average_loss /= max_epoch
    print(f'Evaluation Loss={average_loss}')
    model.train()
    return average_loss


def get_next_sample(dataset_iter, dataset_loader):
    try:
        return dataset_iter, next(dataset_iter)
    except StopIteration:
        dataset_iter = iter(dataset_loader)
        print('Dataset iterator was reset')
        return dataset_iter, next(dataset_iter)


@torch.no_grad()
def generate_sample_text(model, training_dataset, text_length=1000, temperature=1.0):
    model.train(False)
    seed_text = 'First Citizen:'
    char_indices, cumulated_tensor = sample(model, training_dataset, seed_text, temperature)
    for idx in range(text_length):
        char_indices, out_tensor = sample(model, training_dataset, char_indices, temperature)
        cumulated_tensor = torch.cat((cumulated_tensor, out_tensor), dim=1)
    model.train()
    return training_dataset.to_chars(cumulated_tensor[0, 1:])


def sample(model, training_dataset, text, temperature=1.0):
    model.train(False)
    in_tensor = torch.zeros((1, len(text) + 1), dtype=torch.long)
    in_tensor[0, 1:] = torch.tensor([training_dataset.ctoi[c] for c in text], dtype=torch.long)
    out_tensor, _ = model(in_tensor)
    probs = F.softmax(out_tensor[:, -1:, :] / temperature, dim=-1)
    char_indices = torch.multinomial(probs[0], num_samples=1)
    model.train()
    if char_indices[0, 0] == 0:
        return '', char_indices[0, 1:]
    return training_dataset.to_chars(char_indices[0]), char_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Train the recurrent neural network")
    parser.add_argument("--me", "--max-epoch", type=int, default=1e6, help="Training epoch")
    parser.add_argument("--eval", action='store_true',
                        help="Evaluate the recurrent neural network based on a predefined model at the "
                             "given checkpoint (if it exists)")
    parser.add_argument("--from-epoch",
                        type=int,
                        default=0,
                        help="Evaluate the recurrent neural network based on a predefined model at the "
                             "given checkpoint (if it exists). "
                             "If the argument --train is set, this would resume training from that specific epoch")
    args = parser.parse_args()
    if args.eval and args.from_epoch == 0:
        parser.print_help()
        exit(1)

    torch.manual_seed(int(time()))
    training_dataset, testing_dataset = DatasetReader.read('data/tinyshakespeare.txt', sequence_length=100,
                                                           training_data_split=0.7)
    config = ModelConfig(in_features=len(training_dataset.chars) + 1, out_features=len(training_dataset.chars) + 1,
                         hidden_size=512, layers_count=2, cell_type='lstm')
    model = MultiLayerRNN(config)
    print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    if args.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-6)
        train_model(model, optimizer, args.me, training_dataset, testing_dataset, args.from_epoch)
    else:
        checkpoint_filepath = f'pretrained_model/rnn_{args.from_epoch}.pt'
        checkpoint = torch.load(checkpoint_filepath)
        print(f'Checkpoint file {checkpoint_filepath} successfully loaded')
        model.load_state_dict(checkpoint['model_state_dict'])
        eval_model(model, testing_dataset)
        print(generate_sample_text(model, training_dataset, 1000, 0.8))


if __name__ == '__main__':
    main()

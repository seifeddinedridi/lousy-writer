import argparse
import itertools
from datetime import datetime
from time import time

import torch as torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from dataset_reader import DatasetReader
from model import ModelConfig, MultiLayerRNN, PyTorchRNN


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_model(model, optimizer, max_epoch, training_dataset, testing_dataset, batch_size, start_epoch=0):
    if start_epoch > 0:
        checkpoint = torch.load(f'pretrained_model/rnn_{start_epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    dataset_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True,
                                shuffle=False)
    dataset_iter = itertools.cycle(iter(dataset_loader))
    best_loss = float('inf')
    last_tick = time()
    h = model.init_hidden(batch_size)
    for epoch in range(start_epoch, max_epoch):
        in_tensor, target = next(dataset_iter)
        # (batch_size, sequence_length), (batch_size, sequence_length)
        out_tensor, h = model(in_tensor, h)
        h = repackage_hidden(h)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch={epoch + 1}/{max_epoch}, Loss={loss.item()}')
        now = time()
        if now - last_tick >= 10:
            last_tick = now
            eval_loss = eval_model(model, testing_dataset, batch_size)
            if eval_loss < best_loss:
                save_training_checkpoint(model, optimizer, loss, epoch, max_epoch)
                best_loss = eval_loss


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
def eval_model(model, testing_dataset, batch_size):
    model.train(False)
    torch.manual_seed(int(time()))
    dataset_loader = DataLoader(testing_dataset, sampler=RandomSampler(testing_dataset), batch_size=batch_size,
                                shuffle=False, num_workers=0, drop_last=True)
    dataset_iter = itertools.cycle(iter(dataset_loader))
    average_loss = 0
    max_epoch = 10
    h = model.init_hidden(batch_size)
    for epoch in range(max_epoch):
        in_tensor, target = next(dataset_iter)
        out_tensor, h = model(in_tensor, h)  # (batch_size, sequence_length)
        h = repackage_hidden(h)
        loss = F.cross_entropy(out_tensor.view(-1, out_tensor.size(-1)), target.view(-1), ignore_index=-1)
        average_loss += loss.item()
    average_loss /= max_epoch
    print(f'Evaluation Loss={average_loss}')
    model.train()
    return average_loss


@torch.no_grad()
def generate_sample_text(model, training_dataset, text_length=1000, temperature=1.0):
    model.train(False)
    seed_text = 'ROMEO:\n'
    state = model.init_hidden(1)
    char_indices, state = sample(model, state, training_dataset, seed_text, temperature)
    cumulated_tensor = char_indices.clone()
    for idx in range(text_length):
        char_indices, state = sample(model, state, training_dataset, training_dataset.to_chars(char_indices),
                                     temperature)
        cumulated_tensor = torch.cat((cumulated_tensor, char_indices), dim=0)
    model.train()
    return training_dataset.to_chars(cumulated_tensor)


def sample(model, state, training_dataset, text, temperature):
    in_tensor = training_dataset.to_tensor(text).expand(1, -1)
    out_tensor, h = model(in_tensor, state)
    h = repackage_hidden(h)
    out_tensor = out_tensor[:, -1, :] / temperature
    # Apply masking
    # v, _ = torch.topk(out_tensor, k=out_tensor.shape[1])
    # out_tensor[out_tensor < v[:, -1]] = -float('Inf')
    # Apply the softmax max
    # probs = F.softmax(out_tensor, dim=-1)
    probs = torch.distributions.Categorical(logits=out_tensor)
    char_indices = probs.sample((1,))
    # char_indices = torch.multinomial(probs, num_samples=1)
    return char_indices, h


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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    if args.eval and args.from_epoch == 0:
        parser.print_help()
        exit(1)

    torch.manual_seed(int(time()))
    training_dataset, testing_dataset = DatasetReader.read('data/tinyshakespeare.txt', sequence_length=100,
                                                           training_data_split=0.7)
    config = ModelConfig(in_features=len(training_dataset.chars) + 1, out_features=len(training_dataset.chars) + 1,
                         hidden_size=1024, layers_count=1, cell_type='lstm')
    model = MultiLayerRNN(config)
    print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    if args.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-6)
        train_model(model, optimizer, args.me, training_dataset, testing_dataset, args.batch_size, args.from_epoch)
    else:
        checkpoint_filepath = f'pretrained_model/rnn_{args.from_epoch}.pt'
        checkpoint = torch.load(checkpoint_filepath)
        print(f'Checkpoint file {checkpoint_filepath} successfully loaded')
        model.load_state_dict(checkpoint['model_state_dict'])
        eval_model(model, testing_dataset, args.batch_size)
        temperatures = [1.0]
        for temperature in temperatures:
            print('-------------------------------------------------------------')
            print(generate_sample_text(model, training_dataset, 1000, temperature))
            print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()

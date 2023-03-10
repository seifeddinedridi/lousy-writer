# Vanilla Recurrent Neural Network

RNN written with [PyTorch](https://pytorch.org). RNN is a simple language model for generating text. In the first training phase, the model is given a sequence of characters (or a batch of strings with a fixed length), then the model tries to predict the next characters one by one in an attempt to generate similar looking text. In the next testing phase, the model is given a character and asked produce the next one. The latter process continues until the desired number of characters in the output text is generated.

The model can be trained on any sort of text: novel, source code, poetry, etc. Feel free to add more training data in the `/data` folder and test it out yourself.

## Install dependencies

```bash
pip install -r requirements.txt
```

Create the directory where the model state will be persisted:
```bash
mkdir pretrained_model
```

## Training the model

The model state is saved in the `pretrained_model` folder every time it performs better against the validation dataset. The format of the checkpoint' filename is: `rnn_{epoch}.pt`.

```bash
python text_generator.py --train --max-epoch 10000
```

## Generate text the model

```bash
python text_generator.py --eval --from-epoch 10000
```

The quality of the generated text has still room for improvements, this is probably related to the slow decrease in the loss function below the 1.0 mark. Also the lost is still quite high (~1.9) when the model is run against the testing dataset.

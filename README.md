# Deeping-Learning-Exercise-2

* Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

* Imports

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
```

* Check GPU availability

```python
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
    torch.device('cuda')
else:
    print('No GPU available, training on CPU.')
```

* Load data from Google drive and prepare. Here we refer to the data preparation demonstrated in the Zaremba RNN Regularization repository, which loads the train/validation/test data and counts each word's occurence in each set. We also include Zaremba's minibatch function for consistency with his model architecture and run parameters (we will define minibatches of size 20).

```python
def data_init():
    with open("/content/drive/MyDrive/ex2_332450014_906943097/PTB/ptb.train.txt") as f:
        file = f.read()
        trn = file[1:].split(' ')
    with open("/content/drive/MyDrive/ex2_332450014_906943097/PTB/ptb.valid.txt") as f:
        file = f.read()
        vld = file[1:].split(' ')
    with open("/content/drive/MyDrive/ex2_332450014_906943097/PTB/ptb.test.txt") as f:
        file = f.read()
        tst = file[1:].split(' ')
    words = sorted(set(trn))
    char2ind = {c: i for i, c in enumerate(words)}
    trn = [char2ind[c] for c in trn]
    vld = [char2ind[c] for c in vld]
    tst = [char2ind[c] for c in tst]
    return np.array(trn).reshape(-1, 1), np.array(vld).reshape(-1, 1), np.array(tst).reshape(-1, 1), len(words)

def minibatch(data, batch_size, seq_length):
    data = torch.tensor(data, dtype = torch.int64)
    num_batches = data.size(0)//batch_size
    data = data[:num_batches*batch_size]
    data=data.view(batch_size,-1)
    dataset = []
    for i in range(0,data.size(1)-1,seq_length):
        seqlen=int(np.min([seq_length,data.size(1)-1-i]))
        if seqlen<data.size(1)-1-i:
            x=data[:,i:i+seqlen].transpose(1, 0)
            y=data[:,i+1:i+seqlen+1].transpose(1, 0)
            dataset.append((x, y))
    return dataset
```

*  We now define the embedding and linear objects used within the LSTM, in the same manner as Zaremba in his RNN Regularization repository. The Linear object is responsible for performing the affine transform Wx+b for weigtht and bias tensors W and b, and the Embed object is responsible for performing the embedding of the words into the model vocabulary.

```python
class Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC(input: {}, output: {})".format(self.input_size, self.hidden_size)

class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {})".format(self.vocab_size, self.embed_size)
```
$$\text{Perplexity} = \exp \left( \frac{1}{M} \sum_{m=1}^{M} \text{NLL}(\mathbf{s}^{(m)}, \mathbf{y}^{(m)}) \right)$$
where, $$\mathbf{s}^{(m)}$$ and $$\mathbf{y}^{(m)}$$ are the predicted scores and target sequence for the m\-th sample in the test dataset.

We define the performance functions in an equivalent manner to Zaremba, first calculating the negative log-likelihood loss and with it the perplexity.

The negative log-likelihood predicts the true word over all words using a softmax function, given the predicted scores calculated over all previously predicted word.

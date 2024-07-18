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
where, $$\mathbf{s}^{(m)}$$ and $$\mathbf{y}^{(m)}$$ are the predicted scores and target sequence for the m\-th sample in the test dataset.We define the performance functions in an equivalent manner to Zaremba, first calculating the negative log-likelihood loss and with it the perplexity.The negative log-likelihood predicts the true word over all words using a softmax function, given the predicted scores calculated over all previously predicted word.

```python
def nll_loss(scores, y):
    batch_size = y.size(1)
    expscores = scores.exp()
    probabilities = expscores / expscores.sum(1, keepdim = True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    return torch.mean(-torch.log(answerprobs) * batch_size)

def perp_calc(data, model):
    with torch.no_grad():
        losses = []
        states = model.state_init(batch_size, model.LSTM)
        for x, y in data:
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            losses.append(loss.data.item()/batch_size)
    return np.exp(np.mean(losses))
```

* Next we define a model object which can be used to generate a model for any of the four requires setups. This model is defined as in Zaremba's Github with the additional LSTM Boolean argument which defines the model with either LSTM architecture or GRU architecture.

```python
class RNNModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 layer_num,
                 dropout,
                 init_vat,
                 LSTM):

        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.init_val = init_val
        self.embed = Embed(vocab_size, hidden_size)
        self.LSTM = LSTM
        if LSTM:
          self.rnns = [nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
        # If not LSTM then we use GRU
        else:
          self.rnns = [nn.GRU(hidden_size, hidden_size) for i in range(layer_num)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    # Initializes model parameters to a uniform distribution between -init_val and init_val
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.init_val, self.init_val)

    # Initializes the hidden and cell states to 0
    def state_init(self, batch_size, LSTM):
        dev = next(self.parameters()).device
        # LSTM initialization
        if LSTM:
            states = [(torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        # GRU initialization
        else:
            states = [(torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states

    # Detachment of hidden and cell states from gradient computation graph
    def detach(self, states, LSTM):
      # GRU layer contains only hidden states
      if not LSTM:
        return [h.detach() for h in states]
      else:
        return [(h.detach(), c.detach()) for (h,c) in states]

    # Model's forward pass - embedding of input words, dropout on embedding, embedded
    # input pass through hidden layers (200 in our case) with dropout applied between
    # layers, returning scores at the fully connected layer
    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
```

* Next we define the train function which trains the data and returns numpy arrays of the train and test data perplexity performance

```python
def train(data, model, epoch, step_size, decayRate, eLimit, learning_rate, LSTM):
  # Seperate the data
  train, val, test = data
  train_perp = []
  val_perp   = []
  test_perp  = []

  # Number of words dependent on minibatch
  total_words = 0

  for i in range(epoch):
    # initialize hidden and cell states for new epoch
    states = model.state_init(batch_size, LSTM)
    # train the model
    model.train()
    # learning rate decay
    if (i % step_size == 0) & (i >= eLimit):
      learning_rate = learning_rate / decayRate
    for j, (x, y) in enumerate(train):
      total_words += x.numel()
      # gradient reset for given minibatch
      model.zero_grad()
      states = model.detach(states, LSTM)
      scores, states = model.forward(x, states)

      # Clip gradients to prevent exploding gradients
      nn.utils.clip_grad_norm_(model.parameters(), max_norm)
      # perplexity calculated as exponent of cross-entropy loss
      loss = nll_loss(scores, y)
      # back-propagation
      loss.backward()

      # don't save gradients at hidden layers
      with torch.no_grad():
        # clip gradients above max_norm level to prevent exploding gradients
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        for param in model.parameters():
          # upgrade model parameters according to the found gradients
          param -= learning_rate * param.grad

    model.eval()
    print("Epoch : {:d}".format(i+1) +
          " lr = {:.3f}, ".format(learning_rate))
          #" train = {:.3f}, ".format(perp_calc(train, model)) +
          #" validation = {:.3f}, ".format(perp_calc(val, model)) +
          #" test = {:.3f}, ".format(perp_calc(test, model)))

    train_perp.append(perp_calc(train, model))
    val_perp.append(perp_calc(val, model))
    test_perp.append(perp_calc(test, model))

  return train_perp, val_perp, test_perp
```

* Model Constant Parameter Definitions - Initially chosen according to the default

```python
hidden_size   = 200
layer_num     = 2
init_val      = 0.1
batch_size    = 20
learning_rate = 1
seq_length    = 35
epochs        = [13, 39]
max_norm      = 5
#decayRate     = 1.5
step_size     = 1
dropout       = 0.35
```

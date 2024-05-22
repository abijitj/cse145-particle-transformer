import numpy as np
import tensorflow as tf
import keras as k
from tqdm import tqdm
from qkeras import QDense

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 96 # what is the maximum context length for predictions?
eval_interval = 200
learning_rate = 1e-3 #3e-3
epochs = 25
steps_per_epoch = 500
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
eval_iters = 30
n_embd = 256
n_head = 5
dropout = 0.2
n_layer = 5

tf.random.set_seed(0)
# ------------



np.random.seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = np.array(encode(text), dtype=np.dtype('float_'))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(low=0, high=len(data) - block_size, size=batch_size)
    x = np.stack([data[i:i+block_size] for i in ix], dtype=np.float32)
    y = np.stack([data[i+1:i+block_size+1] for i in ix], dtype=np.float32)
    with tf.device(device):
        x = tf.identity(x)
        y = tf.identity(y) 
        #x, y = x.to(device), y.to(device)
    return x, y

def get_training_data():
    ix = len(train_data) - block_size

    x = np.stack([train_data[i:i+block_size] for i in range(ix)], dtype=np.float32)
    y = np.stack([train_data[i+1:i+block_size+1] for i in range(ix)], dtype=np.float32)
    with tf.device(device):
        x = tf.identity(x)
        y = tf.identity(y) 
    return x, y

def estimate_loss():
    out = {}
    # set model to evaluation mode
    for layer in model.layers: 
        layer.trainable = False
        layer.training = False

    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, targets=Y, training=False)
            losses[k] = loss.numpy().mean()
        out[split] = losses.mean()
    
    # set model back to training mode
    for layer in model.layers: 
        layer.trainable = True
        layer.training = True
    
    return out


class Head(k.Model):
    def build(self, input_shape):
        pass
    def __init__(self, head_size): 
        super().__init__()
        self.key = QDense(head_size, use_bias=False)

        self.transpose = k.layers.Permute((2, 1))

        self.query = QDense(head_size, use_bias=False)
        self.value = QDense(head_size, use_bias=False)
        
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x):
        B, T, C = x.shape
        K = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ self.transpose(K) * C**-0.5 # (B, T, C)
        #tf.
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        tril = tf.convert_to_tensor(np.tril(np.ones((T, T), dtype='float_'), 0), dtype=tf.float32)
        ninf = tf.constant(float('-inf'), dtype=tf.float32)
        wei = tf.where(tril[:T, :T] == 0, ninf, wei) # (B, T, T)
        wei = k.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class MultiHeadAttention(k.Model): 
    """ multiple heads of self-attention in parallel """
    def build(self, input_shape):
        pass
    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = QDense(n_embd)
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x):
        out = k.layers.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(k.Model):
    """ a simple linear layer followed by a non-linearity """
    def build(self, input_shape):
        pass
    def __init__(self, n_embd): 
        super().__init__()
        self.l1 = QDense(4 * n_embd, activation='relu')
        #self.relu = k.layers.Activation('relu')
        self.l2 = QDense(n_embd)
        self.dropout = k.layers.Dropout(dropout)
    
    def call(self, x): 
        x = self.l1(x)
        x = self.dropout(self.l2(x))
        return x


class Block(k.Model): 
    """ Transformer block: communication followed by computation """
    def build(self, input_shape):
        pass
    def __init__(self, n_embd, n_head): 
        super().__init__()
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = k.layers.LayerNormalization()
        self.ln2 = k.layers.LayerNormalization()
    
    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        #x = x + self.ln1(x)
        return x
    

class GPTModel(k.Model): 
    """ GPT Decoder-only Model """
    def build(self, input_shape):
        pass

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = k.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = k.layers.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = k.layers.LayerNormalization(n_embd) # final layer norm
        self.lm_head = QDense(vocab_size)
    
    def call(self, idx, training=True, targets=None): 
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(np.arange(T))#, device=device)) # (T,C)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            #B, T, C = logits.shape
            #logits = tf.reshape(logits, (B*T, C))
            loss = None
        else:
            B, T, C = logits.shape
            
            logits = tf.reshape(logits, (B*T, C))
            targets = tf.reshape(targets, (B*T,1))
            
            loss = k.losses.SparseCategoricalCrossentropy(from_logits=True)(targets, logits)
            #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            #loss = k.losses.CategoricalCrossentropy()(targets, logits)
        if not training:
            return logits, loss
        return logits
        #return loss, logits
        #return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in tqdm(range(max_new_tokens)):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.call(idx_cond, training=False, targets=None)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            # warning on the first call b/c the first input is the number 1 dw about it
            probs = k.activations.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution

            # *.9999 to avoid p array being slightly too large and causing an error
            idx_next = np.random.multinomial(1, probs[0,:] * .9999, vocab_size).argmax() # (B, 1)
            
            idx_next = tf.constant(idx_next, dtype=tf.int32, shape=(1, 1))
            #print(idx, idx_next)
            # append sampled index to the running sequence
            idx = k.layers.concatenate((idx, idx_next), axis=1) # (B, T+1)
        return idx
    
model = GPTModel()

with tf.device(device):
    #loss = k.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=1)
    loss_function = k.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.nn.softmax_cross_entropy_with_logits
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function
    )

    x, y = get_training_data()
    # for iter in tqdm(range(max_iters)):
    #     # every once in a while evaluate the loss on train and val sets
    #     if iter % eval_interval == 0:
    #         losses = estimate_loss()
    #         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #     # sample a batch of data
    #     xb, yb = get_batch('train')

    #     # evaluate the loss
    #     #print(xb.shape, yb.shape)
    #     history = model.fit(xb, yb, epochs=epochs_per_iter, verbose=0)
    print(x.numpy().shape, y.numpy().shape)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_split=.2, validation_batch_size=batch_size, validation_steps=10)
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # generate from the model
    context = np.zeros((1, 1), dtype='float_')
    print(decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist()))
# NanoLM

A language model built from scratch.

## Table of Contents

- [Usage](#usage)
- [Datasets](#datasets)
- [BPE Tokenizer](#bpe-tokenizer)
  - [Pre-tokenization](#pre-tokenization)
  - [BPE Training](#bpe-training)
  - [Optimizations](#optimizations)
- [Transformer](#transformer)
  - [Linear Module](#linear-module)
  - [Embedding Module](#embedding-module)
  - [Root Mean Square Layer Normalization](#root-mean-square-layer-normalization)
  - [SwiGLU Feed-Forward Network](#swiglu-feed-forward-network)
  - [Rotary Positional Embedding](#rotary-positional-embedding)
  - [Softmax](#softmax)
- [Resources](#resources)

## Usage

Install dependencies:
```shell
uv sync
```

Run the project:
```shell
uv run python -m nanolm
```

This trains a BPE tokenizer and prints its merges. The dataset, vocabulary size, and special tokens are set at the top of `src/nanolm/__main__.py`. The tokenizer is saved to `tokenizers/<dataset>_<vocab size>.json` and loaded on later runs; delete the file to retrain.

## Datasets

Using two pre-processed datasets, i.e., [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [OpenWebText](https://huggingface.co/datasets/stanford-cs336/owt-sample).

Create `data` folder if it doesn't exist:
```shell
mkdir -p data
cd data
```

Download the TinyStories data:
```shell
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Download a subsample of OpenWebText:
```shell
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

## BPE Tokenizer

Tokenization turns text into a sequence of integers (numbers). There are several ways to cut text into tokens. In this repo, we will train and implement a byte-level byte-pair encoding (BPE). Later, we will use this tokenizer to encode text into tokens for language modeling.

### Pre-tokenization

Before deriving any BPE merges, we split the raw text into chunks called *pre-tokens* (words, numbers, punctuation runs, spaces) using the same [regex pattern GPT-2 uses](https://github.com/openai/tiktoken/pull/234/changes). This is done for two reasons:

- **Efficiency**: rather than performing the merge operation (iterative loop) on the entire text corpus, we condense this into a frequency distribution of unique pre-tokens in order to save memory space. This is an improvement in the hundreds of times factor since unique words increase much slower than all text.

    Imagine the following corpus:
    ```
    low low low low low lower lower newest newest newest widest widest
    ```
    
    That's 12 pre-tokens which could be collapsed to something like this:
    ```python
    {"low": 5, "lower": 2, "newest": 3, "widest": 2}
    ```
    
    This is now only 4 entries. To count the pair `l - o` we don't need to iterate trough 12 words, but just 4 and multiply:
    ```
    "low"   -> contains "l - o" -> +5
    "lower" -> contains "l - o" -> +2
                                  = 7
    ```
    
    This leads to an identical result with a third of the work.

- **Boundaries**: the merging process occurs within a pre-token and goes no further than that. Hence, the word cannot be joined to the word after it, or even to the comma following it. It is also this which enables the first rule to function since all occurrences of the word can be considered one entry because they are all tokenized alike regardless of context.

    Imagine the following corpus:
    ```
    the cat the cat the cat the dog
    ```
  
    Without pre-tokenization this is just one stream of characters, and the space is treated no differently than any other character:
    ```
    t - h     -> th
    th - e    -> the
    the - ␣   -> the␣
    c - a     -> ca
    ca - t    -> cat
    the␣- cat -> the␣cat
    ```

    The word `cat` is now tokenized one way after `the` and another way elsewhere. With pre-tokenization, the corpus is cut into pieces first:

    ```python 
    {"the": 4, "␣cat": 3, "␣dog": 1}
    ```
  
    The token `the␣cat` cannot be formed because no entry contains both words, so the pair is never counted.

### BPE Training

The tokenizer vocabulary is a one-to-one mapping between byte-string tokens and integer IDs. Since this is a byte-level BPE tokenizer, its initial vocabulary contains every possible byte value. There are 256 values (`0` through `255`), so training always starts with 256 byte tokens, followed by any supplied special tokens.

You decide in advance how large the vocabulary should ultimately be. The 256 initial byte tokens and any special tokens occupy part of that total; each learned merge adds one more token. For a final size of 260 with one special token, the initial 257 tokens leave room for three learned merges: `260 - 256 - 1 = 3`. Training can also stop naturally when no adjacent pairs remain to merge.

Each pre-token is encoded as UTF-8 and split into one-byte symbols. A pre-token such as `"cat"` begins as:

```
c - a - t
```

It cannot start as the single symbol `cat`, because BPE needs separately addressable adjacent symbols to learn which parts should be combined.

Imagine the pre-token counts are:

```python
pretoken_counts = {
    "cat": 2,
    "car": 1,
}
```

Initially, each word is split into individual bytes:

```python
words = {
    "cat": [b"c", b"a", b"t"], # frequency 2
    "car": [b"c", b"a", b"r"], # frequency 1
}
```

The trainer counts every adjacent pair, weighted by how often its word appeared:

```python
(b"c", b"a")  # occurs in "cat" twice and "car" once -> 2 + 1 = 3
(b"a", b"t")  # occurs in "cat" twice -> 2
(b"a", b"r")  # occurs in "car" -> 1
```

Pair counts are stored by pair, so all occurrences of `(b"c", b"a")` contribute to one total rather than becoming duplicate entries. The most  frequent pair is `(b"c", b"a")`, so BPE adds `ca` to the vocabulary with the next available ID and records the merge. It then applies that merge to every pre-token:

```python
words = {
    "cat": [b"ca", b"t"], # frequency 2
    "car": [b"ca", b"r"], # frequency 1
}
```

On the next iteration, the pairs are `(b"ca", b"t")` with count `2` and `(b"ca", b"r")` with count `1`, so it would learn `b"cat"` next. 

This repeats until the requested final vocabulary size is reached, or there are no pairs left to merge. If two pairs have the same frequency, the code deterministically chooses the lexicographically greater byte pair.

### Optimizations

BPE training repeatedly finds the most frequent adjacent pair and merges it. The trainer keeps the information needed for that decision up to date as merges are learned, which keeps each iteration focused on the affected pre-tokens and pairs.

Take this corpus (spaces ignored for simplicity):

```python
pretoken_counts = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
```

Every merge decision depends on pair counts over the whole corpus. Computing them from the raw text would mean walking through every occurrence of every word, again and again. Instead, as described in [Pre-tokenization](#pre-tokenization), the trainer stores each distinct pre-token once together with its frequency, and counts each of its pairs once, weighted by that frequency. Building this table still requires one full pass over the file, so the file is split into chunks at `<|endoftext|>` boundaries, and a pool of worker processes pre-tokenizes those chunks concurrently, with each worker handling one chunk at a time.

After each merge, the pair counts must reflect the new symbol sequences. The straightforward approach is to recount every pair in every pre-token, even though a single merge only changes the few pre-tokens that contain the merged pair. Instead, the trainer computes the counts once, before the first merge, and then keeps them up to date. Alongside the weighted count of every adjacent pair, it records the pre-tokens that contain it. For example, the pair `s - t` occurs nine times: three in `widest` and six in `newest`. A reverse lookup associates that pair with those two pre-tokens:

```
s - t: 9 occurrences
s - t appears in: widest, newest
```

When `s - t` becomes `st`, the reverse lookup shows that only `widest` and `newest` need to be revisited, while `low` and `lower` are left untouched. Their previous pair contributions are removed, their symbol sequences are updated, and their new pair contributions are added. The cost of a merge therefore depends on how many pre-tokens contain the chosen pair, rather than on the size of the corpus.

With the counts kept up to date, each iteration still has to select the pair with the highest count. Scanning the pair counts gives the right answer, but it visits every distinct pair on every merge, and that number keeps growing as new tokens form new pairs: on OpenWebText it reaches 3.2M. Instead, pair counts are also stored in a max-heap, ordered first by count and then by byte pair to break ties deterministically, so the most frequent current candidate is always available at the top of the heap.

Updating the heap on every count change would bring back much of that cost, so the heap is maintained lazily. A decreased count is only corrected when its entry reaches the top. After merging `s - t`, the heap can still contain `e - s` with a priority of 9 even though its current count is 0. When that entry reaches the top, its priority is checked against the current count: stale entries are discarded, and pairs that still occur are added back with their updated count. An increased count, on the other hand, is added immediately. Only pairs adjacent to the newly created token can gain occurrences, so only those are pushed. In this example, `e - st` is pushed with a count of 9 and becomes the next merge, `est`.

As a result, each pair retains a heap entry whose priority is at least its current count. Once the priority at the top matches the current count, it identifies the most frequent pair, and selecting that pair costs O(log n) on average.

Training performance with the `<|endoftext|>` special token, on a 10-core Apple Silicon machine (Python 3.9), with pre-tokenization running on 10 processes:

| Dataset | Vocab size | Pre-tokenization | Unique pre-tokens | Pair-statistics setup | Pair selection | Pair merging | Total |
|---|---|---|---|---|---|---|---|
| TinyStories train (2.2 GB) | 10,000 | 24 s | 60K | 0.1 s | 0.03 s | 1.2 s | ~26 s |
| OpenWebText train (11.9 GB) | 32,000 | 123 s | 6.6M | 24 s | 0.6 s | 375 s | ~523 s |

Picking the best pair is now negligible. On OpenWebText, most of the time is spent applying merges, because the first merges combine very common byte pairs that appear in a large share of all pre-tokens.

These measurements can be reproduced with the `nanolm.profiling` package, which trains the tokenizer under [cProfile](https://docs.python.org/3/library/profile.html) and prints a breakdown of the time spent in each step:
```shell
uv run python -m nanolm.profiling --input-path data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --num-processes 10
uv run python -m nanolm.profiling --input-path data/owt_train.txt --vocab-size 32000 --num-processes 10
```

cProfile only observes the main process, so with more than one process the pre-tokenization workers do not appear individually; their total wall time is still reported under `pretokenize_file`. The full profile is saved to `profiles/train.prof` and can be explored with `uvx snakeviz profiles/train.prof`.

## Transformer

The Transformer language model starts by mapping each token ID to a dense vector with its input embedding. It processes those vectors through `num_layers` Transformer blocks, then uses a learned linear projection—called the output embedding or LM head—to produce logits for the next token.

![transformer.png](docs/transformer.png)

### Linear Module

A linear layer transforms a vector $x$ of size `d_in` into a vector $y$ of size `d_out` by multiplying it with a learned weight matrix $W$ of shape `(d_out, d_in)`:

$$y = Wx$$

Each output value is a weighted sum of all input values, and $W$ holds the weights. Training adjusts them so that the outputs become useful features for the task.

Transformers use linear layers throughout the model. Attention uses them to create queries, keys, and values. The feed-forward network uses them to expand and then shrink each token vector. The final linear layer produces one score, called a logit, for each token in the vocabulary.

A linear layer usually also adds a learned bias, $y = Wx + b$ (i.e., an extra learned number added to each output). Some models (e.g. LLaMA) leave the bias out because, same as this implementation.

In practice, inputs are stored as rows with the feature dimension last, so the layer computes $y = xW^T$:

```
x:   (batch, seq_len, d_in)
W^T: (d_in, d_out)
y:   (batch, seq_len, d_out)
```

Before training, the layer starts with small random weights centered around zero. Their typical size is set by:

$$\sigma = \sqrt{\frac{2}{d_{in} + d_{out}}}$$

Larger layers start with smaller weights: when an output combines many inputs, smaller weights help keep the result from growing too large. Including both dimensions also helps keep the signals used during training at a useful scale.

### Embedding Module

The tokenizer turns text into integer IDs, but an ID on its own carries no meaning: token `1820` is not "closer" to `1821` than to `5`. The embedding layer gives every token a learned vector of `d_model` numbers, and the model uses this vector in the layers that follow.

It is a lookup table with one row per vocabulary token. Each ID selects its row:

```
weight: (vocab_size, d_model)

token ID 1820 -> weight[1820] -> [0.12, -0.53, ..., 0.08]
```

A sequence of IDs becomes a sequence of vectors, adding one dimension at the end:

```
token_ids: (batch, seq_len)
output:    (batch, seq_len, d_model)
```

During training, tokens used in similar contexts get pushed toward similar vectors, so the embedding space ends up encoding meaning: related words lie close together.

The embedding weights start as random numbers centered on zero, with standard deviation `1`. Values beyond `−3` and `3` are excluded. Unlike a linear layer, an embedding lookup returns one stored vector without summing many inputs, so its standard deviation is not scaled by the layer’s dimensions.

### Root Mean Square Layer Normalization

As token vectors flow through a deep stack of blocks, their scale drifts: some grow very large, others shrink toward zero. Layers then receive inputs at unpredictable scales, and training becomes unstable. Normalization fixes this by rescaling each token vector to a consistent size before it enters attention or the feed-forward network.

Root Mean Square Layer Normalization (RMSNorm) divides each token vector $a$ of size `d_model` by its root mean square, then multiplies it by a learned gain $g$:

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} \, g_i  \qquad  \text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$$

For example, with $\varepsilon$ ignored:

```
a      = [3, -4, 0, 12]
RMS(a) = sqrt((9 + 16 + 0 + 144) / 4) = 6.5
a/RMS  = [0.46, -0.62, 0, 1.85]   (RMS = 1)
```

The direction of the vector is preserved, only its overall size changes. Each token is normalized on its own, using only its own `d_model` values.

- **Gain $g$**: one learned value per dimension, initialized to `1`. It lets the model scale individual dimensions back up or down where that helps.
- **Epsilon $\varepsilon$**: a small constant (`1e-5`) that avoids dividing by zero when a vector is all zeros.

RMSNorm is a simplification of LayerNorm, which also subtracts the mean of the vector and adds a learned bias. Dropping these steps reaches about the same quality with less computation, which is why modern LLMs such as LLaMA use RMSNorm.

### SwiGLU Feed-Forward Network

Inside each block, attention lets tokens exchange information with each other. The feed-forward network (FFN) then processes every token on its own: it expands the token vector from `d_model` to a larger size `d_ff`, applies a nonlinearity, and projects it back to `d_model`. The wider intermediate space gives the model room to compute, and a large share of what the model learns is stored in these weights.

The nonlinearity is essential. Without it, the expansion and the projection would collapse into a single linear layer, since $W_2(W_1x) = (W_2W_1)x$, and stacking blocks would add no expressive power.

**SiLU.** The activation function is SiLU (also called Swish):

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Like ReLU, it is close to $x$ for large positive inputs and close to $0$ for large negative ones, but it is smooth around zero, which gives better-behaved gradients.

```
x       = [-4,    -1,    0,  1,    4   ]
SiLU(x) = [-0.07, -0.27, 0,  0.73, 3.93]
```

**Gating.** A gated linear unit (GLU) projects the input twice. One projection goes through the activation and acts as a gate, deciding value by value how much of the other projection passes through:

$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1x) \odot W_3x\big)$$

where $\odot$ is element-wise multiplication. $W_1$ and $W_3$ have shape `(d_ff, d_model)`, and $W_2$ has shape `(d_model, d_ff)`:

```
x:                (batch, seq_len, d_model)
SiLU(W1 x), W3 x: (batch, seq_len, d_ff)
gate * value:     (batch, seq_len, d_ff)
W2 (...):         (batch, seq_len, d_model)
```

Because the gate is computed from the input itself, the network can switch features on and off depending on the token, which in practice trains to better quality than a plain activation. The combination of SiLU and GLU is called SwiGLU, and it is used by LLaMA, PaLM, and most modern LLMs.

**Size.** A classic FFN has two matrices and uses $d_{ff} = 4 \cdot d_{model}$. SwiGLU has three, so $d_{ff}$ is set to about $\frac{8}{3} \cdot d_{model}$ to keep the parameter count roughly the same, rounded to a multiple of 64 for efficient hardware use. For $d_{model} = 512$:

```
classic: 2 × 512 × 2048 = 2,097,152 parameters
SwiGLU:  3 × 512 × 1344 = 2,064,384 parameters
```

### Rotary Positional Embedding

Attention compares every token with every other token, but on its own it has no notion of order: `dog bites man` and `man bites dog` contain the same tokens and would look identical. Information about position has to be injected explicitly.

Rotary Positional Embedding (RoPE) does this by rotating the query and key vectors used by attention, by an angle that depends on the token's position. The vector of size `d_k` is split into pairs of adjacent values, and each pair is treated as a point in 2D and rotated by an angle $\theta_{i,k}$:

$$\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos\theta_{i,k} & -\sin\theta_{i,k} \\ \sin\theta_{i,k} & \cos\theta_{i,k} \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix} \qquad \theta_{i,k} = i \cdot \Theta^{-2k/d_k}$$

where $i$ is the token's position, $k = 0, \dots, d_k/2 - 1$ is the index of the pair, and $\Theta$ is a constant, usually `10000`.

Each pair rotates at its own speed. The first pairs rotate quickly, by about one radian per position, so they distinguish neighboring tokens. The last pairs rotate very slowly, so they change meaningfully only across long distances. Together they encode position at many scales, a bit like the hands of a clock:

```
d_k = 8, Θ = 10000

pair k:            0      1      2      3
angle per step:    1      0.1    0.01   0.001   (radians)
```

**Why rotation.** Attention compares a query $q$ at position $m$ with a key $k$ at position $n$ through their dot product. Rotating both and taking the dot product gives the same result as rotating only by their difference:

$$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$$

The attention score therefore depends on how far apart two tokens are, not on where they sit in the sequence. A query at position `5` attending to a key at position `2` gets exactly the same score as the same vectors at positions `20` and `17`. Language works this way too: what matters is that a word came three tokens earlier, not that it was the 17th token of the document.

RoPE has no learned parameters. Rotations do not change the length of a vector, so they move information about position into the angles without distorting the content. The cosines and sines for every position up to the maximum sequence length are computed once, ahead of time, and looked up for each token.

### Softmax

The model often produces raw scores (*logits*) that need to become a probability distribution: attention turns query-key scores into weights over tokens, and the output layer turns one score per vocabulary token into next-token probabilities. Softmax does this conversion:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Every output is between `0` and `1`, and they sum to `1`. The exponential makes all values positive and preserves their order, and it amplifies differences: a score `1` higher gets about `2.7` times more probability.

```
x          = [2,    1,    0   ]
softmax(x) = [0.67, 0.24, 0.09]
```

Most of the probability goes to the largest score, but every value keeps a share, which is why it is a *soft* maximum: gradients reach all scores, not only the winning one. A score of $-\infty$ gets a probability of exactly `0`, which attention uses to hide tokens.

Applied to a tensor, softmax normalizes along one dimension, independently for every other position. For attention scores of shape `(batch, seq_len, seq_len)`, normalizing along the last dimension gives each query a distribution over the keys.

**Numerical stability.** Exponentials grow very fast: $e^{1000}$ overflows to infinity, and the division then produces $\infty / \infty$, which is undefined. Subtracting the same constant $c$ from every score does not change the result, because the factor $e^{-c}$ cancels out:

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{-c} \, e^{x_i}}{e^{-c} \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Choosing $c = \max(x)$ makes the largest exponent $e^0 = 1$ and every other one smaller, so nothing can overflow:

```
x            = [1000, 1001, 1002]
naive        = [nan,  nan,  nan ]
x - max(x)   = [-2,   -1,   0   ]
softmax(x)   = [0.09, 0.24, 0.67]
```

## Resources

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

# NanoLM

A language model built from scratch.

## Table of Contents

- [Usage](#usage)
- [Datasets](#datasets)
- [BPE Tokenizer](#bpe-tokenizer)
  - [Pre-tokenization](#pre-tokenization)
  - [BPE Training](#bpe-training)
  - [Optimizations](#optimizations)
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

1. Train from pre-token frequencies: As described in [Pre-tokenization](#pre-tokenization), the trainer stores each distinct pre-token once together with its frequency. The file is split into chunks at `<|endoftext|>` boundaries. A pool of worker processes then pre-tokenizes those chunks concurrently, with each worker handling one chunk at a time.

2. Update pair statistics locally: Before the first merge, the trainer records the weighted count of every adjacent pair and the pre-tokens that contain it. For example, the pair `s - t` occurs nine times: three in `widest` and six in `newest`. A reverse lookup associates that pair with those two pre-tokens:

       s - t: 9 occurrences
       s - t appears in: widest, newest

   When `s - t` becomes `st`, only `widest` and `newest` need to be revisited. Their previous pair contributions are removed, their symbol sequences are updated, and their new pair contributions are added. The cost of a merge therefore depends on how many pre-tokens contain the chosen pair.

3. Prioritize the most frequent pair: Pair counts are also stored in a max-heap, ordered first by count and then by byte pair to break ties deterministically. The most frequent current candidate is available at the top of the heap. This matters as the number of distinct pairs grows.

   The heap is maintained lazily to limit work when counts change:
     - Decreased counts are checked at the top. After merging `s - t`, the heap can still contain `e - s` with a priority of 9 even though its current count is 0. When that entry reaches the top, its priority is checked against the current count. Stale entries are discarded; pairs that still occur are added back with their updated count.
     - Increased counts are added immediately. Only pairs adjacent to the newly created token can gain occurrences. In this example, `e - st` has a count of 9 and becomes the next merge, `est`.

   Each pair retains a heap entry whose priority is at least its current count. Once the priority at the top matches the current count, it identifies the most frequent pair. Selecting that pair costs O(log n) on average.

Training performance with the `<|endoftext|>` special token, on a 10-core Apple Silicon machine (Python 3.9), with pre-tokenization running on 10 processes:

| Dataset | Vocab size | Pre-tokenization | Unique pre-tokens | Pair-statistics setup | Pair selection | Pair merging | Total |
|---|---|---|---|---|---|---|---|
| TinyStories train (2.2 GB) | 10,000 | 24 s | 60K | 0.1 s | 0.03 s | 1.2 s | ~26 s |
| OpenWebText train (11.9 GB) | 32,000 | 123 s | 6.6M | 24 s | 0.6 s | 375 s | ~523 s |

Picking the best pair is now negligible. On OpenWebText, most of the time is spent applying merges, because the first merges combine very common byte pairs that appear in a large share of all pre-tokens.


## Resources

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

# NanoLM

A language model built from scratch.

## Table of Contents

- [Usage](#usage)
- [Datasets](#datasets)
- [BPE Tokenizer](#bpe-tokenizer)
  - [Pre-tokenization](#pre-tokenization)
  - [Training BPE](#training-bpe)
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

Before deriving any BPE merges, we split the raw text into chunks called *pre-tokens* (words, numbers, punctuation runs, spaces) using the same regex pattern GPT-2 uses. This is done for two reasons:

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

### Training BPE

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


## Resources

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

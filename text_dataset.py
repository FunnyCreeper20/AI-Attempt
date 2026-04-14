import re

import numpy as np

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def build_training_data(corpus, context_size):
    sentences = split_sentences(corpus)
    training_tokens = []

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if sentence_tokens:
            training_tokens.extend(
                [BOS_TOKEN] * context_size + sentence_tokens + [EOS_TOKEN]
            )

    words = training_tokens
    vocab = sorted(set(words))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    corpus_indices = np.array([word_to_idx[word] for word in words], dtype=int)

    if len(corpus_indices) <= context_size:
        raise ValueError("corpus must contain more tokens than context_size")

    context_indices = []
    target_indices = []

    for end_index in range(context_size, len(corpus_indices)):
        context_indices.append(corpus_indices[end_index - context_size : end_index])
        target_indices.append(corpus_indices[end_index])

    return (
        words,
        vocab,
        word_to_idx,
        idx_to_word,
        np.array(context_indices, dtype=int),
        np.array(target_indices, dtype=int),
    )

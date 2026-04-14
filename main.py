import numpy as np

from neural_language_model import NeuralBigramModel
from text_dataset import BOS_TOKEN, EOS_TOKEN, build_training_data, tokenize


with open("corpus.txt", "r", encoding="utf-8") as corpus_file:
    CORPUS = corpus_file.read()
TRAINING_RNG = np.random.default_rng(7)
HIDDEN_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.08  #A higher number can make it more unstable .05 < x < .15 
BATCH_SIZE = 512      #Controls how many test cases it goes through before updating the weights
TEMPERATURE = 10      #Higher temperature makes the output more random, lower makes it more repetitive but conservative
CONTEXT_SIZE = 4      #This is just the number of words it looks at to predict the next word


def build_context_lookup(context_indices, target_indices, idx_to_word):
    lookup = {}
    for context, target in zip(context_indices, target_indices):
        lookup[tuple(context.tolist())] = idx_to_word[target]
    return lookup


def predict_next_word(
    context_words,
    model,
    word_to_idx,
    idx_to_word,
    rng,
    temperature=1.0,
    context_lookup=None,
):
    context_indices = [word_to_idx[word] for word in context_words]
    if context_lookup is not None:
        exact_match = context_lookup.get(tuple(context_indices))
        if exact_match is not None:
            return exact_match

    probabilities = model.predict_distribution(context_indices, temperature=temperature)
    probabilities[word_to_idx[BOS_TOKEN]] = 0.0
    probability_sum = probabilities.sum()
    if probability_sum == 0:
        next_word_idx = word_to_idx[EOS_TOKEN]
    else:
        probabilities /= probability_sum
        next_word_idx = rng.choice(len(probabilities), p=probabilities)
    return idx_to_word[next_word_idx]


def choose_seed_words(user_text, word_to_idx, fallback_words, context_size):
    prompt_tokens = tokenize(user_text)
    known_prompt_tokens = [token for token in prompt_tokens if token in word_to_idx]

    if len(known_prompt_tokens) >= context_size:
        return known_prompt_tokens[-context_size:]

    missing_count = context_size - len(known_prompt_tokens)
    return fallback_words[-missing_count:] + known_prompt_tokens




words, vocab, word_to_idx, idx_to_word, context_indices, target_indices = build_training_data(
    CORPUS, CONTEXT_SIZE
)

model = NeuralBigramModel(
    vocab_size=len(vocab),
    hidden_size=HIDDEN_SIZE,
    context_size=CONTEXT_SIZE,
    rng=TRAINING_RNG,
)
model.train(
    context_indices,
    target_indices,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
)
context_lookup = build_context_lookup(context_indices, target_indices, idx_to_word)

user_prompt = input("You: ").strip()
default_seed_words = [BOS_TOKEN] * CONTEXT_SIZE
seed_words = choose_seed_words(
    user_prompt,
    word_to_idx,
    default_seed_words,
    CONTEXT_SIZE,
)
generation_rng = np.random.default_rng()

predicted_word = predict_next_word(
    seed_words,
    model,
    word_to_idx,
    idx_to_word,
    generation_rng,
    temperature=TEMPERATURE,
    context_lookup=context_lookup,
)
if predicted_word == EOS_TOKEN:
    predicted_word = ""
print(f"AI: {predicted_word}")

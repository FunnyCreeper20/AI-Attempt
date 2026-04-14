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
TEMPERATURE = 1       #Higher temperature makes the output more random, lower makes it more repetitive but conservative
CONTEXT_SIZE = 4      #This is just the number of words it looks at to predict the next word
alpha = 0.7           #Blend weight for combining neural and lookup probabilities (0 for pure neural, 1 for pure lookup)


def build_context_lookup(context_indices, target_indices, idx_to_word):
    lookup = {}

    for context, target in zip(context_indices, target_indices):
        key = tuple(context.tolist())
        word = idx_to_word[target]

        if key not in lookup:
            lookup[key] = {}

        if word not in lookup[key]:
            lookup[key][word] = 0

        lookup[key][word] += 1

    return lookup


def predict_next_word( # I DID ORIGINALLY WROTE THIS FUNCTION, BUT THEN ASKED AI TO HELP ME IMPROVE IT.
    context_words,
    model,
    word_to_idx,
    idx_to_word,
    rng,
    temperature=TEMPERATURE,
    context_lookup=None,
    alpha=alpha,
    step=0,  # optional for future EOS scheduling
):
    vocab_size = len(word_to_idx)
    context_indices = [word_to_idx[word] for word in context_words]

    # =========================
    # 1. Neural probabilities
    # =========================
    neural_probs = model.predict_distribution(
        context_indices,
        temperature=temperature
    )

    neural_probs[word_to_idx[BOS_TOKEN]] = 0.0

    neural_sum = neural_probs.sum()
    if neural_sum > 0:
        neural_probs = neural_probs / neural_sum


    # =========================
    # 2. Lookup probabilities
    # =========================
    lookup_probs = np.zeros(vocab_size, dtype=float)

    if context_lookup is not None:
        matches = context_lookup.get(tuple(context_indices))

        if matches:
            words = list(matches.keys())
            counts = np.array(list(matches.values()), dtype=float)

            # temperature-aware scaling
            counts = counts ** (1.0 / max(temperature, 1e-6))

            total = counts.sum()
            if total > 0:
                probs = counts / total

                for word, prob in zip(words, probs):
                    lookup_probs[word_to_idx[word]] = prob


    # =========================
    # 3. Blend distributions
    # =========================
    if lookup_probs.sum() > 0:
        final_probs = alpha * lookup_probs + (1 - alpha) * neural_probs
    else:
        final_probs = neural_probs


    # =========================
    # 4. Clean invalid tokens
    # =========================
    eos_idx = word_to_idx[EOS_TOKEN]
    bos_idx = word_to_idx[BOS_TOKEN]

    final_probs[bos_idx] = 0.0

    # optional: mild EOS control (prevents instant stopping)
    # remove if you want fully natural EOS behavior
    final_probs[eos_idx] *= 0.3 + 0.7 / (1 + step * 0.5)


    # =========================
    # 5. Normalize safely
    # =========================
    total = final_probs.sum()

    if total <= 0:
        return idx_to_word[eos_idx]

    final_probs = final_probs / total


    # =========================
    # 6. Sample
    # =========================
    next_word_idx = rng.choice(vocab_size, p=final_probs)

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

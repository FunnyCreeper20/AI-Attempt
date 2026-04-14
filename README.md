# AI-Attempt
This is my first AI creation. It is a simple bigram using deterministic (not stochastic / probabilistic) memory, but can be configured depending on the response code.

How it works (I think, I forgot since I made it a little while ago):

This is how a basic AI model works, but I changed it a bit and I am too tired to change the step-by-step.

1. What it does is it uses corpus.txt to get all of its vocabulary.
2. It then turns that data into vectors or matricies
3. It multiplies those matricies and accounts for bias in nn_layers.py
4. Then, it predicts the next word that would come after the user input
5. Finally, it responds with a logical answer


NOTE: This is not finished, and the variables at the top control different aspects of the Neural Network

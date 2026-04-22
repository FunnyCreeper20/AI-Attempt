# AI-Attempt
This is my first AI creation. It is a simple transformer using deterministic (not stochastic / probabilistic) memory, but can be configured depending on the response code.

**!!!WARNINGS!!!***
* You will need to run the command: pip install torch OR pip3 install torch for main.py
* You will need to run the command: pip install numpy or pip3 install numpy for numPyMain.py

numPyMain.py is just main.py but turned into numPy for those who either want:
1. A more "core" understanding of it
2. A version that runs on a basic module/import
3. A version that is pretty slow because it doesn't use an ADAM backwards pass

**To-Do**:
* Add GeLU to the hidden layers
* Add BOS-Tokens
* Save the training data on a server
* Add an automatic pip (like Rohan's)
* Add a dense layer

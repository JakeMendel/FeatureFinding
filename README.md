# FeatureFinding

## Overview

`FeatureFinding` is a repository dedicated to exploring the structure of feature representations in various toy models. This work is inspired by the toy models of superposition architecture as presented in [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html).

## Contents

- **Notes.md**: A collection of technical experiment ideas, many of which are yet to be implemented.
- **FeatureFinding**: The main module or package of the repository. Contains the following main components:
  - **datasets.py**: Responsible for generating datasets. Options include normalized or unnormalized inputs and one-hot or k-hot inputs.
  - **models.py**: Contains code for the primary neural network models:
    - **Net**: A neural network with one hidden layer, offering various customization options.
    - **ResNet**: Inherits from `Net` and provides customizable extra hidden layers with a residual stream.
  - **utils.py**: Houses a suite of utility functions beneficial for:
    - Training the neural networks.
    - Data processing.
    - Data visualization in the notebooks.
  - **notebooks**: A collection of Jupyter notebooks each focusing on a different model architecture:
    - **1d-softmax.ipynb**: Investigates a neural network with a 1D hidden layer, ending with softmax + cross-entropy.
    - **2d-bias.ipynb**: Explores a neural network with a 2D hidden layer with a bias, ending with ReLU-MSE.
    - **2d-untied-nobias.ipynb**: Features a 2D hidden layer without a bias. The embedding and unembedding processes are untied.
    - **bias-hysteresis.ipynb**: Studies hysteresis during the learning phase of a model with a bias undergoing a phase transition.
    - **computation.ipynb**: Introduces a modified training setup where boolean operations must be applied to the inputs to produce outputs.
    - **hessian_analysis.ipynb**: Investigates the same model as in bias-hysteresis and 2d-bias but from the perspective of the Hessian's eigenvalues over training.
    - **non-full-batch.ipynb**: The only notebook that doesn't utilize full-batch gradient descent. It investigates if stochasticity can lead to a model with a lower loss using the same model architecture as in hessian_analysis and bias-hysteresis.
    - **symmetric-nd.ipynb**: Investigates an n-dimensional hidden layer on a tied embed/unembed, bias-free neural network with a ReLU at the end. The primary aim is to count the number of feature directions encoded in the hidden layer.

## Setup & Installation

1. Clone the repository.
2. Navigate to the repository's root directory.
3. Install the required dependencies using `pip install -r requirements.txt`.

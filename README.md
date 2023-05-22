# First Experiment

## Theory
Suppose we train a model, and at layer $l$ there are a large (potentially near infinite) number of features which the model would benefit from learning if it could. Let $\Sigma$ denote the set of hyperparameters of the training process: the learning rate etc, but also the dataset, number of epochs etc, and the details of the model architecture except for the number of activations in layer $l$. If $n$ is the number of activations at layer $l$, then define $\phi_\Sigma(n)$ to be the (average) number of unique features learned in layer $l$ when the number of activations in layer $l$ is $n$ - when $\phi(n)> n$ the features must be stored in superposition. Also define $L_\Sigma(n)$ as the (average) loss achieved by the model architecture with hyperparameters $\Sigma$ and number of activations $n$ in layer $l$. Henceforth we will drop the subscript $\Sigma$ but it's worth remembering that the number of features learned in layer $Ll will in general depend on other properties of the training process and model architecture.

We'd like to study the form of $\phi(n)$ and $L(n)$. Some properties they should have:
- $\phi(n)$ should be monotonically increasing (for suitable $\Sigma$(?)). Increasing the dimension of the activation space allows more features to be packed in without significant penalties from interference. This should also cause $L(n)$ to be monotonically decreasing.
- The rate of increase of the number of features should be decreasing, because each new feature is less useful than all the previous ones, and the pressure to add each new feature is therefore decreasing.
- In fact, $\phi(n)$ should be bounded above by the maximum number of features worth learning at all. This number has to be finite: worst case $\phi_\text{max}\sim 2^{d_\text{input}}$, but the real bound may be much lower than this. When $n > \phi(max)$ features are out of superposition.
- If we can't bound $\phi_\text{max}$ below $2^{d_\text{input}}$, a more useful result may be (hypothesis):
  
  Denote the set of features learned at a low activation number $n$ by $F_n$. 
  
  Define $\Phi_n: F_n \rightarrow \mathbb{R}^n$ to be a map from features to the vector representation the model finds for these features in the $n$ dimensional activation space. Since the model learns important features first, $F_n \in F_N \: \forall n \lt N$
  
  Then we may have that $\forall\epsilon>0, \forall f,g \in F_n, \exists N > n$ s.t. $\Phi_N(f)\cdot\Phi_N(g) < \epsilon$
  This is a useful result if, for any reasonable choice of $n$ and $\epsilon$, the minimum value of $N\ll 2^{d_\text{input}}$.

## Experiment: Universality
Train a model with $n$ neurons using SoLU in layer $l$ to enforce that only $n$ features are learned. Then train an identical model except with $N>n$ SoLU neurons in layer $l$. Are the $n$ neurons which fire most often in the second model encoding the same features as the $n$ features in the first model?

## Experiment: Mixing up the ReLUs for more computation
Train a model with $n$ neurons using SoLU in layer $l$ to enforce that only $n$ features are learned. Then train a model with $N\gg n$ ReLU neurons in layer $l$. Using SVD (or something similar) ablate the $N-n$ least important directions in the second model's activation space. How does model performance compare across models? Are the features the same features?

## Experiment: Counting features
Train pairs of (model with ReLU at layer $l$, model with SoLU at layer $l$) for a range of values of $n$. If SoLU forces there to be $n$ features only, in the neuron basis, then the models should only have similar performance when $n>phi(n)$. If there is some way to enforce that the number of features is the same across different values of $n$ then we could count features like this.

# Second Experiment
There are a range of related experiments we could do to test if the feature directions found by sparse autoencoders are the 'true features'. This section to be continued.

## Theory
What is the chance that this result could have happened by chance? ie: If there are $n$ dimensions and $\phi(n)$ features, and we insist that no more than $k$ of these features are involved in any single activation vector's decomposition, how unique is this? we have already worked this out I think [tbc]

## Experiment: Robustness
If we train the sparse autoencoder twice, do we get the same decomposition of activation vectors?

## Experiment: Avoids irrelevant directions
If we fix some weights on the sparse autencoder to hardcode some features into the model, but we don't learn their biases, does the model set their bias really high so the neurons never activate? 

## Experiment: Compare layer sizes
1. If we fix the model weights between all layers except $W_{L-1,L}$ and $W_{L,L+1}$, and then retrain just those weights, there isn't that much flexibility in what features the model could learn at layer $L$ to slot into the circuits in the rest of the model. Does the sparse autoencoder find the same features and the same feature directions?
2. If we now change the number of neurons in layer $L$, do we find the same features still?
3. If this works, can we increase the number of layers we retrain and still find the same features? Can we do the whole model?
Joe is curious about this one.

# Some other thoughts formed during writing
If A is a human interpretable feature and B is too, but A and B are not independent, will PCA not find a principal component that points along a linear combination of A and B? Is this what we want?
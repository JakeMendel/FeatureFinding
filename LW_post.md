# Superposition and bottlenecks

Where does this happen in the nn:
- From (1) embedding or (A) MLP_out to (2) unembedding or (B) MLP_in
- In ($1\rightarrow 2$) the best setup is something like $n\rightarrow f \rightarrow n \rightarrow$ Crossentropy

    - Since the unembed normally has a bias but the embed doesn't, we have $$L=\sum_i H(\mathbf{W}_U \mathbf{W}_E \mathbf{x}_i+b,\mathbf{x}_i)$$
    
    - If the embed and unembed are tied, $\mathbf{W}_U = \mathbf{W}_E^\intercal$
    - We're ignoring Layernorm

- In ($A\rightarrow B$) the best setup is something like $n\rightarrow f \rightarrow n \rightarrow$ ReLU $\rightarrow$ MSE

    Since both sides of the MLP usually have a bias, we have $$L= \sum_i ||\text{ReLU}(\mathbf{W}_U (\mathbf{W}_E \mathbf{x}_i+b_1)+b_2)-\mathbf{x}_i||^2$$ Really, the number of neurons in each MLP is $4f$, but we assume that they encode $n>4f$ features in superposition. In that case, the structure should be $n\rightarrow 4f \rightarrow f \rightarrow 4f \rightarrow n \rightarrow$ ReLU $\rightarrow$ MSE. We can ignore this, because we can combine the matrices $\mathbf{W}_{n\rightarrow4f} \mathbf{W}_{4f\rightarrow f}$ into a single rank 2 matrix $\mathbf{W}_{n\rightarrow f}$

- There's also something about mixing MLPs and embed/unembed, but there's some confusion here because the layers have different sizes.
    - Maybe a suitable model of ($1\rightarrow B$) is like $n\rightarrow f \rightarrow 4f \rightarrow \text{Relu} \rightarrow n$ with MSE. Where $4f$ is the MLP, and we check if the MLP has learned the real features by insisting that it's just a linear map away from reconstruction.
    - Similarly, maybe a suitable model of ($A\rightarrow 2$) is like $4f\rightarrow f \rightarrow \text{Relu} \rightarrow n$ with CrossEntropy.

- Replication of Marius' setup with a relu in the bottleneck layer. I argue that the original setup is a better description of what we want

- For this setup, I want to analyse 1-sparse features only (this is a simplification but may not be a terrible one since features are likely to be v sparse, and Toy Models p14 points out that features go for sparse interference graphs so that there is limited high-order interference). I will also choose my features to be binary: this will make the setup cleaner to analyse, and it may reflect reality better: in real life, features are likely to either be present or not be present in an input. This means that each input is a 1-hot vector, and my entire dataset is just the identity matrix.

## $A\rightarrow B$
This is the exact setup introduced in [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (TMS), with the simplification that they don't use perfect sparsity or binary features. I'll use a ReLU at the end of the model, and MSE.

### Without bias
- In 2d, models learn a pentagon, as demonstrated in TMS. 
- With GELU, models learn a pentagon that has closer to unit norm and only one feature per direction. Features are also learned faster
  - *Derivation of no gradient along the feature direction in relu case.*
  - *Is there anything to say here about why gradients are better in the GELU case?*
- Why 5 feature directions (maybe a footnote?):
  - As laid out in TMS, in the 1-sparse case: $$L=\sum_i\left(1-||\mathbf{W}_i||^2\right)^2 + \sum_{j\neq i}\text{ReLU}(\mathbf{W}_i\cdot \mathbf{W}_j)^2$$
  - Let's assume $||\mathbf{W}_i|| = 1$ and features are uniformly distributed on a circle. Then if we learn $d$ directions, the loss will be $$L=\sum_{r=-\lfloor d/4\rfloor}^{\lfloor d/4\rfloor}\cos^2\left(\frac{2\pi r}{d}\right)$$ This function is minimised (for integer $d$) at $d=5$.
- Why no pressure to learn 



## 2-sparse features? Is it necessary?
A comment about Johnson-Lindenstrauss lemma. Mean interference < epsilon vs max interference < epsilon. Arranging features uniformly on the sphere gets on average interference $\frac{1}{\sqrt{d}}$ compared to $\frac{1}{d}$ if we go for a long tegum product of low dimensional polytopes. But this method only fits a number of features that is linear in the dimension (assuming a fixed size of polytope).

 
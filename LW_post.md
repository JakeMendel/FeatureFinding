# Superposition and bottlenecks


Where does this happen in the NN:
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

### Tied Without bias
In 2d, models learn a pentagon, as demonstrated in TMS. 
- With GELU, models learn a pentagon that has closer to unit norm and only one feature per direction. Features are also learned faster
  - *Derivation of no gradient along the feature direction in relu case.*
  - *Is there anything to say here about why gradients are better in the GELU case?*
- Why 5 feature directions (maybe a footnote?):
  - As laid out in TMS, in the 1-sparse case: $$L=\sum_i\left(1-||\mathbf{W}_i||^2\right)^2 + \sum_{j\neq i}\text{ReLU}(\mathbf{W}_i\cdot \mathbf{W}_j)^2$$
  - Let's assume $||\mathbf{W}_i|| = 1$ and features are uniformly distributed on a circle. Then if we learn $d$ directions, the loss will be $$L=\sum_{r=-\lfloor d/4\rfloor}^{\lfloor d/4\rfloor}\cos^2\left(\frac{2\pi r}{d}\right)$$ This function is minimised (for integer $d$) at $d=5$.

In 3d, ReLU always learns the 9 sided shape. Images of the polyhedron. GELU on the other hand learns a variety of shapes (square antiprism and pentagonal dipyramid).
In nd, can we plot shape against number of features learned?

### Untied without Bias
- why do we expect the model to learn the same unembed as embed?
    - Bigram statistics
- Why can we do better when untied? 
    - My algorithm with 5 features
        - Diagram
    - Model's algo in 2d with ReLU and GELU
    - Qualitatively, push features to the edges of opposite quadrants
        - Model's algo in 3d
- Relevance for real models:
    - Early MLP will write into superposition in a different place to where a later MLP will write out of superposition. 
    - If we do something like SVD/Sparse autoencoding to find features, and try to compare across layers, we have to be careful. Just finding max cosine similarity or something to pair features up isn't the full story. eg. in the 5 features in 2d case, written feature 1 is most similar in direction to reading direction 2.
- Analogy to the POVM: to separate $v_1$ from $v_2$ with $v_1 \cdot v_2 > 0$, should use unembed vectors $u_1,u_2$ with $u_1\perp v_2$ and $u_2\perp v_1$. More generally, the best direction to read from depends on where the interfering features are. How deep does the analogy go?
    - POVMs are about discriminating between non orthogonal states ~feature directions. 
    - Two studied applications. Unclear which is more suitable:
        - In Minimum error measurement, we want to maximise the expected information we get from a measurement.
        - In minimum uncertainty unambiguous state discrimination, we want to maximise the probability that we identify the feature with certainty.
        -  Construct a set of POVMs by adding a set of non-orthogonal projection operators with some coefficients. Probability of finding each allowed state after measurement is given by the modulus squared of coefficient $\times$ the amplitude of the original state in each projection direction. ~ MSE maybe?
        - POVMs can be thought of as low dimension projections of projective (orthogonal) measurements in a higher dimensional hilbert space ~ Weight superposition is a small network simulating a larger orthogonal one?
        - Similarly, mixed states can be thought of as low dimension projections of pure states in a higher hilbert states ~ Feature superposition is a small activation space simulating a larger one with orthogonal features?
        - Disanalogies:
            - Hilbert spaces are constrained to give their vectors unit norm.
            - Vectors in a Hilbert space have complex coeffs. In ML, they have real coeffs. (Although would be cool to train a NN with complex params)
            - The POVMs must sum to the identity. Unclear what this would mean in ML
            - The POVMs must be positive operators. Unclear what this means in ML, although my gut thinks this is actually a good part of the analogy.

### Tied with End Bias
- The model should be able to reconstruct all the features without loss when a bias is added. Simply put them on a circle and subtract the same bias from each, which is enough that only the feature pointing in the same direction as the projection survives bias + relu. 
    - Diagram
- Empirically I am unable to get this to happen! Instead:
    - A small number of features are learned. Often 6, maybe extremely briefly going through 5.
    - When the initial learning rate is high, the model learns this exact algorithm up to exactly the theoretical weight magnitudes (eg. f=10, seed200, max_lr = 2 initial_lr = 0.005 warmup_frac = 0.05 final_lr = 0.1) but for only 5 features. The features learned are always ones with bias initialised above zero.
    - *proof* Can I explain why this doesn't happen? Show that once 6 features are learned, nothing else is?
    - When initial learning rate is low, the model learns this messy complicated thing (high level details are seed independent):
        - First learn a polygon (normally hexagon). Then, polygon starts to distort, shifting away from one of the points and giving it lots of space. Then, more points start to appear on the surface, and each one that appears causes a drop in loss. But, then the model seems to decide it has learned too many, and it starts to send some inside the polygon.
        - Also, the way the points sent to zero behave depends on whether they were initialised with positive or negative bias. 

Can I encourage the correct algorithm?
- Let's restrict the space of the NN, to encourage normal weights. Enforce that each weight vector has unit norm. Now I need to learn a scale factor. How does the initialisation of the scale factor affect the number of features learned?
    - When it's initialised to one, the features get learned uniformly on a circle... but less than one direction per feature. Why? 
- What happens if I initialise the weights and biases of the NN with pretty much correct values for the function implementation? How close do the values have to be to learn this implementation?

### What about bias in the hidden layer

## With Softmax

- Seed 142, untied
    max_lr = 2
    initial_lr = 0.005
    warmup_frac = 0.05
    final_lr = 0.1
    epochs = 30000
    f = 40, n = 2
    nonlinearity ReLU


## 2-sparse features? Is it necessary?
A comment about Johnson-Lindenstrauss lemma. Mean interference < epsilon vs max interference < epsilon. Arranging features uniformly on the sphere gets on average interference $\frac{1}{\sqrt{d}}$ compared to $\frac{1}{d}$ if we go for a long tegum product of low dimensional polytopes. But this method only fits a number of features that is linear in the dimension (assuming a fixed size of polytope).
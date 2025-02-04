# LeNet-5

## Historical Context
LeNet-5, designed by Yann LeCun et al. in 1998, is one of the earliest convolutional neural networks. Initially developed for handwritten digit recognition (MNIST), it laid the foundation for modern deep learning-based vision systems.

## Architectural Overview and Mathematical Analysis

### Convolutional Layers
- **First Convolution (C1):**  
  Input images of size $32\times32$ (after resizing) pass through a convolutional layer with a kernel of size $5\times5$, no padding ($p=0$) and stride $s=1$.  
  The output dimension is computed as:  
  $$n_{out} = n_{in} - k + 1 = 32 - 5 + 1 = 28$$  
  Producing an output of shape $(6, 28, 28)$ since there are 6 filters.

- **First Pooling (S2):**  
  An average pooling with kernel size $2\times2$ and stride $2$ reduces spatial dimensions by a factor of 2:  
  $$28\to \frac{28}{2} = 14$$  
  Thus, the feature map becomes $(6, 14, 14)$.

- **Second Convolution (C3):**  
  The second convolution layer uses a $5\times5$ kernel. The output size is:  
  $$14 - 5 + 1 = 10$$  
  With 16 filters, yielding $(16, 10, 10)$.

- **Second Pooling (S4):**  
  Similarly applying average pooling reduces dimensions:  
  $$10\to \frac{10}{2} = 5$$  
  Resulting in $(16, 5, 5)$.

### Fully Connected Layers
- **Flattening:**  
  The feature maps are flattened into a vector of size $16 \times 5 \times 5 = 400$.
  
- **First Fully-Connected (C5) Layer:**  
  Transforms the 400-dimensional vector into 120 features using a linear transformation. Mathematically, this layer computes:  
  $$\mathbf{y} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$  
  where $\mathbf{W}_1 \in \mathbb{R}^{120\times400}$.

- **Second Fully-Connected (F6) Layer:**  
  Further processes the 120 features into 84 features:  
  $$\mathbf{z} = \mathbf{W}_2 \mathbf{y} + \mathbf{b}_2$$  
  with $\mathbf{W}_2 \in \mathbb{R}^{84\times120}$.

- **Output Layer:**  
  Finally, an output layer maps the 84 features to 10 classes:  
  $$\mathbf{o} = \mathbf{W}_3 \mathbf{z} + \mathbf{b}_3$$  
  where $\mathbf{W}_3 \in \mathbb{R}^{10\times84}$.

### Activation Functions
- **ReLU Activation:**  
  After each convolution and fully connected layer (except the final output layer), the ReLU function is applied:  
  $$\text{ReLU}(x) = \max(0, x)$$  
  ensuring non-linearity.

### Weight Initialization
- **Convolution Layers:**  
  Weights are initialized using Kaiming Normal initialization suited for ReLU activations, i.e.,  
  $$\text{Var}(w) \propto \frac{2}{\text{fan\_out}}$$
- **Linear Layers:**  
  Weights use a normal distribution with mean 0 and standard deviation 0.01.

## Advantages
- **Simplicity:** The architecture, while historically significant, is straightforward and ideal for educational purposes.
- **Lightweight:** Contains a low number of parameters, making it trainable on modest hardware.
- **Clear Mathematical Foundation:** Each layer's operation can be described mathematically, making it a good example to understand CNN fundamentals.

## Disadvantages
- **Limited Capacity:** The model may not capture highly complex visual patterns, limiting its performance on advanced tasks.
- **Lack of Modern Techniques:** Does not incorporate methods like dropout, batch normalization, or deeper layer stacking that are common in more recent CNN architectures.

## Other Considerations
LeNet-5’s structure demonstrates key CNN operations such as convolution, pooling, and fully connected layers, and the inline LaTeX helps precisely convey the mathematical computations involved. Understanding these formulas is crucial:
- Convolution operation: $$o(i,j) = \sum_{m,n} x(i+m, j+n) \cdot k(m,n)$$
- Pooling reduction: $$\text{pool}(x) = \frac{1}{|R|} \sum_{(i,j) \in R} x(i,j)$$

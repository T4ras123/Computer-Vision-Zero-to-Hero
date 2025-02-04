# AlexNet

## Historical Context
AlexNet, introduced by Krizhevsky et al. in 2012, sparked the deep learning revolution by winning the ImageNet competition with a wide margin. Its success demonstrated the power of deep convolutional neural networks for large-scale image recognition tasks.

## Architectural Overview and Mathematical Analysis

AlexNet is composed of a feature extractor and a classifier:

### Feature Extractor (Convolutional Layers)
Given an input image (typically of size $227 \times 227 \times 3$), the network applies a series of convolution, activation, and pooling operations:

1. **First Convolutional Layer:**  
   - **Operation:**  
     $$ n_{out} = \left\lfloor \frac{n_{in} - k + 2p}{s} \right\rfloor + 1 $$
   - **Parameters:**  
     Kernel size: $11$, Stride: $4$, Padding: $2$  
   - **Calculation:**  
     $$ n_{out} = \left\lfloor \frac{227 - 11 + 2 \times 2}{4} \right\rfloor + 1 = \left\lfloor \frac{227 - 11 + 4}{4} \right\rfloor + 1 = \left\lfloor \frac{220}{4} \right\rfloor + 1 = 55 + 1 = 56 $$
   - **Output:** Feature map of dimensions $(64, 56, 56)$.

2. **First Max Pooling:**  
   - **Parameters:**  
     Kernel size: $3$, Stride: $2$
   - **Calculation:**  
     $$ n_{pool} = \left\lfloor \frac{56 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{53}{2} \right\rfloor + 1 = 26 + 1 = 27 $$
   - **Output:** Feature map of dimensions $(64, 27, 27)$.

3. **Second Convolutional Layer:**  
   - **Parameters:**  
     Kernel size: $5$, Padding: $2$ (with stride $1$, implicitly)  
   - **Calculation:**  
     Using the same formula with $k=5$, $p=2$, $s=1$:
     $$ n_{out} = \left\lfloor (27 - 5 + 2 \times 2) \right\rfloor + 1 = 27 $$
   - **Output:** Feature map of dimensions $(192, 27, 27)$.

4. **Second Max Pooling:**  
   - **Parameters:**  
     Kernel size: $3$, Stride: $2$
   - **Calculation:**  
     $$ n_{pool} = \left\lfloor \frac{27 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{24}{2} \right\rfloor + 1 = 12 + 1 = 13 $$
   - **Output:** Feature map of dimensions $(192, 13, 13)$.

5. **Third Convolutional Layer:**  
   - **Parameters:**  
     Kernel size: $3$, Padding: $1$
   - **Calculation:**  
     With $k=3$, $p=1$, $s=1$:
     $$ n_{out} = \left\lfloor (13 - 3 + 2 \times 1) \right\rfloor + 1 = 13 $$
   - **Output:** Feature map of dimensions $(384, 13, 13)$.

6. **Fourth Convolutional Layer:**  
   - **Parameters:**  
     Kernel size: $3$, Padding: $1$
   - **Output:** Changes channels: $(384 \to 256)$ while spatial dimensions stay $(13, 13)$.

7. **Fifth Convolutional Layer:**  
   - **Parameters:**  
     Kernel size: $3$, Padding: $1$
   - **Output:** Maintains spatial dimensions $(13, 13)$ and produces $(256, 13, 13)$.

8. **Third Max Pooling:**  
   - **Parameters:**  
     Kernel size: $3$, Stride: $2$
   - **Calculation:**  
     $$ n_{pool} = \left\lfloor \frac{13 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{10}{2} \right\rfloor + 1 = 5 + 1 = 6 $$
   - **Output:** Final feature map of dimensions $(256, 6, 6)$.

### Classifier (Fully Connected Layers)
After the feature extraction, the output is flattened and passed to the classifier:

1. **Flattening:**  
   - **Calculation:**  
     $$ \text{Flattened size } = 256 \times 6 \times 6 = 9216 $$
   
2. **Fully Connected Layers:**  
   - **First FC Layer:** Projects the $9216$-dimensional vector to $4096$ neurons:
     $$ \mathbf{y}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{W}_1 \in \mathbb{R}^{4096 \times 9216} $$
   - **Dropout & ReLU Activation:** Applied after the layer for regularization and non-linearity.
   - **Second FC Layer:** Further maps $4096$ to $4096$:
     $$ \mathbf{y}_2 = \mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{4096 \times 4096} $$
   - **Final FC Layer:** Maps $4096$ features to the number of classes:
     $$ \mathbf{o} = \mathbf{W}_3 \mathbf{y}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{\text{num\_classes} \times 4096} $$

3. **Softmax Activation:**  
   After the classifier, a softmax function computes the class probabilities:
   $$ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

## Advantages
- **Breakthrough Performance:** Significant advancements on ImageNet demonstrated the viability of deep CNNs.
- **Effective Regularization:** Dropout and data augmentation help mitigate overfitting.
- **Clear Architectural Design:** The varying kernel sizes capture multi-scale features, and the inline mathematical expressions clarify the operations.

## Disadvantages
- **High Computational Cost:** The model’s depth and large fully connected layers require substantial computational resources.
- **Large Number of Parameters:** This increases the risk of overfitting on smaller datasets without proper regularization.
- **Inflexible Structure:** The rigid architecture provides limited adaptability compared to more recent modular designs.

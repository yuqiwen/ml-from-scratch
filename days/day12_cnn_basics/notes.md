# Day 12: CNN Basics

## 1. Goal

Today's goal is to understand the basics of Convolutional Neural Networks.

Key concepts:

```text
image tensor
channel
kernel / filter
convolution
feature map
stride
padding
pooling
Conv2d
MaxPool2d
Flatten
```

## 2. Why CNN?

MLP usually flattens an image:

```text
28 x 28 -> 784
```

This loses spatial structure.

CNN keeps spatial structure and learns local patterns using convolution kernels.

## 3. Image Tensor Shape

PyTorch image tensors usually use:

```text
NCHW
```

Meaning:

```text
N = batch size
C = channels
H = height
W = width
```

Example:

```text
X shape = (32, 3, 64, 64)
```

This means:

```text
32 images
3 channels
64 height
64 width
```

## 4. Kernel / Filter

A kernel is a small learnable matrix.

For example:

```text
3 x 3 kernel
```

It slides over the image and extracts local features.

CNN filters can learn to detect:

```text
edges
corners
textures
local patterns
```

## 5. Conv2d

PyTorch example:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
)
```

Meaning:

```text
input channels = 1
output channels = 8
kernel size = 3 x 3
stride = 1
padding = 1
```

`out_channels` means the number of filters.

## 6. Feature Map

The output of a convolution layer is called a feature map.

If input is:

```text
(B, 1, 28, 28)
```

and conv is:

```text
Conv2d(1, 8, kernel_size=3, padding=1)
```

then output is:

```text
(B, 8, 28, 28)
```

This means the layer learns 8 feature maps.

## 7. Stride

Stride controls how far the kernel moves each step.

```text
stride = 1
```

moves one pixel at a time.

```text
stride = 2
```

moves two pixels at a time and reduces spatial size.

## 8. Padding

Padding adds zeros around the input image.

Without padding:

```text
input: 28 x 28
kernel: 3 x 3
stride: 1
padding: 0

output: 26 x 26
```

With padding 1:

```text
input: 28 x 28
kernel: 3 x 3
stride: 1
padding: 1

output: 28 x 28
```

## 9. Output Size Formula

For one spatial dimension:

```text
output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1
```

Example:

```text
input_size = 28
padding = 1
kernel_size = 3
stride = 1
```

Then:

```text
output_size = floor((28 + 2*1 - 3) / 1) + 1
            = 28
```

## 10. Max Pooling

Max pooling reduces spatial size.

Example:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

If input is:

```text
(B, 8, 28, 28)
```

output is:

```text
(B, 8, 14, 14)
```

Pooling keeps channels the same but reduces height and width.

## 11. Simple CNN Structure

A simple CNN can look like:

```text
input
-> Conv2d
-> ReLU
-> MaxPool2d
-> Conv2d
-> ReLU
-> MaxPool2d
-> Flatten
-> Linear
-> output
```

## 12. ML Systems Connection

CNNs are important for ML systems because convolution layers are compute-heavy and memory-layout sensitive.

Important systems topics later include:

```text
NCHW vs NHWC layout
cuDNN convolution kernels
im2col
operator fusion
activation memory
mixed precision
inference benchmark
```

## 13. Checklist

- [ ] Understand image tensor shape NCHW
- [ ] Understand channel
- [ ] Understand kernel / filter
- [ ] Understand Conv2d
- [ ] Understand feature map
- [ ] Understand stride
- [ ] Understand padding
- [ ] Understand pooling
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass

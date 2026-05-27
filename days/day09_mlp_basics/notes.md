# Day 09: MLP / Neural Network Basics

## 1. Goal

Today's goal is to understand a basic neural network called MLP.

MLP stands for:

```text
Multi-Layer Perceptron
```

Key concepts:

```text
Linear layer
Hidden layer
Activation function
ReLU
Logits
BCEWithLogitsLoss
Binary classification
```

## 2. From Linear Model to MLP

A linear model is:

```text
y_hat = X @ W + b
```

In PyTorch:

```python
nn.Linear(input_dim, output_dim)
```

A simple MLP is:

```text
input -> Linear -> ReLU -> Linear -> output
```

## 3. Hidden Layer

A hidden layer creates intermediate representations.

```text
h = X @ W1 + b1
```

Then the output layer uses this hidden representation:

```text
output = h @ W2 + b2
```

The hidden layer helps the model learn more complex patterns.

## 4. Why Activation Functions Are Needed

If we stack linear layers without activation:

```text
Linear -> Linear -> Linear
```

the whole model is still equivalent to one linear transformation.

To learn nonlinear patterns, we need activation functions.

## 5. ReLU

ReLU means:

```text
Rectified Linear Unit
```

Formula:

```text
ReLU(x) = max(0, x)
```

Example:

```text
ReLU([-2, -1, 0, 3, 5]) = [0, 0, 0, 3, 5]
```

## 6. MLP for Binary Classification

For binary classification, the model can output one raw score called a logit.

```text
logit = model(x)
probability = sigmoid(logit)
```

If:

```text
probability >= 0.5
```

predict class 1.

Otherwise, predict class 0.

## 7. BCEWithLogitsLoss

In PyTorch, we often use:

```python
nn.BCEWithLogitsLoss()
```

This combines:

```text
sigmoid
binary cross entropy
```

into one numerically stable loss function.

So during training:

```python
logits = model(x_batch)
loss = loss_fn(logits, y_batch)
```

During prediction:

```python
probabilities = torch.sigmoid(logits)
predictions = probabilities >= 0.5
```

## 8. Training Loop

The training loop is the same as Day 08:

```python
for x_batch, y_batch in train_loader:
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The only difference is that the model is now an MLP instead of a single linear layer.

## 9. ML Systems Connection

MLP introduces the standard deep learning block pattern:

```text
Linear
Activation
Linear
Loss
Optimizer
```

This pattern later scales into:

```text
CNN
Transformer
LLM
```

Modern AI infrastructure still trains models using the same loop structure:

```text
forward
loss
backward
optimizer step
validation
checkpoint
```

## 10. Checklist

- [ ] Understand MLP
- [ ] Understand hidden layer
- [ ] Understand why activation is needed
- [ ] Understand ReLU
- [ ] Understand logits
- [ ] Understand BCEWithLogitsLoss
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass

# Day 03: Batch Training, SGD, Epoch, and Step

## 1. Goal

Today's goal is to understand how model training works when the dataset is split into batches.

This is the foundation of modern deep learning training loops and ML systems.

Key concepts:

```text
dataset size
batch size
epoch
step / iteration
full-batch gradient descent
stochastic gradient descent
mini-batch gradient descent
shuffle
```

## 2. Dataset Size

`dataset_size` means the total number of training samples.

Example:

```text
dataset_size = 1000
```

This means the training dataset contains 1000 samples.

## 3. Batch Size

`batch_size` means how many samples are used for one parameter update.

Example:

```text
batch_size = 100
```

This means each training step uses 100 samples.

For each batch, the model does:

```text
forward pass
compute loss
compute gradients
update parameters
```

So:

```text
1 batch -> 1 training step -> 1 parameter update
```

## 4. Epoch

One epoch means the model has seen the entire training dataset once.

Example:

```text
dataset_size = 1000
batch_size = 100
```

Then one epoch contains:

```text
steps_per_epoch = dataset_size / batch_size
                = 1000 / 100
                = 10
```

So:

```text
1 epoch = 10 steps
```

If:

```text
epochs = 5
```

Then:

```text
total_steps = epochs * steps_per_epoch
            = 5 * 10
            = 50
```

The model updates its parameters 50 times.

## 5. What If Dataset Size Is Not Divisible by Batch Size?

Example:

```text
dataset_size = 1030
batch_size = 100
```

Then:

```text
10 full batches with 100 samples
1 last batch with 30 samples
```

So:

```text
steps_per_epoch = ceil(dataset_size / batch_size)
                = 11
```

Some training loops allow `drop_last=True`.

If `drop_last=True`, the last incomplete batch is discarded:

```text
steps_per_epoch = floor(dataset_size / batch_size)
                = 10
```

## 6. Step / Iteration

A step, also called an iteration, means one parameter update.

For one step:

```text
take one batch
compute predictions
compute loss
compute gradients
update parameters
```

So:

```text
step = one optimizer update
```

In PyTorch, one step usually looks like:

```python
y_hat = model(x_batch)
loss = loss_fn(y_hat, y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 7. Full-Batch Gradient Descent

Full-batch gradient descent uses the entire dataset for one update.

```text
batch_size = dataset_size
```

Example:

```text
dataset_size = 1000
batch_size = 1000
```

Then:

```text
steps_per_epoch = 1
```

Advantages:

- Stable gradient
- Uses all training data for each update

Disadvantages:

- Slow for large datasets
- High memory usage
- Not practical for modern large-scale deep learning

## 8. Stochastic Gradient Descent

Stochastic Gradient Descent, or SGD, uses one sample for one update.

```text
batch_size = 1
```

Example:

```text
dataset_size = 1000
batch_size = 1
```

Then:

```text
steps_per_epoch = 1000
```

Advantages:

- Very frequent updates
- Low memory usage

Disadvantages:

- Very noisy gradients
- Less efficient on GPUs
- Training can be unstable

## 9. Mini-Batch Gradient Descent

Mini-batch gradient descent uses a small batch of samples for one update.

Example:

```text
dataset_size = 1000
batch_size = 32
```

Then:

```text
steps_per_epoch = ceil(1000 / 32) = 32
```

This is the most common approach in modern deep learning.

Advantages:

- More efficient on GPUs
- More stable than single-sample SGD
- Lower memory usage than full-batch gradient descent
- Good balance between throughput and optimization stability

## 10. Why Shuffle?

At the beginning of each epoch, we usually shuffle the dataset.

Without shuffling, data order may hurt training.

Example:

```text
first 500 samples: class 0
last 500 samples: class 1
```

If we do not shuffle, early batches may contain only class 0 and later batches may contain only class 1.

This can make training unstable.

Shuffling makes each batch more representative of the full dataset.

## 11. ML Systems Connection

Batch size is not only a machine learning concept. It is also a systems concept.

Batch size affects:

```text
GPU utilization
memory usage
training throughput
gradient stability
number of optimizer steps
```

Small batch size:

```text
lower memory usage
more noisy gradients
poor GPU utilization
more steps per epoch
```

Large batch size:

```text
better GPU utilization
more stable gradients
higher memory usage
fewer steps per epoch
```

In AI infrastructure and ML systems, batch size is one of the most important tuning knobs.

## 12. Key Formulas

```text
steps_per_epoch = ceil(dataset_size / batch_size)

total_steps = epochs * steps_per_epoch
```

If `drop_last=True`:

```text
steps_per_epoch = floor(dataset_size / batch_size)
```

## 13. Checklist

- [ ] Understand dataset size
- [ ] Understand batch size
- [ ] Understand epoch
- [ ] Understand step / iteration
- [ ] Understand full-batch gradient descent
- [ ] Understand stochastic gradient descent
- [ ] Understand mini-batch gradient descent
- [ ] Understand why shuffling is needed
- [ ] Understand the relationship between dataset size, batch size, epoch, and total steps
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass

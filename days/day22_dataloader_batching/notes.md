# Day 22: Batching and DataLoader Engineering

## 1. Goal

Today's goal is to understand batching and PyTorch DataLoader from both ML and systems perspectives.

Key concepts:

```text
batch size
Dataset
DataLoader
shuffle
num_workers
pin_memory
CPU -> GPU transfer
GPU utilization
```

---

## 2. Batch

A batch is a group of samples processed together.

Mathematically:

```text
batch_size = number of samples used for one gradient update
```

Systems-wise, batching helps:

```text
increase GPU utilization
reduce Python loop overhead
make matrix multiplication larger and more efficient
```

---

## 3. Dataset

A PyTorch Dataset defines how to access individual samples.

Minimum interface:

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return number_of_samples

    def __getitem__(self, idx):
        return x, y
```

Dataset answers:

```text
How many samples exist?
How do I get one sample?
```

---

## 4. DataLoader

A DataLoader wraps a Dataset and provides batches.

Example:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
```

DataLoader handles:

```text
batching
shuffling
parallel workers
collation
optional pinned memory
```

---

## 5. Shuffle

For training:

```python
shuffle=True
```

This randomizes sample order and reduces dependence on data ordering.

For validation and testing:

```python
shuffle=False
```

Evaluation does not update parameters, so deterministic order is preferred.

---

## 6. num_workers

`num_workers` controls how many worker processes load data.

```python
num_workers=0
```

means the main process loads data.

```python
num_workers=4
```

means 4 worker processes load data in parallel.

This can improve throughput when data loading is expensive.

Examples of expensive data loading:

```text
image decoding
resize / crop
augmentation
tokenization
reading from disk
```

---

## 7. pin_memory

For CUDA training, `pin_memory=True` can speed up CPU-to-GPU transfer.

Example:

```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    pin_memory=True,
)
```

Then transfer with:

```python
x_batch = x_batch.to(device, non_blocking=True)
```

Pinned memory is mainly useful when training on GPU.

---

## 8. Training Loop With DataLoader

```python
for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)

    y_hat = model(x_batch)
    loss = loss_fn(y_hat, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 9. Data Pipeline Bottleneck

Training pipeline:

```text
CPU loads/preprocesses data
CPU transfers batch to GPU
GPU runs forward/backward
```

If CPU loading is slow:

```text
GPU waits for data
GPU utilization drops
```

DataLoader tuning helps avoid this bottleneck.

---

## 10. ML Systems Connection

Data loading is a real ML infrastructure problem.

Important systems topics:

```text
batch size tuning
data prefetching
parallel data loading
CPU-GPU transfer
pinned memory
storage throughput
tokenization pipeline
distributed data loading
```

For large-scale training, the model can be fast but the input pipeline can still bottleneck the whole system.

---

## 11. Checklist

- [ ] Understand batch size
- [ ] Understand Dataset
- [ ] Understand DataLoader
- [ ] Understand train shuffle vs validation no shuffle
- [ ] Understand num_workers
- [ ] Understand pin_memory
- [ ] Understand CPU-to-GPU transfer
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass

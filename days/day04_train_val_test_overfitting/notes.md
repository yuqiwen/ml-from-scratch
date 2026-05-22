# Day 04: Train / Validation / Test Split and Overfitting

## 1. Goal

Today's goal is to understand how we evaluate whether a model truly generalizes to unseen data.

A low training loss does not always mean the model is good.

The key concepts are:

```text
train set
validation set
test set
overfitting
underfitting
generalization
```

## 2. Train Set

The train set is used to update model parameters.

During training, the model uses the train set to compute:

```text
prediction
loss
gradient
parameter update
```

For linear regression:

```text
y_hat = X_train @ w + b
loss = mean((y_hat - y_train)^2)
```

The optimizer updates `w` and `b` using the training loss.

## 3. Validation Set

The validation set is used to evaluate the model during development.

It helps us choose hyperparameters such as:

```text
learning rate
number of epochs
batch size
regularization strength
model complexity
```

The validation set is not used for direct parameter updates.

## 4. Test Set

The test set is used only for final evaluation.

It should not be used repeatedly to tune the model.

If we tune the model based on test performance, the test set is no longer a fair estimate of real-world performance.

## 5. Overfitting

Overfitting happens when the model performs very well on the training set but poorly on validation or test data.

Typical pattern:

```text
training loss: low
validation loss: high
```

This means the model may have memorized the training data instead of learning a general pattern.

## 6. Underfitting

Underfitting happens when the model is too simple or poorly trained.

Typical pattern:

```text
training loss: high
validation loss: high
```

This means the model cannot even fit the training data well.

## 7. Good Fit

A good model usually has:

```text
training loss: low
validation loss: low
gap between them: small
```

This means the model has learned a pattern that generalizes to unseen data.

## 8. Generalization

Generalization means the model can perform well on data it has not seen during training.

In machine learning, the real goal is not to minimize training loss only.

The real goal is to achieve good generalization.

## 9. ML Systems Connection

In real ML systems, training loops usually track:

```text
training loss
validation loss
metrics
best checkpoint
early stopping
```

A common rule is:

```text
save the model checkpoint with the best validation loss
```

not necessarily the final epoch.

This is because training loss can keep decreasing while validation loss starts increasing.

That pattern often indicates overfitting.

## 10. Checklist

- [ ] Understand train set
- [ ] Understand validation set
- [ ] Understand test set
- [ ] Understand overfitting
- [ ] Understand underfitting
- [ ] Understand generalization
- [ ] Understand why test set should not be used for tuning
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass

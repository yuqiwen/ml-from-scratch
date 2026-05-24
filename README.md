# ML From Scratch

This repository tracks my daily self-study in machine learning, deep learning, and LLM systems.

The goal is to:

- understand core ML ideas from first principles
- implement key algorithms from scratch
- write small runnable tests to verify understanding
- build a clean GitHub record of daily progress

## Repository Structure

Each study day gets its own folder under `days/`.

```text
ml-from-scratch/
|- README.md
|- .gitignore
|- requirements.txt
|- docs/
|  |- roadmap.md
|  |- workflow.md
|- reviews/
|  |- weekly-template.md
|- days/
|  |- day01-linear-regression/
|  |  |- notes.md
|  |  |- implementation.py
|  |  |- test_implementation.py
```

This structure keeps each topic self-contained:

- `notes.md`: concept summary, formulas, intuition, review checklist
- `implementation.py`: from-scratch code
- `test_implementation.py`: small runnable checks
- `docs/`: repository-level study plan and workflow rules
- `reviews/`: reusable review and reflection templates

## Recommended Daily Workflow

For each new study day:

1. Create a new folder like `days/day02-logistic-regression/`
2. Write the learning notes in `notes.md`
3. Implement the idea in `implementation.py`
4. Add a few simple tests in `test_implementation.py`
5. Update the index below

## Naming Convention

- day folders: `dayXX-topic-name`
- notes file: `notes.md`
- code file: `implementation.py`
- test file: `test_implementation.py`

This keeps the structure predictable and easy to scale.

## Study Index

- [Day 01: Linear Regression](days/day01-linear-regression/notes.md)
- [Day 02: Multiple Linear Regression](days/day02_multiple_linear_regression/notes.md)
- [Day 03: Batch Training and SGD](days/day03_batch_training_and_sgd/notes.md)
- [Day 04: Train / Validation / Test Split and Overfitting](days/day04_train_val_test_overfitting/notes.md)
- [Day 05: Logistic Regression](days/day05_logistic_regression/notes.md)
- [Day 06: Optimizer and Regularization](days/day06_optimizer_regularization/notes.md)

## Template For New Days

You can copy `days/_template/` and rename it to the next topic, for example:

- `days/day02-logistic-regression/`
- `days/day03-softmax-and-cross-entropy/`
- `days/day04-mlp/`

## Repository Guides

- [Roadmap](docs/roadmap.md)
- [Study Workflow](docs/workflow.md)
- [Weekly Review Template](reviews/weekly-template.md)

## Why This Layout

This repo is organized by day/topic instead of by file type because the main goal is learning continuity:

- easier to review what was learned on a specific day
- easier to publish a clean GitHub study log
- easier to keep notes, code, and tests together
- easier to scale when topics become larger

## Current Status

- Stage: ML fundamentals
- Latest completed topic: Optimizer and regularization
- Next suggested topics: softmax, cross-entropy, MLP, backpropagation

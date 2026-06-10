# Day 23: Model Serving Basics with FastAPI

## 1. Goal

Today's goal is to understand basic model serving.

Key concepts:

```text
model serving
inference API
request
response
FastAPI
model.eval()
torch.no_grad()
preprocess
postprocess
```

---

## 2. Training vs Serving

Training:

```text
input + target
forward
loss
backward
optimizer.step
checkpoint
```

Serving:

```text
request
preprocess
forward only
postprocess
response
```

Serving does not update model parameters.

---

## 3. Inference Mode

During serving, use:

```python
model.eval()

with torch.no_grad():
    prediction = model(x)
```

`model.eval()` switches the model to evaluation mode.

`torch.no_grad()` disables gradient tracking, which saves memory and improves inference speed.

---

## 4. Request and Response

Example request:

```json
{
  "features": [1.0, 2.0]
}
```

Example response:

```json
{
  "prediction": 3.72
}
```

---

## 5. FastAPI

FastAPI lets us expose a Python model as an HTTP API.

Example endpoint:

```python
@app.post("/predict")
def predict(request: PredictRequest):
    ...
```

---

## 6. Serving Pipeline

A simple serving pipeline:

```text
JSON request
-> validate input
-> convert to tensor
-> move to device
-> model forward
-> convert output to Python float
-> JSON response
```

---

## 7. ML Systems Connection

Model serving is a core AI infrastructure task.

Important production topics:

```text
latency
throughput
batching
GPU utilization
model loading
warmup
request validation
monitoring
autoscaling
versioning
A/B testing
```

For LLMs, serving adds:

```text
tokenization
prefill/decode
KV cache
continuous batching
streaming output
sampling
```

---

## 8. Project Files

This day is a small multi-file serving demo rather than the usual single `implementation.py` layout:

- `train_and_save_model.py`: train and save a tiny regression model
- `serve_model.py`: FastAPI app and prediction logic
- `client_demo.py`: simple HTTP client example
- `test_serving_logic.py`: local logic tests

---

## 9. Commands

Install dependencies:

```bash
pip install fastapi uvicorn requests
```

Train and save model:

```bash
python days/day23_model_serving_fastapi/train_and_save_model.py
```

Start server from the day folder:

```bash
cd days/day23_model_serving_fastapi
uvicorn serve_model:app --reload
```

In another terminal:

```bash
python days/day23_model_serving_fastapi/client_demo.py
```

Run tests:

```bash
python days/day23_model_serving_fastapi/test_serving_logic.py
```

---

## 10. Checklist

- [ ] Understand model serving
- [ ] Understand training vs inference
- [ ] Understand request / response
- [ ] Understand FastAPI endpoint
- [ ] Understand `model.eval()`
- [ ] Understand `torch.no_grad()`
- [ ] Run `train_and_save_model.py`
- [ ] Run FastAPI server
- [ ] Run `client_demo.py`
- [ ] Run `test_serving_logic.py`

# Day 26: Dockerize a FastAPI Model Serving Service

## 1. Goal

Today's goal is to package a FastAPI inference service into a Docker image.

Key concepts:

```text
Docker image
Docker container
Dockerfile
build
run
port mapping
container filesystem
model artifact
```

---

## 2. Why Docker?

A model service depends on:

```text
Python version
Python packages
application code
model checkpoint
system libraries
startup command
```

Docker packages these pieces into a repeatable runtime environment.

This helps reduce environment differences between:

```text
developer machine
testing environment
production server
```

---

## 3. Image vs Container

Docker image:

```text
static application template
```

Docker container:

```text
running instance of an image
```

One image can start multiple containers.

---

## 4. Dockerfile

A Dockerfile describes how to build an image.

Example:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 5. Build Time vs Runtime

`RUN` executes while building the image:

```dockerfile
RUN pip install -r requirements.txt
```

`CMD` executes when a container starts:

```dockerfile
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Port Mapping

The API listens inside the container on port 8000.

Run:

```bash
docker run --rm -p 8000:8000 day26-model-serving
```

Meaning:

```text
host port 8000 -> container port 8000
```

Then access:

```text
http://127.0.0.1:8000/health
```

---

## 7. Model Artifact

For this small demo, the model checkpoint is copied into the image:

```text
regression_model.pt
```

The service loads it during startup.

For large production models, model artifacts are often loaded from:

```text
object storage
model registry
mounted volume
network filesystem
```

---

## 8. Project Files

This day is a small multi-file deployment demo:

- `model.py`: model definition
- `train_model.py`: train and save the checkpoint
- `serve_model.py`: FastAPI service
- `test_service_logic.py`: local service logic tests
- `Dockerfile`: image build instructions
- `.dockerignore`: build context cleanup

---

## 9. Commands

Install dependencies locally:

```bash
pip install -r days/day26_docker_model_serving/requirements.txt
```

Train and save model:

```bash
python days/day26_docker_model_serving/train_model.py
```

Run local tests:

```bash
python days/day26_docker_model_serving/test_service_logic.py
```

Build Docker image from the day folder:

```bash
cd days/day26_docker_model_serving
docker build -t day26-model-serving .
```

Run container:

```bash
docker run --rm -p 8000:8000 day26-model-serving
```

Test health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

PowerShell alternative:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Test prediction endpoint:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"features":[1.0,2.0]}'
```

---

## 10. ML Systems Connection

Docker is commonly used for:

```text
model serving deployment
training jobs
CI testing
reproducible environments
Kubernetes workloads
autoscaling services
model version rollout
```

A common deployment flow is:

```text
source code
-> build Docker image
-> push image to registry
-> deploy containers
-> expose service endpoint
```

---

## 11. Checklist

- [ ] Understand image vs container
- [ ] Understand Dockerfile
- [ ] Understand RUN vs CMD
- [ ] Understand port mapping
- [ ] Train and save model
- [ ] Run local tests
- [ ] Build Docker image
- [ ] Start container
- [ ] Call health endpoint
- [ ] Call prediction endpoint

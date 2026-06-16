# Day 28: gRPC Model Serving Basics

## 1. Goal

Today's goal is to understand basic gRPC model serving.

Key concepts:

```text
gRPC
Remote Procedure Call
Protocol Buffers
.proto file
stub
unary RPC
serialization
client/server contract
```

---

## 2. REST vs gRPC

REST / FastAPI commonly uses:

```text
HTTP endpoint
JSON request
JSON response
```

Example:

```text
POST /predict
```

gRPC uses:

```text
service method
protobuf request
protobuf response
```

Example:

```text
ModelService.Predict()
```

---

## 3. Protocol Buffers

Protocol Buffers define structured messages.

Example:

```proto
message PredictRequest {
  repeated float features = 1;
}
```

This is conceptually similar to:

```python
class PredictRequest:
    features: list[float]
```

The number `1` is the protobuf field number.

It is used in binary serialization and should remain stable.

---

## 4. Service Definition

```proto
service ModelService {
  rpc Predict(PredictRequest) returns (PredictResponse);
}
```

This defines one unary RPC:

```text
one request
-> one response
```

---

## 5. Stub

A stub is a generated client-side proxy.

Example:

```python
stub = ModelServiceStub(channel)
response = stub.Predict(request)
```

The stub handles:

```text
serialization
network communication
waiting for response
deserialization
```

---

## 6. Generated Files

Run:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto
```

This generates:

```text
model_pb2.py
model_pb2_grpc.py
```

`model_pb2.py` contains protobuf message classes.

`model_pb2_grpc.py` contains gRPC service and stub classes.

Do not manually edit generated files.

---

## 7. RPC Flow

```text
client creates PredictRequest
-> stub serializes request
-> server receives request
-> model inference
-> server creates PredictResponse
-> client receives response
```

---

## 8. Project Files

This day is a small multi-file gRPC serving demo:

- `model.proto`: protobuf service and message schema
- `model.py`: model definition
- `train_model.py`: train and save checkpoint
- `grpc_server.py`: gRPC service implementation
- `grpc_client.py`: gRPC client example
- `test_serving_logic.py`: local service logic tests

---

## 9. Commands

Install dependencies:

```bash
pip install -r days/day28_grpc_model_serving/requirements.txt
```

Generate protobuf code:

```bash
python -m grpc_tools.protoc -Idays/day28_grpc_model_serving --python_out=days/day28_grpc_model_serving --grpc_python_out=days/day28_grpc_model_serving days/day28_grpc_model_serving/model.proto
```

Train model:

```bash
python days/day28_grpc_model_serving/train_model.py
```

Start server:

```bash
python days/day28_grpc_model_serving/grpc_server.py
```

In another terminal:

```bash
python days/day28_grpc_model_serving/grpc_client.py
```

Run tests:

```bash
python days/day28_grpc_model_serving/test_serving_logic.py
```

---

## 10. ML Systems Connection

gRPC is commonly used for:

```text
internal microservices
model serving systems
distributed training control plane
worker coordination
low-latency backend communication
typed service interfaces
streaming APIs
```

Compared with REST, gRPC commonly provides:

```text
smaller binary messages
generated client code
strict schemas
HTTP/2
streaming support
```

---

## 11. Checklist

- [ ] Understand RPC
- [ ] Understand gRPC
- [ ] Understand protobuf
- [ ] Understand field numbers
- [ ] Understand `.proto` files
- [ ] Understand generated code
- [ ] Understand stub
- [ ] Generate protobuf files
- [ ] Train model
- [ ] Start gRPC server
- [ ] Run gRPC client
- [ ] Run tests

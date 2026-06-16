import grpc

import model_pb2
import model_pb2_grpc


SERVER_ADDRESS = "localhost:50051"


def predict(
    stub: model_pb2_grpc.ModelServiceStub,
    features: list[float],
) -> float:
    request = model_pb2.PredictRequest(
        features=features
    )

    response = stub.Predict(request)

    return response.prediction


def check_health(
    stub: model_pb2_grpc.ModelServiceStub,
) -> None:
    request = model_pb2.HealthRequest()

    response = stub.Health(request)

    print("Health response")
    print(f"status = {response.status}")
    print(f"device = {response.device}")
    print()


def main() -> None:
    with grpc.insecure_channel(
        SERVER_ADDRESS
    ) as channel:
        stub = model_pb2_grpc.ModelServiceStub(
            channel
        )

        check_health(stub)

        prediction = predict(
            stub=stub,
            features=[1.0, 2.0],
        )

        print("Prediction response")
        print(f"prediction = {prediction}")


if __name__ == "__main__":
    main()
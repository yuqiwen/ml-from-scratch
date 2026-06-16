from concurrent import futures
from pathlib import Path

import grpc
import torch

import model_pb2
import model_pb2_grpc
from model import TinyRegressionModel


MODEL_PATH = Path(__file__).with_name("regression_model.pt")
SERVER_ADDRESS = "[::]:50051"


def load_model(
    path: Path = MODEL_PATH,
) -> tuple[TinyRegressionModel, torch.device]:
    """
    Load the trained model and move it to the serving device.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. "
            "Run `python train_model.py` first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = TinyRegressionModel(
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, device


class ModelService(
    model_pb2_grpc.ModelServiceServicer
):
    """
    Implementation of the protobuf ModelService.
    """

    def __init__(
        self,
        model: TinyRegressionModel,
        device: torch.device,
    ):
        self.model = model
        self.device = device

    def Predict(
        self,
        request: model_pb2.PredictRequest,
        context: grpc.ServicerContext,
    ) -> model_pb2.PredictResponse:
        """
        Unary RPC:
            PredictRequest -> PredictResponse
        """
        features = list(request.features)

        if len(features) != 2:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "features must contain exactly two values.",
            )

        X = torch.tensor(
            [features],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            prediction = self.model(X)

        return model_pb2.PredictResponse(
            prediction=float(prediction.item())
        )

    def Health(
        self,
        request: model_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> model_pb2.HealthResponse:
        return model_pb2.HealthResponse(
            status="ok",
            device=self.device.type,
        )


def create_server() -> grpc.Server:
    """
    Create and configure the gRPC server.
    """
    model, device = load_model()

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=4
        )
    )

    model_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelService(model, device),
        server,
    )

    server.add_insecure_port(SERVER_ADDRESS)

    return server


def main() -> None:
    server = create_server()
    server.start()

    print(
        "gRPC server started on "
        "localhost:50051"
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop(grace=2)


if __name__ == "__main__":
    main()

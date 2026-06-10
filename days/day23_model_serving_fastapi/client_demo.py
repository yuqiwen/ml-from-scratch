import requests


def main() -> None:
    url = "http://127.0.0.1:8000/predict"

    payload = {
        "features": [1.0, 2.0],
    }

    response = requests.post(url, json=payload, timeout=5)

    print("Status code:", response.status_code)
    print("Response JSON:", response.json())


if __name__ == "__main__":
    main()
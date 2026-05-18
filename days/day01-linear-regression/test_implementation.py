import numpy as np

from implementation import LinearRegressionFromScratch


def test_y_equals_2x():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    model = LinearRegressionFromScratch(learning_rate=0.01, epochs=2000)
    model.fit(x, y, verbose=False)

    print("Test 1: y = 2x")
    print(f"Learned w = {model.w:.6f}")
    print(f"Learned b = {model.b:.6f}")

    assert abs(model.w - 2.0) < 0.05
    assert abs(model.b - 0.0) < 0.20

    print("Passed.\n")


def test_y_equals_2x_plus_1():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)

    model = LinearRegressionFromScratch(learning_rate=0.01, epochs=3000)
    model.fit(x, y, verbose=False)

    print("Test 2: y = 2x + 1")
    print(f"Learned w = {model.w:.6f}")
    print(f"Learned b = {model.b:.6f}")

    assert abs(model.w - 2.0) < 0.05
    assert abs(model.b - 1.0) < 0.20

    print("Passed.\n")


def main():
    test_y_equals_2x()
    test_y_equals_2x_plus_1()
    print("All Day 01 tests passed.")


if __name__ == "__main__":
    main()

import torch

from implementation import train_small_model_with_optimizer


def test_sgd_updates_parameter_correctly():
    w = torch.tensor(2.0, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.1)

    loss = w ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Test 1: SGD updates parameter correctly")
    print(f"w = {w.item():.6f}")

    assert torch.isclose(w.detach(), torch.tensor(1.6), atol=1e-6)

    print("Passed.\n")


def test_optimizer_zero_grad_clears_gradient():
    w = torch.tensor(2.0, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.1)

    loss = w ** 2
    loss.backward()

    assert w.grad is not None

    optimizer.zero_grad()

    print("Test 2: optimizer.zero_grad clears gradient")
    print(f"w.grad = {w.grad}")

    assert w.grad is None or torch.isclose(w.grad, torch.tensor(0.0))

    print("Passed.\n")


def test_momentum_optimizer_has_state_after_step():
    w = torch.tensor(2.0, requires_grad=True)

    optimizer = torch.optim.SGD(
        [w],
        lr=0.1,
        momentum=0.9,
    )

    loss = w ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Test 3: momentum optimizer has state")
    print(f"optimizer.state = {optimizer.state}")

    assert len(optimizer.state) > 0

    print("Passed.\n")


def test_adam_optimizer_has_state_after_step():
    w = torch.tensor(2.0, requires_grad=True)

    optimizer = torch.optim.Adam(
        [w],
        lr=0.1,
    )

    loss = w ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Test 4: Adam optimizer has state")
    print(f"optimizer.state = {optimizer.state}")

    assert len(optimizer.state) > 0

    print("Passed.\n")


def test_all_optimizers_reduce_loss():
    optimizer_names = ["sgd", "momentum", "adam", "adamw"]

    print("Test 5: all optimizers reduce loss")

    for name in optimizer_names:
        losses, _ = train_small_model_with_optimizer(
            optimizer_name=name,
            num_epochs=50,
        )

        print(
            f"{name:8s}: "
            f"initial_loss={losses[0]:.6f}, "
            f"final_loss={losses[-1]:.6f}"
        )

        assert losses[-1] < losses[0]

    print("Passed.\n")


def main():
    test_sgd_updates_parameter_correctly()
    test_optimizer_zero_grad_clears_gradient()
    test_momentum_optimizer_has_state_after_step()
    test_adam_optimizer_has_state_after_step()
    test_all_optimizers_reduce_loss()

    print("All Day 11 tests passed.")


if __name__ == "__main__":
    main()
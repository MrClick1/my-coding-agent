from calculator import add


def test_adds_two_positive_integers() -> None:
    assert add(2, 3) == 5


def test_adds_negative_integer() -> None:
    assert add(-2, 5) == 3

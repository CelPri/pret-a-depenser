import pytest


def validate_input(age: int, income: float):
    if age < 0:
        raise ValueError("Age must be positive")
    if income <= 0:
        raise ValueError("Income must be positive")
    return True


def test_negative_age_raises_error():
    with pytest.raises(ValueError):
        validate_input(age=-5, income=1000)


def test_zero_income_raises_error():
    with pytest.raises(ValueError):
        validate_input(age=30, income=0)
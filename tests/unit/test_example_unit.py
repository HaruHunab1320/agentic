import pytest

def example_function_to_test(x):
    return x + 1

def test_example_function():
    """
    An example unit test.
    Unit tests should be focused, testing a single unit of code in isolation.
    """
    assert example_function_to_test(3) == 4
    assert example_function_to_test(0) == 1
    with pytest.raises(TypeError):
        example_function_to_test("not_a_number")

# More test cases for different scenarios can be added here.
# Consider parameterization for testing multiple inputs efficiently.
# Example:
# @pytest.mark.parametrize("input_val, expected_val", [
#     (1, 2),
#     (-1, 0),
#     (100, 101),
# ])
# def test_example_function_parameterized(input_val, expected_val):
#     assert example_function_to_test(input_val) == expected_val

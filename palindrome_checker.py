import re

def is_palindrome(text: str) -> bool:
    """
    Checks if a given string is a palindrome.

    A palindrome is a word, phrase, number, or other sequence of characters
    that reads the same backward as forward, such as "madam" or "racecar".
    The check is case-insensitive and ignores non-alphanumeric characters.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize the string: remove non-alphanumeric and convert to lowercase
    normalized_text = re.sub(r'[^a-z0-9]', '', text.lower())

    # Check if the normalized string is equal to its reverse
    return normalized_text == normalized_text[::-1]
import re

def is_palindrome(text: str) -> bool:
    """
    Checks if a given string is a palindrome.

    The function ignores case, spaces, and punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Remove non-alphanumeric characters and convert to lowercase
    processed_text = re.sub(r'[^a-z0-9]', '', text.lower())

    # Check if the processed text is equal to its reverse
    return processed_text == processed_text[::-1]
import re

def is_palindrome(text: str) -> bool:
    """
    Checks if a given string is a palindrome.

    The function ignores case, spaces, and punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Remove non-alphanumeric characters and convert to lowercase
    processed_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    if not processed_text:
        return True  # An empty string or string with only non-alphanumeric is a palindrome

    return processed_text == processed_text[::-1]
import re

def is_palindrome(text: str) -> bool:
    """
    Checks if a given string is a palindrome.

    A palindrome is a word, phrase, number, or other sequence of characters
    that reads the same backward as forward, such as "madam" or "racecar".
    The check is case-insensitive and ignores non-alphanumeric characters.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Remove non-alphanumeric characters and convert to lowercase
    processed_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    # Check if the processed text is empty
    if not processed_text:
        return True # Or False, depending on definition of palindrome for empty/non-alphanumeric string

    return processed_text == processed_text[::-1]

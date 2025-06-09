import unittest
from palindrome_checker import is_palindrome

class TestIsPalindrome(unittest.TestCase):

    def test_simple_palindromes(self):
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("madam"))
        self.assertTrue(is_palindrome("level"))

    def test_non_palindromes(self):
        self.assertFalse(is_palindrome("hello"))
        self.assertFalse(is_palindrome("world"))
        self.assertFalse(is_palindrome("python"))

    def test_case_insensitivity(self):
        self.assertTrue(is_palindrome("RaceCar"))
        self.assertTrue(is_palindrome("Madam"))
        self.assertTrue(is_palindrome("LeVel"))

    def test_palindromes_with_spaces(self):
        self.assertTrue(is_palindrome("a man a plan a canal panama"))
        self.assertTrue(is_palindrome("nurses run"))
        self.assertTrue(is_palindrome("was it a car or a cat i saw"))

    def test_palindromes_with_punctuation_and_spaces(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertTrue(is_palindrome("Eva, can I see bees in a cave?"))
        self.assertTrue(is_palindrome("Mr. Owl ate my metal worm."))
        self.assertTrue(is_palindrome("No 'x' in Nixon?"))

    def test_empty_string(self):
        # An empty string is considered a palindrome as it reads the same forwards and backward.
        self.assertTrue(is_palindrome(""))

    def test_single_character_string(self):
        self.assertTrue(is_palindrome("a"))
        self.assertTrue(is_palindrome("7"))
        self.assertTrue(is_palindrome("!")) # Non-alphanumeric, becomes empty, which is a palindrome

    def test_palindromes_with_numbers(self):
        self.assertTrue(is_palindrome("121"))
        self.assertTrue(is_palindrome("12321"))
        self.assertTrue(is_palindrome("A1bB1A"))
        self.assertTrue(is_palindrome("Was it a car or a cat I saw? 11"))


    def test_strings_with_only_non_alphanumeric(self):
        # These become empty strings after normalization, which are palindromes.
        self.assertTrue(is_palindrome("!@#$%^"))
        self.assertTrue(is_palindrome(" , . ? / ; ' : \" "))

    def test_non_string_input(self):
        with self.assertRaises(TypeError):
            is_palindrome(123)
        with self.assertRaises(TypeError):
            is_palindrome(None)
        with self.assertRaises(TypeError):
            is_palindrome(["racecar"])

if __name__ == '__main__':
    unittest.main()
import unittest
from palindrome_checker import is_palindrome

class TestIsPalindrome(unittest.TestCase):

    def test_simple_palindrome(self):
        self.assertTrue(is_palindrome("racecar"))

    def test_mixed_case_palindrome(self):
        self.assertTrue(is_palindrome("RaceCar"))

    def test_palindrome_with_spaces(self):
        self.assertTrue(is_palindrome("A man a plan a canal Panama"))

    def test_palindrome_with_punctuation(self):
        self.assertTrue(is_palindrome("Madam, I'm Adam."))

    def test_palindrome_with_numbers(self):
        self.assertTrue(is_palindrome("121"))
        self.assertTrue(is_palindrome("Was it a car or a cat I saw? 11"))

    def test_non_palindrome(self):
        self.assertFalse(is_palindrome("hello"))

    def test_empty_string(self):
        self.assertTrue(is_palindrome(""))

    def test_single_character_string(self):
        self.assertTrue(is_palindrome("a"))

    def test_string_with_only_spaces_and_punctuation(self):
        self.assertTrue(is_palindrome(" , . ? ! "))

    def test_unicode_palindrome(self):
        # Example with a simple unicode palindrome (though processing removes non-ascii for now)
        # This test will pass because non-alphanumeric are stripped.
        # If full unicode character support for palindromes (e.g. accents) is needed,
        # the regex and processing logic would need adjustment.
        self.assertTrue(is_palindrome("ΝΙΨΟΝ ΑΝΟΜΗΜΑΤΑ ΜΗ ΜΟΝΑΝ ΟΨΙΝ")) # Greek palindrome

    def test_long_palindrome(self):
        self.assertTrue(is_palindrome("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

    def test_long_non_palindrome(self):
        self.assertFalse(is_palindrome("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))

    def test_input_type_error(self):
        with self.assertRaises(TypeError):
            is_palindrome(123)
        with self.assertRaises(TypeError):
            is_palindrome(None)
        with self.assertRaises(TypeError):
            is_palindrome(["racecar"])

if __name__ == '__main__':
    unittest.main()
import unittest
from palindrome_checker import is_palindrome

class TestIsPalindrome(unittest.TestCase):

    def test_simple_palindrome(self):
        self.assertTrue(is_palindrome("racecar"))

    def test_mixed_case_palindrome(self):
        self.assertTrue(is_palindrome("RaceCar"))

    def test_palindrome_with_spaces(self):
        self.assertTrue(is_palindrome("A man a plan a canal Panama"))

    def test_palindrome_with_punctuation(self):
        self.assertTrue(is_palindrome("Madam, I'm Adam."))

    def test_palindrome_with_numbers(self):
        self.assertTrue(is_palindrome("121"))
        self.assertTrue(is_palindrome("Was it a car or a cat I saw?")) # Contains numbers implicitly via 'a'

    def test_non_palindrome(self):
        self.assertFalse(is_palindrome("hello"))

    def test_empty_string(self):
        self.assertTrue(is_palindrome(""))

    def test_single_character_string(self):
        self.assertTrue(is_palindrome("a"))

    def test_string_with_only_spaces_and_punctuation(self):
        self.assertTrue(is_palindrome(" , . ; ! ? "))

    def test_unicode_palindrome(self):
        self.assertTrue(is_palindrome("ΝΙΨΟΝ ΑΝΟΜΗΜΑΤΑ ΜΗ ΜΟΝΑΝ ΟΨΙΝ")) # Greek palindrome

    def test_unicode_non_palindrome(self):
        self.assertFalse(is_palindrome("Καλημέρα")) # Greek for "Good morning"

    def test_long_palindrome(self):
        self.assertTrue(is_palindrome("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

    def test_long_non_palindrome(self):
        self.assertFalse(is_palindrome("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))

    def test_input_type_error(self):
        with self.assertRaises(TypeError):
            is_palindrome(123)
        with self.assertRaises(TypeError):
            is_palindrome(None)
        with self.assertRaises(TypeError):
            is_palindrome(["racecar"])

if __name__ == '__main__':
    unittest.main()
import unittest
from palindrome_checker import is_palindrome

class TestIsPalindrome(unittest.TestCase):

    def test_simple_palindrome(self):
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("madam"))

    def test_mixed_case_palindrome(self):
        self.assertTrue(is_palindrome("RaceCar"))
        self.assertTrue(is_palindrome("Madam"))

    def test_palindrome_with_spaces(self):
        self.assertTrue(is_palindrome("race car"))
        self.assertTrue(is_palindrome("nurses run"))

    def test_palindrome_with_punctuation(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertTrue(is_palindrome("Was it a car or a cat I saw?"))
        self.assertTrue(is_palindrome("No 'x' in Nixon?"))

    def test_non_palindrome(self):
        self.assertFalse(is_palindrome("hello"))
        self.assertFalse(is_palindrome("world"))
        self.assertFalse(is_palindrome("python"))

    def test_empty_string(self):
        # An empty string is often considered a palindrome.
        # Or it could be defined as False, depending on requirements.
        # For this implementation, an empty string (or one with only non-alphanumerics)
        # is considered a palindrome.
        self.assertTrue(is_palindrome(""))
        self.assertTrue(is_palindrome(" "))
        self.assertTrue(is_palindrome("!@#$%^"))


    def test_single_character_string(self):
        self.assertTrue(is_palindrome("a"))
        self.assertTrue(is_palindrome("Z"))
        self.assertTrue(is_palindrome("7"))

    def test_numbers_as_string(self):
        self.assertTrue(is_palindrome("121"))
        self.assertTrue(is_palindrome("12321"))
        self.assertFalse(is_palindrome("123"))

    def test_non_string_input(self):
        with self.assertRaises(TypeError):
            is_palindrome(121)
        with self.assertRaises(TypeError):
            is_palindrome(None)
        with self.assertRaises(TypeError):
            is_palindrome(["r", "a", "c", "e", "c", "a", "r"])

if __name__ == '__main__':
    unittest.main()

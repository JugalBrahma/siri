"""The prompt files load, and their placeholders are always filled."""

import unittest

from prompts import PROMPT_DIR, load_prompt

PLACEHOLDERS = {
    "researcher": {"max_searches": 2, "max_searches_after_failure": 3},
    "screen_reader": {"max_rounds": 6},
    "vision": {"what": "the screen", "width": 1470, "height": 956, "question": "Which app is open?"},
}


class TestPrompts(unittest.TestCase):
    def test_every_prompt_file_loads_with_its_values(self):
        names = sorted(path.stem for path in PROMPT_DIR.glob("*.md"))
        self.assertIn("screen_reader", names)
        for name in names:
            with self.subTest(name):
                text = load_prompt(name, **PLACEHOLDERS.get(name, {}))
                self.assertTrue(text.strip())
                self.assertNotIn("$", text)

    def test_values_are_filled_in(self):
        self.assertIn("at most 6 rounds", load_prompt("screen_reader", max_rounds=6))
        self.assertTrue(load_prompt("vision", what="the Chrome window", width=10, height=20, question="Q?")
                        .startswith("This is a screenshot of the Chrome window (10x20 pixels). Q?"))

    def test_missing_value_fails_loudly(self):
        with self.assertRaises(KeyError):
            load_prompt("screen_reader")

    def test_unknown_prompt_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("no_such_prompt")


if __name__ == "__main__":
    unittest.main()

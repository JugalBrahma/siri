import unittest

from agents.output_sanitizer import redact_secrets


class TestRedactSecrets(unittest.TestCase):
    def test_tavily_key_is_redacted(self):
        text = "key tvly-dev-abc123zzKEYxyz789 here"
        self.assertEqual(redact_secrets(text), "key [REDACTED_SECRET] here")

    def test_tavily_key_in_url_is_redacted(self):
        text = "https://mcp.tavily.com/mcp/?tavilyApiKey=abczzKEY123&x=1"
        cleaned = redact_secrets(text)
        self.assertNotIn("zzKEY", cleaned)
        self.assertIn("&x=1", cleaned)

    def test_existing_patterns_still_work(self):
        self.assertEqual(redact_secrets("ghp_" + "a" * 36), "[REDACTED_SECRET]")

    def test_normal_text_is_untouched(self):
        text = "Tavily found 3 results about cats."
        self.assertEqual(redact_secrets(text), text)


if __name__ == "__main__":
    unittest.main()

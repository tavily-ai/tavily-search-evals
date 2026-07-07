"""Test YoucomHandler post-processing against a mocked You.com Search response.

Stubs `gpt_researcher` (pulled in transitively by handlers/__init__.py via
gptr_handler) so the test can run without the full GPTR/langchain dependency
chain — YoucomHandler itself never uses it.

Run: python -m unittest tests.test_youcom_handler
"""
import os
import sys
import types
import unittest

# Stub gpt_researcher so handlers/__init__.py can import gptr_handler without
# requiring the (heavy, fragile) gpt_researcher/langchain chain to be installed.
_gptr_stub = types.ModuleType("gpt_researcher")
_gptr_stub.GPTResearcher = object  # placeholder; never used by YoucomHandler
sys.modules.setdefault("gpt_researcher", _gptr_stub)

os.environ.setdefault("YDC_API_KEY", "test-key-for-post-process")

from handlers.youcom_handler import YoucomHandler
from utils.utils import EvaluationType


class YoucomHandlerPostProcessTest(unittest.IsolatedAsyncioTestCase):
    """post_process is the observable behaviour: a You.com-shaped response
    must produce a prompt string containing each result's URL and snippet."""

    def _fake_response(self):
        return {
            "results": {
                "web": [
                    {
                        "title": "You.com",
                        "url": "https://you.com",
                        "description": "You.com is a search engine.",
                        "snippets": ["You.com is a search engine."],
                    },
                    {
                        "title": "Nous Research",
                        "url": "https://nousresearch.com",
                        "description": "Nous Research builds AI agents.",
                        "snippets": ["Nous Research builds AI agents."],
                    },
                ],
                "news": [],
            },
            "metadata": {"search_uuid": "test-uuid"},
        }

    async def test_post_process_extracts_url_and_content(self):
        handler = YoucomHandler()
        search_result = {"search_response": self._fake_response()}
        formatted, token_count, token_avg = await handler.post_process(
            search_result, evaluation_type=EvaluationType.SIMPLEQA
        )
        self.assertIn("https://you.com", formatted)
        self.assertIn("You.com is a search engine.", formatted)
        self.assertIn("https://nousresearch.com", formatted)
        self.assertIn("Nous Research builds AI agents.", formatted)
        self.assertGreater(token_count, 0)
        self.assertGreater(token_avg, 0)

    async def test_post_process_handles_empty_response(self):
        handler = YoucomHandler()
        search_result = {"search_response": {"results": {"web": [], "news": []}, "metadata": {}}}
        formatted, token_count, token_avg = await handler.post_process(
            search_result, evaluation_type=EvaluationType.SIMPLEQA
        )
        self.assertEqual(formatted, "")
        self.assertEqual(token_count, 0)
        self.assertEqual(token_avg, 0)

    async def test_post_process_handles_missing_search_response(self):
        handler = YoucomHandler()
        formatted, token_count, token_avg = await handler.post_process(
            {"search_response": None}
        )
        self.assertEqual(formatted, "")
        self.assertEqual(token_count, 0)
        self.assertEqual(token_avg, 0)


if __name__ == "__main__":
    unittest.main()

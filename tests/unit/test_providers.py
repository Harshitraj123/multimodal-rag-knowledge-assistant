"""Unit tests for local Ollama LLM and Vision providers."""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.generation.llm_provider import LLMProvider
from app.generation.vision_provider import VisionProvider


class TestLLMProvider(unittest.TestCase):
    def test_default_initialization(self):
        llm = LLMProvider()
        self.assertEqual(llm.provider, "ollama")
        self.assertEqual(llm.model, "llama3.2")
        self.assertEqual(llm.base_url, "http://localhost:11434/v1")
        self.assertTrue(llm.available)

    def test_backward_compatibility_api_key(self):
        llm = LLMProvider(base_url="gsk_test123", model="llama-3.3-70b-versatile")
        self.assertEqual(llm.provider, "groq")
        self.assertEqual(llm.api_key, "gsk_test123")
        self.assertTrue(llm.available)

    def test_groq_unavailable_without_key(self):
        llm = LLMProvider(provider="groq", api_key=None)
        self.assertFalse(llm.available)
        with self.assertRaises(RuntimeError):
            llm._get_client()

    def test_generation_with_mock_client(self):
        llm = LLMProvider(base_url="http://localhost:11434/v1", model="llama3.2")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a response from Ollama."
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        output = llm.generate("Hello world", system_prompt="You are helpful")
        self.assertEqual(output, "This is a response from Ollama.")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "llama3.2")
        self.assertEqual(len(call_kwargs["messages"]), 2)
        self.assertEqual(call_kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["messages"][1]["role"], "user")

    def test_callable_interface(self):
        llm = LLMProvider()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Callable answer"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        llm._client = mock_client

        result = llm("Test query")
        self.assertEqual(result, "Callable answer")

    def test_unreachable_ollama_fallback_error(self):
        llm = LLMProvider(base_url="http://localhost:99999/v1", model="llama3.2")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Connection refused")
        llm._client = mock_client

        with self.assertRaises(RuntimeError) as ctx:
            llm.generate("Test prompt")
        self.assertIn("Local Ollama generation failed", str(ctx.exception))


class TestVisionProvider(unittest.TestCase):
    def test_default_initialization(self):
        vision = VisionProvider()
        self.assertEqual(vision.provider, "ollama")
        self.assertEqual(vision.model, "llava")
        self.assertEqual(vision.base_url, "http://localhost:11434/v1")
        self.assertTrue(vision.available)

    def test_openai_requires_api_key(self):
        vision = VisionProvider(provider="openai", api_key=None)
        self.assertFalse(vision.available)
        with self.assertRaises(RuntimeError):
            vision._get_client()

    def test_describe_image_with_mock_client(self):
        vision = VisionProvider(provider="ollama", model="llava")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "A bar chart showing quarterly revenue growth."
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        vision._client = mock_client

        img_bytes = b"fake-png-bytes"
        description = vision.describe_image(img_bytes, question="What does this chart show?")
        self.assertEqual(description, "A bar chart showing quarterly revenue growth.")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "llava")
        content = call_kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "What does this chart show?")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_generate_visual_description_fallback_on_error(self):
        vision = VisionProvider(provider="ollama", model="llava")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Ollama offline")
        vision._client = mock_client

        description = vision.generate_visual_description(b"image_data", source_info="chart.png")
        self.assertIn("Vision model (llava) not available or unreachable", description)


# Try testing Settings if pydantic_settings is installed
try:
    from app.config.settings import Settings

    class TestSettingsOllama(unittest.TestCase):
        def test_ollama_settings_defaults(self):
            s = Settings()
            self.assertEqual(s.llm_provider, "ollama")
            self.assertEqual(s.ollama_base_url, "http://localhost:11434/v1")
            self.assertEqual(s.ollama_model, "llama3.2")
            self.assertEqual(s.vision_provider, "ollama")
            self.assertEqual(s.vision_model, "llava")
            self.assertTrue(s.llm_available)
            self.assertTrue(s.vision_available)

        def test_report_capabilities(self):
            s = Settings()
            caps = s.report_capabilities()
            self.assertTrue(caps["llm_generation"])
            self.assertTrue(caps["vision_analysis"])
            self.assertEqual(caps["llm_model"], "llama3.2")
            self.assertEqual(caps["vision_model"], "llava")
except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()

"""Automated tests for FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient
from api.app import create_app
from api.schemas import ChatRequest
from api.service import SiriAgentService


class FakeState:
    def __init__(self, tasks=None):
        self.tasks = tasks or []


class FakeBuilder:
    def __init__(self, output: str = "Test response from Siri"):
        self.output = output
        self.graph = self

    def build(self):
        return self

    def create_turn_state(self, messages):
        return {"messages": messages}

    def get_state(self, config=None):
        return FakeState([])

    def stream(self, initial_state, stream_mode="values", config=None):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        messages = initial_state.get("messages", []) if isinstance(initial_state, dict) else []
        yield {"messages": messages, "next": "supervisor"}
        yield {"messages": [FakeMessage(self.output)], "next": "output_sanitizer"}


class FakeMemoryStore:
    def __init__(self, profile: str = "- Fake memory profile"):
        self.profile = profile

    def format_for_injection(self, user_id=None) -> str:
        return self.profile

    def get_last_retrieval_metrics(self) -> dict:
        return {"fact_count": 1}


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.fake_builder = FakeBuilder("Hello! How can I help you today?")
        self.fake_memory = FakeMemoryStore("- User lives in Tokyo")
        self.service = SiriAgentService(
            default_user_id="test_user",
            memory_store=self.fake_memory,
            builder=self.fake_builder,
        )
        self.app = create_app(agent_service=self.service)
        self.app.state.agent_service = self.service
        self.client = TestClient(self.app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("docs_url", data)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["semantic_memory_enabled"])
        self.assertEqual(data["default_user_id"], "test_user")

    def test_chat_single_message(self):
        response = self.client.post(
            "/chat",
            json={"message": "What is the weather?", "user_id": "test_user"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "Hello! How can I help you today?")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user_id"], "test_user")
        self.assertGreaterEqual(data["hop_count"], 1)

    def test_chat_multi_turn_messages(self):
        response = self.client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "user_id": "test_user",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "Hello! How can I help you today?")

    def test_chat_validation_empty_fails(self):
        response = self.client.post("/chat", json={})
        self.assertEqual(response.status_code, 422)

    def test_chat_streaming(self):
        response = self.client.post(
            "/chat/stream",
            json={"message": "Stream this test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn("node_update", body)
        self.assertIn("complete", body)

    def test_get_memory_profile(self):
        response = self.client.get("/memory/test_user")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user_id"], "test_user")
        self.assertEqual(data["formatted_profile"], "- User lives in Tokyo")

    def test_cors_headers(self):
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)

    def test_clear_memory(self):
        response = self.client.delete("/memory/test_user")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user_id"], "test_user")

    def test_chat_resume_endpoint(self):
        response = self.client.post(
            "/chat/resume",
            json={"response": "Tokyo", "user_id": "test_user"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "Hello! How can I help you today?")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user_id"], "test_user")

    def test_chat_resume_stream_endpoint(self):
        response = self.client.post(
            "/chat/resume/stream",
            json={"response": "Tokyo", "user_id": "test_user"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn("node_update", body)
        self.assertIn("complete", body)


if __name__ == "__main__":
    unittest.main()

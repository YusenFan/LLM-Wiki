"""Azure protocol and embedding input/cache checks. All inference and CLI calls are mocked."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import llm_wiki_bench
import llm_client
import model_auth
from embedding_client import EmbeddingClient
import smoke_azure_models


def completion(content="OK", *, reason="stop", **fields):
    return Mock(status_code=200, json=Mock(return_value={
        "choices": [{"message": {"role": "assistant", "content": content, **fields}, "finish_reason": reason}],
        "usage": {"total_tokens": 12}}))


class AzureLLMTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_BASE_URL": "https://unit.services.ai.azure.com/openai/v1",
                                         "LLM_MAX_ATTEMPTS": "2"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_auth_errors_do_not_retry_or_disable_json(self):
        with patch("llm_client.requests.post", return_value=Mock(status_code=401, text="secret-body")) as post:
            with self.assertRaisesRegex(llm_client.ModelCallError, "HTTP 401") as failure:
                llm_client.call_llm_json("JSON", "q")
            self.assertEqual(post.call_count, 1)
            self.assertNotIn("secret", str(failure.exception))

    def test_only_unsupported_json_mode_retries_without_response_format(self):
        bad = Mock(status_code=400, text="response_format json_object is not supported")
        with patch("llm_client.requests.post", side_effect=[bad, completion('{"ok":true}')]) as post:
            self.assertEqual(llm_client.call_llm_json("JSON", "q"), {"ok": True})
            self.assertIn("response_format", post.call_args_list[0].kwargs["json"])
            self.assertNotIn("response_format", post.call_args_list[1].kwargs["json"])
        with patch("llm_client.requests.post", return_value=Mock(status_code=400, text="invalid deployment")) as post:
            with self.assertRaises(llm_client.ModelCallError):
                llm_client.call_llm_json("JSON", "q")
            self.assertEqual(post.call_count, 1)

    def test_reasoning_only_and_truncated_results_are_not_answers(self):
        for response in (completion(None, reasoning_content="private reasoning"), completion("partial", reason="length")):
            with self.subTest(response=response), patch("llm_client.requests.post", return_value=response) as post:
                with self.assertRaises(llm_client.ModelCallError):
                    llm_client.call_llm("s", "q")
                self.assertEqual(post.call_count, 1)

    def test_tool_budget_and_provider_fields_survive_roundtrip(self):
        tool = {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        with patch.dict(os.environ, {"LLM_TOOL_MAX_TOKENS": "8192", "LLM_TOKEN_PARAMETER": "max_completion_tokens"}), \
                patch("llm_client.requests.post", return_value=completion(None, tool_calls=[tool], reasoning_content="state")) as post:
            result = llm_client.call_llm_with_tools([{"role": "user", "content": "q"}], [], model="wiki-glm-51")
            self.assertEqual(result["tool_calls"], [tool])
            self.assertEqual(result["reasoning_content"], "state")
            self.assertEqual(result["_usage"], {"total_tokens": 12})
            self.assertEqual(post.call_args.args[0], "https://unit.services.ai.azure.com/openai/v1/chat/completions")
            self.assertEqual(post.call_args.kwargs["json"]["max_completion_tokens"], 8192)
            self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_rate_limit_honors_retry_after(self):
        limited = Mock(status_code=429, text="limited", headers={"Retry-After": "3"})
        with patch("llm_client.requests.post", side_effect=[limited, completion()]) as post, patch("llm_client.time.sleep") as sleep:
            self.assertEqual(llm_client.call_llm("s", "q"), "OK")
            self.assertEqual(post.call_count, 2)
            sleep.assert_called_once_with(3)

    def test_json_requires_an_object(self):
        with patch("llm_client.requests.post", return_value=completion("[1,2]")):
            with self.assertRaisesRegex(llm_client.ModelCallError, "JSON object"):
                llm_client.call_llm_json("JSON", "q")


class AzureAuthTest(unittest.TestCase):
    def setUp(self):
        model_auth._tokens.clear()
        self.addCleanup(model_auth._tokens.clear)

    def test_cli_token_cached_and_refreshed_without_outputting_credentials(self):
        with patch.dict(os.environ, {"LLM_AUTH_MODE": "azure_cli"}, clear=True), \
                patch("model_auth.subprocess.run", return_value=Mock(stdout=json.dumps({
                    "accessToken": "test-token", "expires_on": time.time() + 3600}))) as run:
            endpoint = "https://unit.services.ai.azure.com/openai/v1"
            self.assertEqual(model_auth.auth_headers("LLM", endpoint=endpoint)["Authorization"], "Bearer test-token")
            model_auth.auth_headers("LLM", endpoint=endpoint)
            self.assertEqual(run.call_count, 1)
            resource = "https://cognitiveservices.azure.com/"
            model_auth._tokens[resource] = ("old-token", time.time() - 1)
            model_auth.auth_headers("LLM", endpoint=endpoint)
            self.assertEqual(run.call_count, 2)

    def test_cli_auth_rejects_non_azure_or_http_endpoints(self):
        with patch.dict(os.environ, {"LLM_AUTH_MODE": "azure_cli"}), patch("model_auth.subprocess.run") as run:
            for url in ("https://third-party.example/v1", "http://unit.services.ai.azure.com"):
                with self.assertRaises(ValueError):
                    model_auth.auth_headers("LLM", endpoint=url)
            run.assert_not_called()


class AzureEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "vectors.sqlite3"
        self.env = patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-3-large",
            "EMBEDDING_ENDPOINT": "https://unit.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
            "EMBEDDING_DIMENSIONS": "2"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, dimensions=2):
        return Mock(ok=True, status_code=200, json=Mock(return_value={
            "data": [{"index": 0, "embedding": [1.0] * dimensions}]}))

    def test_documents_queries_are_unprefixed_and_reuse_cache(self):
        with patch("embedding_client.requests.post", return_value=self.response()) as post:
            client = EmbeddingClient(self.cache)
            client.embed_documents(["same words"])
            self.assertEqual(post.call_args.args[0], "https://unit.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15")
            self.assertEqual(post.call_args.kwargs["json"]["input"], ["same words"])
            client.embed_queries(["same words"])
            query = post.call_args.kwargs["json"]["input"][0]
            self.assertEqual(query, "same words")
            client.embed_queries(["same words"])
            client.embed_documents(["same words"])
            self.assertEqual(post.call_count, 1)
            with patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "3"}):
                post.return_value = self.response(3)
                self.assertEqual(len(EmbeddingClient(self.cache).embed_documents(["same words"])[0]), 3)
            self.assertEqual(post.call_count, 2)

    def test_azure_api_key_header(self):
        with patch.dict(os.environ, {"EMBEDDING_AUTH_MODE": "api_key",
                "EMBEDDING_API_KEY_HEADER": "api-key", "EMBEDDING_API_KEY": "test-key"}), \
                patch("embedding_client.requests.post", return_value=self.response()) as post:
            EmbeddingClient(self.cache).embed_documents(["solar"])
            self.assertEqual(post.call_args.kwargs["headers"]["api-key"], "test-key")
            self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_explicit_endpoint_never_inherits_unrelated_chat_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "chat-secret"}), \
                patch("embedding_client.requests.post", return_value=self.response()) as post:
            client = EmbeddingClient(self.cache)
            client.embed_documents(["a"])
            self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_tei_payload_and_return_shape(self):
        with patch.dict(os.environ, {"EMBEDDING_PROTOCOL": "tei"}), \
                patch("embedding_client.requests.post", return_value=Mock(ok=True, status_code=200,
                    json=Mock(return_value=[[3.0, 4.0]]))) as post:
            vector = EmbeddingClient(self.cache).embed_queries(["solar"])[0]
            self.assertEqual(vector, [0.6, 0.8])
            body = post.call_args.kwargs["json"]
            self.assertFalse(body["truncate"])
            self.assertIn("inputs", body)
            self.assertNotIn("input", body)

    def test_wrong_dimension_is_not_cached(self):
        with patch("embedding_client.requests.post", return_value=self.response(3)) as post:
            client = EmbeddingClient(self.cache)
            with self.assertRaisesRegex(RuntimeError, "malformed vectors"):
                client.embed_documents(["a"])
            post.return_value = self.response()
            self.assertEqual(len(client.embed_documents(["a"])[0]), 2)
            self.assertEqual(post.call_count, 2)

    def test_oversize_inputs_fail_before_network(self):
        with patch.dict(os.environ, {"EMBEDDING_MAX_INPUT_BYTES": "6"}), patch("embedding_client.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "byte budget"):
                EmbeddingClient(self.cache).embed_documents(["中文长"])
            post.assert_not_called()


class SmokePreflightTest(unittest.TestCase):
    def test_llm_scope_does_not_claim_embedding_readiness(self):
        values = {"OPENAI_BASE_URL": "https://unit.services.ai.azure.com/openai/v1",
                  "LLM_FAST_MODEL": "glm-deployment", "LLM_PREMIUM_MODEL": "glm-deployment",
                  "LLM_AUTH_MODE": "azure_cli"}
        with patch.dict(os.environ, values, clear=True):
            smoke_azure_models.preflight("llm")
            with self.assertRaisesRegex(RuntimeError, "EMBEDDING_ENDPOINT"):
                smoke_azure_models.preflight("full")

    def test_missing_configuration_blocks_before_inference(self):
        with patch.dict(os.environ, {}, clear=True), patch("llm_client.requests.post") as post:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_BASE_URL"):
                smoke_azure_models.preflight()
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()

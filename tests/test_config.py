import os
import pytest
from unittest.mock import patch


class TestMentatConfig:
    def test_defaults(self):
        from mentat.core.hub import MentatConfig

        # Remove all MENTAT_ env vars so .env doesn't interfere
        mentat_keys = [k for k in os.environ if k.startswith("MENTAT_")]
        cleaned = {k: v for k, v in os.environ.items() if not k.startswith("MENTAT_")}
        with patch.dict(os.environ, cleaned, clear=True):
            c = MentatConfig()
        assert c.db_path == "./mentat_db"
        assert c.storage_dir == "./mentat_files"
        assert c.summary_model == "gpt-4o-mini"
        assert c.embedding_model == "text-embedding-3-small"
        assert c.embedding_provider == "litellm"
        assert c.summary_api_key == ""
        assert c.summary_api_base == ""
        assert c.embedding_api_key == ""
        assert c.embedding_api_base == ""
        # instruction_model falls back to summary_model
        assert c.instruction_model == "gpt-4o-mini"
        assert c.instruction_api_key == ""
        assert c.instruction_api_base == ""

    def test_explicit_args_override_defaults(self):
        from mentat.core.hub import MentatConfig

        c = MentatConfig(
            summary_model="anthropic/claude-sonnet-4-5-20250929",
            instruction_model="openai/gpt-4o",
            embedding_model="openai/text-embedding-3-large",
            summary_api_key="sk-test",
            summary_api_base="http://localhost:8080",
            instruction_api_key="sk-inst",
            instruction_api_base="http://inst:9090",
        )
        assert c.summary_model == "anthropic/claude-sonnet-4-5-20250929"
        assert c.instruction_model == "openai/gpt-4o"
        assert c.embedding_model == "openai/text-embedding-3-large"
        assert c.summary_api_key == "sk-test"
        assert c.summary_api_base == "http://localhost:8080"
        assert c.instruction_api_key == "sk-inst"
        assert c.instruction_api_base == "http://inst:9090"

    def test_env_vars_override_defaults(self):
        from mentat.core.hub import MentatConfig

        env = {
            "MENTAT_DB_PATH": "/tmp/test_db",
            "MENTAT_STORAGE_DIR": "/tmp/test_files",
            "MENTAT_SUMMARY_MODEL": "ollama/llama3",
            "MENTAT_SUMMARY_API_KEY": "sk-env",
            "MENTAT_SUMMARY_API_BASE": "http://env:9000",
            "MENTAT_INSTRUCTION_MODEL": "openai/gpt-4o",
            "MENTAT_INSTRUCTION_API_KEY": "sk-inst-env",
            "MENTAT_INSTRUCTION_API_BASE": "http://inst-env:9000",
            "MENTAT_EMBEDDING_MODEL": "cohere/embed-english-v3.0",
            "MENTAT_EMBEDDING_API_KEY": "sk-embed-env",
            "MENTAT_EMBEDDING_API_BASE": "http://embed:9000",
        }
        with patch.dict(os.environ, env, clear=False):
            c = MentatConfig()

        assert c.db_path == "/tmp/test_db"
        assert c.storage_dir == "/tmp/test_files"
        assert c.summary_model == "ollama/llama3"
        assert c.summary_api_key == "sk-env"
        assert c.summary_api_base == "http://env:9000"
        assert c.instruction_model == "openai/gpt-4o"
        assert c.instruction_api_key == "sk-inst-env"
        assert c.instruction_api_base == "http://inst-env:9000"
        assert c.embedding_model == "cohere/embed-english-v3.0"
        assert c.embedding_api_key == "sk-embed-env"
        assert c.embedding_api_base == "http://embed:9000"

    def test_instruction_model_falls_back_to_summary_model(self):
        from mentat.core.hub import MentatConfig

        c = MentatConfig(summary_model="ollama/llama3", summary_api_key="sk-sum")
        assert c.instruction_model == "ollama/llama3"
        assert c.instruction_api_key == "sk-sum"

    def test_explicit_args_beat_env_vars(self):
        from mentat.core.hub import MentatConfig

        env = {"MENTAT_SUMMARY_MODEL": "env-model"}
        with patch.dict(os.environ, env, clear=False):
            c = MentatConfig(summary_model="explicit-model")

        assert c.summary_model == "explicit-model"

"""Configuration: secrets load from AWS Secrets Manager first, env vars as fallback.

Expected Secrets Manager secret (JSON): ``support-agent/config``
    {"LANGSMITH_API_KEY": "...", "BEDROCK_MODEL_ID": "..."}

Local/dev fallback env vars: LANGSMITH_API_KEY, LANGSMITH_PROJECT,
AWS_REGION, BEDROCK_MODEL_ID, MOCK_MODE (default "true").
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

SECRET_NAME = os.getenv("SECRET_NAME", "support-agent/config")


def _from_secrets_manager(secret_name: str) -> dict:
    """Best-effort fetch of a JSON secret. Returns {} on any failure."""
    try:
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=secret_name)
        payload = json.loads(resp.get("SecretString") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:  # no creds / no network / no secret -> env fallback
        log.debug("Secrets Manager unavailable (%s); using env vars", exc)
        return {}


@dataclass
class Settings:
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_max_tokens: int = 512
    langsmith_api_key: str | None = None
    langsmith_project: str = "langgraph-support-agent"
    mock_mode: bool = True
    data_dir: str = "data"

    @classmethod
    def load(cls) -> "Settings":
        secrets = _from_secrets_manager(SECRET_NAME)

        def get(k, d=None):
            return secrets.get(k, os.getenv(k, d))

        mock_raw = get("MOCK_MODE", "true")
        return cls(
            aws_region=get("AWS_REGION", "us-east-1"),
            bedrock_model_id=get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            langsmith_api_key=get("LANGSMITH_API_KEY"),
            langsmith_project=get("LANGSMITH_PROJECT", "langgraph-support-agent"),
            mock_mode=str(mock_raw).lower() not in ("false", "0", "no"),
            data_dir=get("DATA_DIR", "data"),
        )

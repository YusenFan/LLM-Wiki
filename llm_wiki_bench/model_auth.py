"""Endpoint authentication. Azure CLI tokens remain in process memory."""
import json
import os
import subprocess
import time
from urllib.parse import urlparse

_tokens = {}


def auth_headers(prefix: str, api_key: str = "", endpoint: str = "") -> dict:
    mode = os.environ.get(f"{prefix}_AUTH_MODE", "api_key")
    headers = {"Content-Type": "application/json"}
    if mode == "azure_cli":
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host.endswith((".azure.com", ".azure.net", ".microsoft.com")):
            raise ValueError("azure_cli authentication requires an Azure endpoint")
        resource = os.environ.get(f"{prefix}_AZURE_RESOURCE", "https://cognitiveservices.azure.com/")
        cached = _tokens.get(resource)
        if cached is None or cached[1] <= time.time() + 120:
            try:
                result = subprocess.run(
                    ["az", "account", "get-access-token", "--resource", resource, "-o", "json"],
                    capture_output=True, text=True, timeout=45, check=True)
                data = json.loads(result.stdout)
                token = data["accessToken"]
                if not isinstance(token, str) or not token:
                    raise ValueError("missing token")
                expires = float(data.get("expires_on", time.time() + 300))
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
                raise RuntimeError("Azure CLI token acquisition failed; check az login and endpoint access") from None
            cached = _tokens[resource] = (token, expires)
        headers["Authorization"] = f"Bearer {cached[0]}"
    elif mode == "api_key":
        header = os.environ.get(f"{prefix}_API_KEY_HEADER", "Authorization")
        if header not in {"Authorization", "api-key"}:
            raise ValueError(f"{prefix}_API_KEY_HEADER must be Authorization or api-key")
        if api_key:
            headers[header] = f"Bearer {api_key}" if header == "Authorization" else api_key
    else:
        raise ValueError(f"Unsupported {prefix}_AUTH_MODE: {mode}")
    return headers

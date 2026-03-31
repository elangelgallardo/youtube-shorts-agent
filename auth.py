"""Authentication helpers for Google APIs.

Run `python3 auth.py` once interactively to complete the YouTube OAuth flow
and persist the token. Subsequent runs refresh silently.
"""
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config

logger = logging.getLogger(__name__)


# ── YouTube OAuth 2.0 ───────────────────────────────────────────────────────

def get_youtube_credentials() -> Credentials:
    """Return valid OAuth2 credentials for YouTube, refreshing or re-authorizing as needed."""
    creds: Credentials | None = None
    token_path = Path(config.YOUTUBE_TOKEN_FILE)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), config.YOUTUBE_SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing YouTube OAuth token…")
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
        return creds

    # Interactive flow — only needed once
    secrets_path = Path(config.YOUTUBE_CLIENT_SECRETS)
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"YouTube client secrets not found at {secrets_path}.\n"
            "Download from Google Cloud Console → APIs & Services → Credentials."
        )

    # Validate the client secrets type before starting the flow
    try:
        secrets_data = json.loads(secrets_path.read_text())
        client_type = next(iter(secrets_data), None)
        if client_type == "web":
            raise ValueError(
                "Your client_secrets.json was created as a 'Web application' OAuth client.\n"
                "This causes 'redirect_uri_mismatch' errors.\n\n"
                "Fix:\n"
                "  1. Go to https://console.cloud.google.com → APIs & Services → Credentials\n"
                "  2. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'\n"
                "  3. Select application type: 'Desktop app'\n"
                "  4. Download the new JSON and replace credentials/client_secrets.json\n"
                "  5. Re-run: python3 setup.py"
            )
    except ValueError:
        raise
    except Exception:
        pass  # If we can't parse, proceed and let the flow fail with its own error

    logger.info("Starting YouTube OAuth flow…")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), config.YOUTUBE_SCOPES)
    flow.redirect_uri = "http://localhost:8080"

    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

    print("\n" + "=" * 60)
    print("YOUTUBE AUTHORIZATION REQUIRED")
    print("=" * 60)
    print("\n1. Open this URL in your browser (on any device):\n")
    print(f"   {auth_url}\n")
    print("2. Sign in and click Allow.")
    print("3. Your browser will redirect to http://localhost:8080")
    print("   and show a 'connection refused' or blank page — that's OK.")
    print("4. Copy the FULL URL from the browser address bar.")
    print("   It looks like: http://localhost:8080/?code=4/0A...&scope=...")
    print("=" * 60)

    redirected = input("\nPaste the full redirect URL here: ").strip()

    parsed = urlparse(redirected)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    if not code:
        # User may have pasted just the code
        code = redirected

    flow.fetch_token(code=code)
    creds = flow.credentials

    token_path.write_text(creds.to_json())
    logger.info("YouTube credentials saved to %s", token_path)
    return creds


def build_youtube_client():
    return build("youtube", "v3", credentials=get_youtube_credentials())


def build_analytics_client():
    return build("youtubeAnalytics", "v2", credentials=get_youtube_credentials())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Completing YouTube OAuth flow…")
    creds = get_youtube_credentials()
    print(f"Success! Token saved to {config.YOUTUBE_TOKEN_FILE}")
    print(f"Token valid: {creds.valid}, has refresh: {bool(creds.refresh_token)}")

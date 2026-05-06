"""Shared HTTP session for Hugging Face Inference Router calls.

Wraps `requests` with `urllib3.util.retry.Retry` so transient DNS failures
(common on Railway during cold-start egress provisioning) and HF gateway
hiccups are retried automatically with exponential backoff instead of
surfacing as request errors.
"""
import logging
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


hf_session: requests.Session = _build_session()


def warn_if_token_missing(logger: logging.Logger) -> None:
    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN is not set; Hugging Face requests will be sent unauthenticated "
            "and will likely return 401. Set HF_TOKEN in the deployment environment."
        )

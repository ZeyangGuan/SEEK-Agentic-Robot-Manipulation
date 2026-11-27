from copy import deepcopy
from urllib.parse import urlparse


RESPONSES = "responses"


def validate_api_mode(value):
    if value != RESPONSES:
        raise ValueError(f"Unsupported api_mode {value!r}; SEEK talks to the Responses API")
    return RESPONSES


def validate_endpoint(api_mode, endpoint):
    validate_api_mode(api_mode)
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("The model endpoint is empty; expected a Responses API URL")
    address = urlparse(endpoint)
    if address.scheme not in ("http", "https") or not address.netloc:
        raise ValueError("The model endpoint must be an absolute http(s) URL")
    if not address.path.rstrip("/").endswith("/responses"):
        raise ValueError("The model endpoint must end in /responses")
    return endpoint


def to_wire_payload(payload, api_mode):
    validate_api_mode(api_mode)
    return deepcopy(payload)


def from_wire_response(response, api_mode):
    validate_api_mode(api_mode)
    if not isinstance(response, dict):
        raise ValueError("Responses API returned a non-object payload")
    return response


def _tokens(value):
    return value if type(value) is int else None


def provider_cache_usage(usage):
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    total, cached = usage.get("input_tokens"), details.get("cached_tokens")
    reported = all(type(count) is int and count >= 0 for count in (total, cached))
    return dict(input_tokens=_tokens(total), cached_tokens=_tokens(cached),
                cache_write_tokens=_tokens(details.get("cache_write_tokens")),
                cache_reporting_valid=reported)

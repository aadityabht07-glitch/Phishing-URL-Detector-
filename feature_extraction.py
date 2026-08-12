"""
feature_extraction.py
-----------------------
Extracts the same 16 lexical/structural features used to train the
model, computed directly from a raw URL string. No network calls are
made — this is pure string parsing, which is what makes the live demo
reliable (Section 10/22 of the project spec).

WHY THIS FILE EXISTS SEPARATELY FROM predict.py:
Training used a CSV that already had these 16 features pre-computed.
But at prediction time, the user types a raw URL string, not a feature
vector — so we need this module to recompute those same 16 numbers
on the fly, in the exact same way, or the model receives inputs it
was never trained to understand.

PROVENANCE NOTE: the original dataset's feature-extraction code was not
distributed with it. Each formula below was reverse-engineered by
testing candidate formulas against ~100,000 labeled rows in the
training set and keeping the version with the highest exact-match
rate. Validated at 97.5% full-row exact match overall; individual
features range 98.4-100% match. This is treated as reliable, and is
disclosed as a limitation in the README.
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse

FEATURE_COLUMNS = [
    "url_length", "has_ip_address", "dot_count", "https_flag",
    "url_entropy", "token_count", "subdomain_count", "query_param_count",
    "tld_length", "path_length", "has_hyphen_in_domain", "number_of_digits",
    "tld_popularity", "suspicious_file_extension", "domain_name_length",
    "percentage_numeric_chars",
]

_IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_TOKEN_DELIMS = set("./?=@&")
_POPULAR_TLDS = {"com", "org", "net", "edu", "gov"}
_SUSPICIOUS_EXTENSIONS = (
    ".exe", ".zip", ".scr", ".bat", ".js", ".dll",
    ".apk", ".jar", ".msi", ".vbs", ".cmd", ".bin",
)


def _ensure_scheme(url: str) -> str:
    """urlparse needs a scheme to correctly split netloc from path."""
    """
    urlparse needs a scheme to correctly split netloc from path.
 
    When the user doesn't type a scheme (e.g. "google.com" instead of
    "https://google.com"), we have to guess one. We default to HTTPS
    rather than HTTP, since the large majority of real-world domains
    serve HTTPS by default in 2026 and browsers auto-upgrade bare
    domains to HTTPS. Defaulting to HTTP would manufacture a false
    "insecure" signal for ordinary, legitimate bare-domain input.
    """
    if "://" not in url:
        return "https://" + url
    return url


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> dict:
    """
    Compute the 16-feature vector for a single raw URL string.
    Returns a dict keyed by FEATURE_COLUMNS, in the exact order the
    model expects.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    parsed_url = _ensure_scheme(url)
    parsed = urlparse(parsed_url)
    netloc = parsed.netloc  # includes port if present, matching training data
    path = parsed.path

    domain_parts = netloc.split(".") if netloc else [""]

    url_length = len(url)
    number_of_digits = sum(c.isdigit() for c in url)

    features = {
        "url_length": url_length,
        "has_ip_address": 1 if _IP_PATTERN.match(netloc.split(":")[0]) else 0,
        "dot_count": url.count("."),
        "https_flag": 1 if parsed.scheme == "https" else 0,
        "url_entropy": _shannon_entropy(url),
        "token_count": sum(1 for c in url if c in _TOKEN_DELIMS) + 1,
        "subdomain_count": max(len(domain_parts) - 2, 0),
        "query_param_count": (
            parsed.query.count("&") + 1 if "?" in url else 1
        ),
        "tld_length": len(domain_parts[-1]) if domain_parts else 0,
        "path_length": len(path),
        "has_hyphen_in_domain": 1 if "-" in netloc else 0,
        "number_of_digits": number_of_digits,
        "tld_popularity": 1 if domain_parts[-1].lower() in _POPULAR_TLDS else 0,
        "suspicious_file_extension": 1 if url.lower().split("?")[0].endswith(
            _SUSPICIOUS_EXTENSIONS
        ) else 0,
        "domain_name_length": len(domain_parts[-2]) if len(domain_parts) >= 2 else 0,
        "percentage_numeric_chars": (
            (number_of_digits / url_length) * 100 if url_length else 0.0
        ),
    }

    return {col: features[col] for col in FEATURE_COLUMNS}

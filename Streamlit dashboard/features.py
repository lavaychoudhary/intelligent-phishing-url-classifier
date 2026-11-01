import re, math
from urllib.parse import urlparse

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "bank",
                    "signin", "confirm", "password", "paypal", "webscr"]

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_digits", "digit_ratio",
    "has_at", "is_ip", "path_depth", "num_subdomains", "num_params",
    "suspicious_words", "entropy", "has_double_slash_in_path",
    "tld_length", "num_special_chars",
]

def _entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))

def normalize(url):
    """Remove scheme and leading www so the model can't exploit dataset bias."""
    url = str(url).strip()
    url = re.sub(r"^[a-zA-Z]+://", "", url)
    url = re.sub(r"^www\d*\.", "", url, flags=re.I)
    return url

def extract_features(url):
    u = normalize(url)
    try:
        p = urlparse("http://" + u)
        host, path, q = p.hostname or "", p.path or "", p.query or ""
    except ValueError:  # malformed URLs (e.g. bad IPv6 brackets) exist in this data
        m = re.match(r"([^/?#]*)([^?#]*)\??([^#]*)", u)
        host, path, q = m.group(1).lower(), m.group(2), m.group(3)

    parts = host.split(".")
    is_ip = int(bool(IP_PATTERN.match(host)))
    digits = sum(c.isdigit() for c in u)
    low = u.lower()

    return [
        len(u), len(host), len(path), len(q),
        u.count("."), u.count("-"), u.count("_"), digits,
        digits / max(len(u), 1),
        int("@" in u), is_ip, path.count("/"),
        0 if is_ip else max(len(parts) - 2, 0),
        len(q.split("&")) if q else 0,
        sum(w in low for w in SUSPICIOUS_WORDS),
        _entropy(u), int("//" in path),
        0 if is_ip else len(parts[-1]),
        len(re.findall(r"[^a-zA-Z0-9]", u)),
    ]

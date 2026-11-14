import re, math
import tldextract
from urllib.parse import urlparse

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "bank",
                    "signin", "confirm", "password", "paypal", "webscr"]

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_digits", "digit_ratio",
    "has_at", "is_ip", "path_depth", "num_subdomains", "num_params",
    "suspicious_words", "entropy", "has_double_slash_in_path",
    "tld_length", "num_special_chars", "is_safe_tld"
]

def _entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))

def normalize(url):
    url = str(url).strip()
    url = re.sub(r"^[a-zA-Z]+://", "", url)
    url = re.sub(r"^www\d*\.", "", url, flags=re.I)
    return url

def extract_features(url):
    u = normalize(url)
    
    # Use tldextract for highly accurate domain parsing (fixes .edu.in issue)
    ext = tldextract.extract(u)
    is_ip = int(bool(IP_PATTERN.match(ext.domain)))
    
    try:
        p = urlparse("http://" + u)
        host, path, q = p.hostname or "", p.path or "", p.query or ""
    except ValueError: 
        m = re.match(r"([^/?#]*)([^?#]*)\??([^#]*)", u)
        host, path, q = m.group(1).lower(), m.group(2), m.group(3)

    digits = sum(c.isdigit() for c in u)
    low = u.lower()

    # Calculate true subdomains (e.g. 'login.manavrachna.edu.in' -> 1 subdomain)
    # If there is no subdomain text, it is 0.
    num_subs = len([s for s in ext.subdomain.split('.') if s]) if ext.subdomain and not is_ip else 0

    # Identify trusted institutional TLDs
    safe_tlds = {"edu.in", "ac.in", "gov.in", "nic.in", "edu", "gov"}
    is_safe_tld = int(ext.suffix in safe_tlds)

    # Normalize TLD length so .edu.in is not penalized for being long
    tld_len = len(ext.suffix) if not is_ip else 0
    if ext.suffix.endswith(".in") and tld_len > 3:
        tld_len -= 3

    return [
        len(u), len(host), len(path), len(q),
        u.count("."), u.count("-"), u.count("_"), digits,
        digits / max(len(u), 1),
        int("@" in u), is_ip, path.count("/"),
        num_subs,
        len(q.split("&")) if q else 0,
        sum(w in low for w in SUSPICIOUS_WORDS),
        _entropy(u), int("//" in path),
        tld_len, 
        len(re.findall(r"[^a-zA-Z0-9]", u)),
        is_safe_tld
    ]
"""Free Wikipedia API (no key needed). Used for evidence when the user gives no reference text."""
import json
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "AICES-student-project/1.0 (educational)"}


def fetch_evidence(query, max_pages=2, timeout=6):
    """Return (knowledge_text, [page titles]), or ("", []) if nothing is found or there is no internet."""
    if not query or not query.strip():
        return "", []
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query[:250],
        "gsrlimit": max_pages, "prop": "extracts", "exintro": 1, "explaintext": 1,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return "", []
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    texts = [p.get("extract", "") for p in pages if p.get("extract")]
    titles = [p.get("title", "") for p in pages if p.get("extract")]
    return " ".join(texts)[:4000], titles

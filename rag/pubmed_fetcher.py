"""Fetch and cache PubMed abstracts via BioPython Entrez."""

from __future__ import annotations
import json
import os
import time
from typing import List, Dict

try:  # pragma: no cover
    from Bio import Entrez  # type: ignore
except Exception:  # pragma: no cover
    Entrez = None  # type: ignore

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL_SECONDS = 86400  # 24 h
ENTREZ_TIMEOUT_SECONDS = 10

if Entrez is not None:
    Entrez.email = os.getenv("ENTREZ_EMAIL", "researcher@fem-net.ai")


def fetch_pubmed_abstracts(condition: str, max_results: int = 20) -> List[Dict]:
    """Return list of {title, abstract, pmid} dicts. Caches for 24 h."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{condition}_pubmed.json")

    if _cache_valid(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    if Entrez is None:
        # Dependency missing: treat as no evidence available.
        with open(cache_path, "w") as f:
            json.dump([], f, indent=2)
        return []

    query = f"{condition} early detection diagnosis women"
    try:
        docs = _fetch(query, max_results)
    except Exception as exc:
        print(f"[PubMed] fetch failed: {exc}. Using empty list.")
        docs = []

    with open(cache_path, "w") as f:
        json.dump(docs, f, indent=2)

    return docs


def _cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < CACHE_TTL_SECONDS


def _fetch(query: str, max_results: int) -> List[Dict]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, timeout=ENTREZ_TIMEOUT_SECONDS)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    if not ids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(ids),
        rettype="abstract",
        retmode="xml",
        timeout=ENTREZ_TIMEOUT_SECONDS,
    )
    records = Entrez.read(handle)
    handle.close()

    docs = []
    for article in records.get("PubmedArticle", []):
        try:
            med = article["MedlineCitation"]
            art = med["Article"]
            title = str(art.get("ArticleTitle", ""))
            abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(t) for t in abstract_texts) if abstract_texts else ""
            pmid = str(med["PMID"])
            if abstract:
                docs.append({"title": title, "abstract": abstract, "pmid": pmid})
        except Exception:
            continue

    return docs

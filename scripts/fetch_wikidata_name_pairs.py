"""
Fase 0 - real-data grounding: pull (Persian name, Latin/English name) pairs for
real Iranian people from Wikidata via SPARQL.

These are genuine transliteration pairs (e.g. "رضا پهلوی" / "Reza Pahlavi") used
to validate the username<->full-name transliteration matcher in Phase 3 against
something other than our own synthetic generator.

Output: data/real_validation/name_pairs.csv (columns: qid, fa_label, en_label)
"""
import csv
import json
import os
import subprocess
import sys
import time

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT DISTINCT ?person ?faLabel ?enLabel WHERE {
  ?person wdt:P31 wd:Q5 .        # instance of: human
  ?person wdt:P27 wd:Q794 .      # country of citizenship: Iran
  ?person rdfs:label ?faLabel .
  FILTER(LANG(?faLabel) = "fa")
  ?person rdfs:label ?enLabel .
  FILTER(LANG(?enLabel) = "en")
}
LIMIT 8000
"""

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "real_validation", "name_pairs.csv"
)


def fetch(max_retries: int = 3) -> list[dict]:
    # Note: python http clients (httpx/requests) get a 403 from Wikidata's
    # query service (likely TLS-fingerprint based bot mitigation on their
    # free public endpoint), while plain curl is accepted. Shelling out to
    # curl is a pragmatic workaround for this public, unauthenticated,
    # read-only SPARQL endpoint.
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-f",
                    SPARQL_ENDPOINT,
                    "--data-urlencode", f"query={QUERY}",
                    "--data-urlencode", "format=json",
                    "-H", "User-Agent: Mozilla/5.0 IdentityGraph-FeasibilityResearch/1.0",
                    "-H", "Accept: application/sparql-results+json",
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
            data = json.loads(result.stdout)
            rows = []
            for binding in data["results"]["bindings"]:
                qid = binding["person"]["value"].rsplit("/", 1)[-1]
                fa_label = binding["faLabel"]["value"]
                en_label = binding["enLabel"]["value"]
                rows.append({"qid": qid, "fa_label": fa_label, "en_label": en_label})
            return rows
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"Attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch Wikidata results after {max_retries} attempts: {last_err}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Querying Wikidata for Iranian person name pairs (fa/en labels)...")
    rows = fetch()
    print(f"Fetched {len(rows)} raw rows.")

    # Drop rows where fa_label / en_label are identical after normalizing case
    # (these are cases where Wikidata has no real Persian label and just
    # duplicated the English one), and drop rows where en_label contains
    # Persian script (means the "en" label wasn't actually transliterated).
    def is_latin(s: str) -> bool:
        return all((ord(c) < 128 or c.isspace() or not c.isalpha()) for c in s)

    cleaned = []
    seen = set()
    for r in rows:
        if r["fa_label"].strip() == r["en_label"].strip():
            continue
        if not is_latin(r["en_label"]):
            continue
        key = (r["fa_label"], r["en_label"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)

    print(f"Kept {len(cleaned)} cleaned fa/en pairs after de-duplication and filtering.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "fa_label", "en_label"])
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Saved to {OUTPUT_PATH}")
    print("\nSample rows:")
    for r in cleaned[:10]:
        print(f"  {r['fa_label']!r:30s} <-> {r['en_label']!r}")


if __name__ == "__main__":
    main()

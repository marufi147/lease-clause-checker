import os, re, json, joblib, pdfplumber
from sentence_transformers import SentenceTransformer

os.chdir(os.path.expanduser("~/lease-checker"))
encoder = SentenceTransformer("all-MiniLM-L6-v2")
emb_model = joblib.load("lease_emb_model.joblib")
RULES = json.load(open("lease_rules.json"))

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)

def split_clauses(text):
    pattern = r'\n(?=\s*(?:\d{1,2}[\.\)]\s|Section\s+\d|SECTION\s+\d|ARTICLE\s+\w))'
    parts = re.split(pattern, text)
    if len(parts) < 3:
        parts = re.split(r'\n\s*\n', text)
    return [c for c in (" ".join(p.split()) for p in parts) if len(c) > 40]

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

PREFIX = r"^\s*(?:\d{1,2}(?:\.\d+)*[\.\)]?\s+)?(?:[A-Z][A-Z\s/&'\-]{2,}[:\.]\s*)?"

def strip_prefix(t):
    return re.sub(PREFIX, '', t).strip() or t

def check_clause(text, threshold=0.5):
    p = emb_model.predict_proba(encoder.encode([strip_prefix(text)], normalize_embeddings=True))[0, 1]
    reasons = [(n, e) for n, pat, e in RULES if re.search(pat, text.lower())]
    return (p >= threshold or len(reasons) > 0), round(float(p), 2), reasons

def report(parts, title):
    print("=" * 80)
    print(f"REPORT: {title}   |   Clauses checked: {len(parts)}")
    print("=" * 80)
    flagged = 0
    for c in parts:
        risky, p, reasons = check_clause(c)
        if risky:
            flagged += 1
            print(f'\n⚠ #{flagged} (model {p})  "{c[:300]}"')
            for n, e in reasons:
                print(f"   → {n}: {e}")
            if not reasons:
                print("   → The meaning is similar to unfair clauses. Read it carefully.")
    print(f"\nFlagged {flagged} of {len(parts)}." if flagged else "\n✓ No risky clauses found.")
    print("Note: This gives warnings, not legal advice.")

def check_document(path, doc_type="lease"):
    parts = split_clauses(read_pdf(path))
    if doc_type == "notice":
        parts = [s for p in parts for s in split_sentences(p) if len(s) > 40]
    report(parts, os.path.basename(path))

def check_text(text, title="Pasted text"):
    report(split_clauses("\n" + text), title)

print("Lease checker ready (embeddings + rules).")

import re, json, joblib
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

encoder    = SentenceTransformer("all-MiniLM-L6-v2")
risk_model = joblib.load(HERE / "lease_emb_model_v2.joblib")
type_model = joblib.load(HERE / "lease_type_model_v2.joblib")
RULES      = json.load(open(HERE / "lease_rules.json"))

TYPE_LABEL = {"rent": "Rent", "deposit": "Deposit", "entry": "Landlord entry", "repairs": "Repairs",
              "termination": "Ending the lease", "fees": "Fees & penalties",
              "liability": "Liability & rights", "pets_guests": "Pets & guests"}

def read_file(path):
    path = str(path)
    if path.lower().endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)

def split_clauses(text):
    # split before "1. ", "2) ", "Section 3", "ARTICLE IV" at a newline OR mid-line after a space
    pattern = r'(?:\n|(?<=[.;:!?])\s)(?=\s*(?:\d{1,2}[\.\)]\s+[A-Z]|Section\s+\d|SECTION\s+\d|ARTICLE\s+\w))'
    parts = re.split(pattern, text)
    if len(parts) < 3:
        parts = re.split(r'\n\s*\n', text)
    return [c for c in (" ".join(p.split()) for p in parts) if len(c.split()) >= 4]
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

PREFIX = r"^\s*(?:\d{1,2}(?:\.\d+)*[\.\)]?\s+)?(?:[A-Z][A-Z\s/&'\-]{2,}[:\.]\s*)?"
def strip_prefix(t):
    return re.sub(PREFIX, '', t).strip() or t

def check_clause(text, threshold=0.5):
    x = encoder.encode([strip_prefix(text)], normalize_embeddings=True)
    p = float(risk_model.predict_proba(x)[0, 1])
    ctype = TYPE_LABEL.get(type_model.predict(x)[0], "Other")
    reasons = [(n, e) for n, pat, e in RULES if re.search(pat, text.lower())]
    return (p >= threshold or bool(reasons)), round(p, 2), ctype, reasons

def report(parts, title):
    print("=" * 80)
    print(f"REPORT: {title}   |   Clauses checked: {len(parts)}")
    print("=" * 80)
    flagged = 0
    for c in parts:
        risky, p, ctype, reasons = check_clause(c)
        if risky:
            flagged += 1
            print(f'\n⚠ #{flagged} [{ctype}] (model {p})  "{c[:300]}"')
            for n, e in reasons:
                print(f"   → {n}: {e}")
            if not reasons:
                print("   → The meaning is similar to unfair clauses. Read it carefully.")
    print(f"\nFlagged {flagged} of {len(parts)}." if flagged else "\n✓ No risky clauses found.")
    print("Note: This gives warnings, not legal advice.")

def check_document(path, doc_type="lease"):
    parts = split_clauses(read_file(path))
    if doc_type == "notice":
        parts = [s for p in parts for s in split_sentences(p) if len(s) > 40]
    report(parts, Path(path).name)

def check_text(text, title="Pasted text"):
    report(split_clauses("\n" + text), title)

print("Lease checker v2 ready (embeddings + 4 datasets + rules).")

# ---------- Notice completeness check ----------
NOTICE_CHECKLIST = {
    "reason for eviction":        r"\b(because|reason|due to|breach|violat\w*|non-?payment|unpaid)\b",
    "amount owed (if rent)":      r"\$\s?\d",
    "specific deadline / date":   r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december|within \d+ days)\b",
    "mention of court / process": r"\b(court|tribunal|hearing|legal proceedings|eviction proceedings|judge|state law)\b",
    "tenant's right to respond":  r"\b(right to (respond|contest|dispute|cure)|you may (pay|dispute|contest)|legal (aid|advice))\b",
}

def notice_completeness(text):
    print("\nCompleteness check (what a fair notice usually contains):")
    for item, pat in NOTICE_CHECKLIST.items():
        ok = re.search(pat, text, flags=re.I)
        print(f"   {'✓' if ok else '✗ missing:'} {item}")

def check_document(path, doc_type="lease"):
    text = read_file(path)
    parts = split_clauses(text)
    if doc_type == "notice":
        parts = [s for p in parts for s in split_sentences(p) if len(s.split()) >= 4]
    report(parts, Path(path).name)
    if doc_type == "notice":
        notice_completeness(text)

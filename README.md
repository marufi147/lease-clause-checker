# Plain-Language Lease & Eviction Notice Checker

A text mining project that reads rental agreements and eviction notices, flags clauses that may be unfair to tenants, and explains why in plain language.

## How it works
1. **Extract** text from PDF or Word files (pdfplumber, python-docx)
2. **Split** documents into clauses (numbered sections or sentences)
3. **Classify** each clause as normal or risky using sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression
4. **Rules**: 11 warning-phrase patterns (e.g. entry without notice, non-refundable deposit, waiver of rights), each with a plain-language explanation
5. A clause is flagged if the model **or** a rule marks it risky

## Dataset
- 131 normal clauses: from public sample lease and notice templates, plus hand-written plain-style clauses
- 32 risky clauses: hand-written examples of common unfair lease terms

## Results
| Method | Test | Score |
|---|---|---|
| TF-IDF + Logistic Regression | 5-fold CV | F1 (risky) 0.65 |
| TF-IDF + rules | Unseen test set 1 | 60% accuracy |
| Embeddings + rules | Unseen test set 1 (used for tuning) | 80% accuracy |
| **Embeddings + rules** | **Fresh final test set** | **65% accuracy** |

## Key findings
- **Shortcut learning:** the model first learned writing style (short sentences, clause numbers) instead of meaning; fixed by balancing styles and stripping numbering.
- Keyword rules give clear explanations but miss paraphrases.
- With only 32 risky examples, the system catches clauses similar to known ones but misses new types of unfairness.

## Usage
```python
%run lease_checker.py
check_document("my_lease.pdf", "lease")      # PDF
check_text("1. Landlord may enter at any time.")   # pasted text
```

## Limitations & future work
Small dataset, hand-written risky examples, weak on numbers (e.g. rent increases). Future: real unfair clauses from tenant-rights cases, fine-tuned Legal-BERT.

**Disclaimer:** This tool gives warnings, not legal advice.

## Author
Bhuyan Maruf · [GitHub](https://github.com/marufart)

# Plain-Language Lease & Eviction Notice Checker

A text mining project that reads rental agreements and eviction notices, flags clauses that may be unfair to tenants, labels each clause by topic, and explains why in plain language.

## How it works
1. **Extract** text from PDF or Word files (pdfplumber, python-docx)
2. **Split** documents into clauses (numbered sections or sentences)
3. **Classify risk**: sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression
4. **Classify clause type**: rent, deposit, entry, repairs, termination, fees, liability, pets/guests
5. **Rules**: 11 warning-phrase patterns, each with a plain-language explanation
6. A clause is flagged if the model **or** a rule marks it risky
7. **Eviction notices** also get a completeness check (reason, amount, deadline, court process, right to respond)

## Datasets
| Dataset | What it contains | How it is used |
|---|---|---|
| Own data | 163 lease/notice clauses (131 normal, 32 risky) + 64 seed clauses with types | core training data (weighted ×10) |
| [UNFAIR-ToS](https://huggingface.co/datasets/coastalcph/lex_glue) (LexGLUE) | consumer terms of service; experts marked unfair clauses | **only source of real fair/unfair labels**: 641 risky training clauses (liability, termination, unilateral change, arbitration, jurisdiction) |
| [Lease Contract Review](https://gitlab.com/spyretta.leiv/lease_contract_review) | 23,450 sentences from commercial leases with "red flag" topic tags | clause types + 500 real-lease normal sentences + a 2,000-sentence **false-alarm test** |
| [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (LexGLUE) | 80,000 contract provisions, 100 labels | clause types (fees, liability, termination) + boilerplate as normal examples |
| [CUAD](https://github.com/TheAtticusProject/cuad) | 510 contracts, 41 clause categories | clause types (fees, liability, termination) |

Lease Contract Review, LEDGAR and CUAD label **topics**, not fairness (e.g. "landlord_repairs" also tags fair repair clauses), so they are not used as risky examples.

**Final training data:** risk model 673 risky / 2,437 normal (before: 32 / 131); clause-type model ~2,000 examples.

## Results: risk detection (full system = model OR rules)
| Test set | n | Old model (own data only) | New v1 (unweighted) | **New v2 (final)** |
|---|---|---|---|---|
| Fresh final lease test (accuracy) | 20 | **0.65** | 0.55 | 0.55 |
| UNFAIR-ToS test (F1, risky) | 368 | 0.30 | 0.74 | **0.71** |
| Real-lease false-alarm rate (lower is better) | 1,999 | 8% | 15% | **7%** |
| LEDGAR boilerplate false-alarm rate | 89 | 1% | 1% | 2% |

v2 settings (own clauses weighted ×10, +500 real-lease normal sentences) were chosen by 5-fold cross-validation on own data and a separate false-alarm validation set, **not** on the final test.

## Results: clause type (accuracy)
| Test set | n | Seed only | **All 4 datasets + seed ×10** |
|---|---|---|---|
| Lease test | 16 | 0.94 | **1.00** |
| CUAD test (unseen contracts) | 142 | 0.74 | **0.89** |
| LEDGAR test | 103 | 0.69 | **0.91** |

## Key findings
- **Domain shift:** 641 terms-of-service unfair clauses raised UNFAIR-ToS F1 from 0.30 to 0.71, but did not help on lease-only risks. All 9 errors on the final lease test are risk types no dataset covers (tenant-association bans, seizing belongings, rent +20% every 3 months, fees per maintenance request).
- **Drowning:** without weighting own clauses, false alarms on real leases doubled (8% → 15%). Weighting ×10 and adding real-lease normal sentences brought it back to 7%.
- **Over-correction on termination:** the model flags fair eviction language (e.g. "formal eviction proceedings in accordance with state law", score 0.50–0.54), because terms of service taught it that termination wording is often unfair.
- **Shortcut learning:** the first model learned writing style instead of meaning; fixed by balancing styles and stripping numbering. UNFAIR-ToS adds the same risk (all lowercase/tokenized), handled by detokenizing.
- **Most legal datasets don't label fairness:** only 1 of 4 labels whether a clause is fair to the weaker party.

## Usage
```python
%run lease_checker_v2.py
check_document("my_lease.pdf", "lease")        # PDF or .docx
check_document("notice.pdf", "notice")         # eviction notice + completeness check
check_text("1. Landlord may enter at any time.")

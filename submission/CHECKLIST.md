# Elsevier Pattern Recognition Submission Checklist

| Check / Requirement | Status | Verification Detail |
| :--- | :---: | :--- |
| **Manuscript File Format** | `VERIFIED` | LaTeX source using official `elsarticle` document class (`elsarticle.cls`). |
| **Compiled PDF** | `VERIFIED` | [`main.pdf`](file:///d:/nullconsensus/submission/main.pdf) compiled cleanly with zero errors via Tectonic (322 KB). |
| **Cover Letter** | `VERIFIED` | [`COVER_LETTER.md`](file:///d:/nullconsensus/submission/COVER_LETTER.md) addressed to Editor-in-Chief. |
| **Highlights** | `VERIFIED` | [`HIGHLIGHTS.md`](file:///d:/nullconsensus/submission/HIGHLIGHTS.md) formatted with 5 bullet points ($\le 85$ characters each). |
| **Keywords** | `VERIFIED` | Formatted with standard `\sep` delimiter under the `keyword` environment in `main.tex`. |
| **CRediT Authorship Statement** | `VERIFIED` | Included as formal section in `main.tex` following standard contributor taxonomies. |
| **Competing Interests Declaration** | `VERIFIED` | Formal *Declaration of Competing Interest* included in `main.tex`. |
| **Data and Code Availability** | `VERIFIED` | Public GitHub repository, Zenodo archive intent, and `data/test_manifest.json` cited. |
| **Vector Figures** | `VERIFIED` | 5 high-resolution vector figures in `paper/figures/` (both PDF and PNG formats, 300 DPI). |
| **Tables & Single Source of Truth** | `VERIFIED` | 5 publication tables in `paper/tables/` populated dynamically from `numbers.tex` and `results/test_benchmark.csv`. |
| **Pre-Registration Integrity** | `VERIFIED` | Protocol v2.0.0 in `PREREGISTRATION.md` (commit `ac0fcd3`), freeze tag `freeze-v1`, zero test data leakage. |
| **Unit Test Battery** | `VERIFIED` | All 32 unit tests passing cleanly (`pytest tests/`, 100% pass rate). |

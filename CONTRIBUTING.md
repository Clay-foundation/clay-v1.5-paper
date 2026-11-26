# Contributing and giving feedback

This repository accompanies the Clay v1.5 technical report. Public review is welcome. The most
useful feedback is specific and checkable, so please open **one issue per claim**.

## How to report

| You found | Use | Include |
|---|---|---|
| A wrong number, claim, citation, table cell or figure | Issue template "Error in the paper" | Location (section, table, line in `DRAFT.md`), the exact text, what you believe is correct and the evidence (URL, DOI, file, command output) |
| A script or result that does not reproduce | Issue template "Reproduction failure" | Commit hash, command run, hardware, `pip freeze` or lockfile, the full log, the number you got vs. the number in the repo |
| A licensing or attribution problem | Issue template "Error in the paper", label `licensing` | The artifact, the licence you read, and the source URL |
| A general question or idea | GitHub Discussions if enabled, otherwise a plain issue | |

Security or confidentiality concerns (for example, an accidentally committed secret): do not open a
public issue. Email the corresponding author listed in the manuscript.

## What happens next

An author triages each issue. Corrections to the manuscript are made in `DRAFT.md` and mirrored in
`latex/`, and `python3 build.py` checks that the two agree. Numbers in the paper come from
`experiments/results/*.json`; a change to a number must come with the script run that produced it.

## Pull requests

Small, focused pull requests are welcome: typo fixes, broken links, clearer scripts, environment
pins. Please do not reformat unrelated files. By submitting a pull request you agree that your
contribution is licensed under the terms in [`NOTICE`](NOTICE): Apache-2.0 for code and
CC BY 4.0 for text, figures and results.

## Licensing at a glance

Code: Apache-2.0 (`LICENSE`). Manuscript text, figures, results: CC BY 4.0 (`LICENSE-docs`).
Third-party datasets and model weights keep their own licences: see
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md).

## Conduct

Be courteous and critique the work, not the people. Maintainers may close or edit comments that are
abusive or off topic.

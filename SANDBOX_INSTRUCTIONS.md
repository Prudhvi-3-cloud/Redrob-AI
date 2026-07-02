# Sandbox Instructions

The competition asks for a hosted sandbox/demo link. Do not submit a fabricated
URL. Create the sandbox first, confirm it runs, then paste that public link into
the portal and `submission_metadata.yaml`.

## Recommended Colab Flow

1. Upload `sandbox_demo.ipynb` to Google Colab.
2. Upload the project to a reachable location:
   - a public or organizer-accessible GitHub repository, or
   - a zip file in Google Drive.
3. In the first notebook cell, set one of:
   - `REPO_URL = "https://github.com/<user>/<repo>.git"`
   - or keep `REPO_URL = "TODO"` and upload/extract the repository manually.
4. Run all notebook cells.
5. Confirm the notebook:
   - installs `requirements.txt`
   - finds `rank.py`, `artifacts/`, and `models/`
   - runs `rank.py`
   - runs `patch_reasoning.py`
   - runs `validate_submission.py`
   - prints a small preview of `submission.csv`
6. In Colab, use `Share` and set access according to the competition portal
   requirement.
7. Paste the Colab share link into:
   - the portal sandbox field
   - `submission_metadata.yaml` under `sandbox_link`

## Runtime Notes

The notebook runs the same release command with `--cross-encoder-limit 30`.
That is the submitted reproduction mode. If you only need a quick smoke test
while preparing the sandbox, temporarily change the limit to `8`; change it
back to `30` before final validation.

## Files Required In The Sandbox

The sandbox needs these files and directories:

```text
rank.py
patch_reasoning.py
validate_submission.py
requirements.txt
candidates.jsonl
job_description.docx
artifacts/
models/
```

`precompute.py` is included for completeness, but the sandbox does not need to
rerun precomputation when the release artifacts are present.

## Expected Success Behavior

`rank.py` prints progress and writes `submission.csv`.

`patch_reasoning.py` is silent on success.

`validate_submission.py` is silent on success. If it prints issues, fix those
before submitting the sandbox link.

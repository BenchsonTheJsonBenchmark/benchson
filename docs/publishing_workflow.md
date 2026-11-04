# Publishing workflow (nothing is published yet)

Rule: `aviv1ron1/Benchson` is never modified, pushed to, branched or tagged. Everything stays local until approved.
Results go to a **new** dataset repo under our own account/organization.

| Branch of the new repo | Content |
|---|---|
| `main` | pinned original baseline, byte-identical to `aviv1ron1/Benchson` @ `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6` |
| `scale-only-7k` | original 3,000 train rows (verbatim, first) + new standard examples; test byte-identical |
| `mixed-80-20` | original train rows (verbatim, first) + new 80/20 examples; test byte-identical |

## Step 1: copy the pinned original into `main` of our private repo (guided, from the MacBook)

Target: `benchson-research/benchson-data-experiments` (private, already created). Source: `aviv1ron1/Benchson` @
`7962b0947f7d96eb3a0e6b9b928d939c7f3061a6` (public, read only, no token is sent to it). All downloads go to a temporary
folder (default `$TMPDIR/benchson_hf_transfer`) that `cleanup` deletes at the end. The BGU cluster is not needed for this step.

```bash
python3 -m venv ~/.venvs/hf && source ~/.venvs/hf/bin/activate && python -m pip install --upgrade huggingface_hub
read -s HF_TOKEN && export HF_TOKEN        # fine-grained token, write access to ONLY our repo; not echoed, not stored
python src/hf_baseline.py auth-check       # read only
python src/hf_baseline.py prepare          # download + build release/main + byte-for-byte checks
python src/hf_baseline.py upload           # the only writing step; you type the repo id to confirm
python src/hf_baseline.py verify-remote    # downloads again from our repo and compares with the original
python src/hf_baseline.py cleanup          # deletes the temporary folder
unset HF_TOKEN
```

Credential scan: `verify-local` and `upload` scan every file row by row (stringified JSON fields are decoded, so each match
has a field path). A match is accepted only when it is clearly a placeholder (AWS documentation key ending in `EXAMPLE`,
private-key header without a key body, token made of <= 4 distinct characters, entropy < 3 bits/char, an alphabet/digit
sequence, or an obvious placeholder word). Random-looking values, and every kind without a placeholder rule, block the upload.
`python src/hf_baseline.py classify-credentials` groups the matches by value and field and prints only file, line, field path,
class and reason, never the values. After a code update use `python src/hf_baseline.py prepare --rebuild-release` to rebuild
`release/` from the already verified download (no network).

Reviewed exceptions (`docs/credential_reviews.json`): a flagged value can be allowed only through
`python src/hf_baseline.py record-credential-review --body-sha256 <64hex> --expect-occurrences N` (the exact body is selected by its full SHA-256, never by the header group; other bodies under the same header keep their normal classification), and only when the structure check says the value is
the beginning of an RSA key container cut off inside the public modulus (no private exponent, primes or CRT values; the
library cannot parse it). The record pins the FULL SHA-256 of the encoded body and the exact places (file, line, field path);
the same value elsewhere, a changed value, or a longer/complete key stays blocked. No general rule allows malformed or
partial private keys. Residual risk, stated in the record: it may be the first bytes of the public modulus of a real key.
The decision for value group `3021d90e` (body fingerprint `3bbf6cc3`, 96 decoded bytes, declared container length 1217 bytes) is documented by its entry in `docs/credential_reviews.json` once it has been recorded; until then the upload stays blocked.

Known limitation of the original benchmark (documented in the release README, NOTICE and PROVENANCE, data unchanged): 8
schema contents occur in both the train and the test split under different example ids (the original split was made per
schema file). `python src/hf_baseline.py diagnose` lists them; no example is identical across the splits.

What lands on `main`: the six data files (byte-identical, verified by git blob id, size, sha256 and row counts 1000/150 per
task), `README.md` (card, license from the original), `ORIGINAL_DATASET_CARD.md` (the original card, unchanged), `NOTICE.md`
(attribution) and `PROVENANCE.json`. The stray `.DS_Store` of the original is not copied. The `main` commit SHA differs from
the original SHA, so record both.

Result (2026-11-04): `benchson-research/benchson-data-experiments` `main` = commit `1e8aad270c7f0e1f82d139bbe1e32a345188b697`, copied from
`aviv1ron1/Benchson` @ `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6`; `verify-remote` passed.

## Step 2 (later): branches

Not handled by `hf_baseline.py` and not created yet. Local preparation without network:
```bash
python src/prepare_hf_release.py --variant scale-only-7k --baseline data_published/7962b0947f7d --new-dir data_scale_only/<run> --new-repo-id benchson-research/benchson-data-experiments
python src/prepare_hf_release.py --variant mixed-80-20   --baseline data_published/7962b0947f7d --new-dir data_mixed/<run>      --new-repo-id benchson-research/benchson-data-experiments
```
(`--baseline` comes from `import_published_benchmark.py`, which the generation runs on the cluster already use.)
This writes `release/<variant>/` (+ NOTICE, original card, PROVENANCE) and `release/PUBLISH_COMMANDS.md` with the branch
commands (not executed). The script refuses the original repo id, a non-empty output dir, new test examples, new examples
whose schema is in the original test split, and any change to the test split or the original train rows.
The scale-only generation, merge and duplicate checks run on the cluster against `data_published/<sha12>` (imported from
the original pinned revision, not from our repo), so they do not depend on our repo.

# Limitations

Status: stub. Written before results exist. Each item must be kept, including null results.

## What this does not show

- One model family (Qwen2.5-0.5B-Instruct) and one hardware target (Intel i7-4770HQ, CPU only).
- TODO: power limits, for example "n=100 cannot detect refusal drops under about 10 pts".
- TODO: the refusal classifier is not validated by a human yet (see `audit/`).
- TODO: no GPU, no long-context study, no imatrix sweep.

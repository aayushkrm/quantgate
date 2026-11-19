@AGENTS.md

## Claude Code notes

- For the step-by-step workflow (fetch, eval, gate, report) read `.claude/skills/quantgate/SKILL.md` first.
- Change `src/quantgate/schema.py` only with a `schema_version` decision and an entry in
  `docs/deviations.md`.
- Do not run `quantgate eval` or any model-loading code without an explicit request. Model runs take
  hours and must follow `docs/prereg.md`.

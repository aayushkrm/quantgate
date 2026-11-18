@AGENTS.md

## Claude Code notes

- For the step-by-step workflow (fetch, eval, gate, report) read `.claude/skills/quantgate/SKILL.md` first.
- `src/quantgate/schema.py` is frozen. If you think it needs a change, say so in your reply and stop there.
- Do not run `quantgate eval` or any model-loading code without an explicit request. Model runs are
  compute-queued and must follow `docs/prereg.md`.

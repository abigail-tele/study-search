Paste this as your first message in Claude Code (run from the repo root):

Read CLAUDE.md and docs/PLAN.md. You are the orchestrator; delegate using the subagents in .claude/agents/.
Start Phase 1 now. Run these in parallel:
1. grunt: repo scaffold, requirements.txt, config.py, CREDITS.md, AI_USE.md, README stub, mock JSON for all 4 endpoints.
2. grunt: data/cancer_types.json (~25 cancer types, lay + clinical terms).
3. scout: verify ClinicalTrials.gov v2 params and field names; write the notes into docs/PLAN.md §6.
4. ml-engineer: pipeline/pdq_pairs.py (start with 5 cancer types).
After scout reports, start data-engineer on db.py + indexer.py for 3 cancer types, and api-builder on the frontend against mock JSON.
Then run test-runner, then reviewer for Gate 1. Give me a ≤15-line status and a list of anything only a human can do.

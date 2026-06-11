# v13 Repetition Rule Audit: Long-Check Misdraw Fix

## Summary
- The current engine already had a `TERMINAL_PERPETUAL_CHECK_LOSS` code, but its detection was too narrow.
- Old logic only checked whether the side to move was in check at the final repeated position.
- In many real perpetual-check cycles, the repeated position occurs after the defender escapes, so the side to move is no longer in check. Those games were incorrectly adjudicated as repetition draws.
- This can poison value targets by writing `0` where the correct Xiangqi result is a loss for the checking side.

## Fix
- `xqcpp_ext_hist8_115.cpp` now keeps full position-key history and a per-move `gave_check` flag.
- On repetition terminal, it finds the last cycle between two occurrences of the current position.
- If one side made at least two moves in that cycle and all of that side's moves gave check while the other side gave no checks, the checking side loses.
- The legacy fallback remains: if the final repeated position itself is in check, the checker loses as before.
- No full long-chase rule has been shipped yet; long-chase remains audit-only because exact Xiangqi repetition law is more nuanced.

## Verified Results
- V13 refutation curriculum replay:
  - Old recorded terminals: `mate=89, nocap=22, rep=9, max=2, longcheck=1`
  - Replayed after fix: `mate=89, nocap=22, rep=2, max=2, longcheck=8`
  - 7 games previously scored as repetition draws are now correctly long-check losses for the checking side.
- All external arena replay:
  - Old recorded terminals: `mate=6710, nocap=535, rep=295, longcheck=29, max=186`
  - Replayed after fix: `mate=6710, nocap=535, rep=99, longcheck=225, max=186`
  - The audit found 25 possible long-chase-like games, but V13 refutation games had 0 conservative long-chase suspects.

## Commands
```bash
/home/laure/alphaxiang/venv_nospace/bin/python tools/repetition_rule_audit.py \
  '/home/laure/alphaxiang/v13_refutation_curriculum/**/*.json' \
  --output-json /home/laure/alphaxiang/repetition_rule_audit/v13_refutation_curriculum_audit_after_cpp_fix.json

/home/laure/alphaxiang/venv_nospace/bin/python tools/repetition_rule_audit.py \
  '/home/laure/alphaxiang/**/external_arena_*.json' \
  --output-json /home/laure/alphaxiang/repetition_rule_audit/all_external_arenas_after_cpp_fix.json
```

## Next Steps
- Rerun a small V13 arena smoke because some previously reported draws now become losses if AlphaXiang was the perpetual-checking side.
- Consider a root-level "no illegal perpetual check" guard before retraining, so MCTS does not learn to use perpetual check as an escape hatch.
- Treat full long-chase as a separate v13.1/v14 rule project; implement only after manual verification of the 25 audit suspects.

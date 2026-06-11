"""Quick loss-progression check for v12 train_log.jsonl"""
import json, sys
last_loss = None
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if d.get("kind") in ("ingest", "eval", "complete"):
        continue
    step = d.get("step")
    pl = d.get("policy_loss")
    if pl is None: continue
    n_oracle = d.get("n_oracle_samples", 0)
    sp = d.get("selfplay_ratio", 0)
    flag = ""
    if last_loss is not None and pl > last_loss * 5:
        flag = " *** JUMP ***"
    print(f"step {step:>7} policy_loss {pl:>15.4f} n_oracle {n_oracle:>4.0f} sp_ratio {sp:.3f}{flag}")
    last_loss = pl

# v12.8 Global Strategic Attention Results

Date: 2026-05-03

## Verified Baseline

- Base checkpoint: `/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt`
- v12.6-micro verified handoff anchors:
  - Pika d3: 18W-21L-11D, 47.0%
  - Pika d4: 15W-26L-9D, 39.0%
  - Pika d5 handoff estimate: about 16%

## Code Changes

- Added `use_global_strategic_attention` and `num_global_strategy_tokens` to `xiangqi_transformer_model.py`.
- The global adapter uses 6 learned strategic query tokens:
  - strategy tokens cross-attend to current 90 board tokens,
  - board tokens cross-attend back to strategy tokens,
  - zero-init output projection adds context back to board tokens.
- Added global feature hints into the strategy query:
  - material,
  - rank/file line-of-sight histogram,
  - one-screen cannon target counts,
  - king-file exposure.
- Added old-checkpoint tolerant loading for `global_strategy_*`, `line_of_sight_attention_bias`, `history_memory_*`, and relative-bias params.
- Added training CLI:
  - `--use-global-strategic-attention`
  - `--num-global-strategy-tokens`
  - `--train-only-transformer-adapters`
  - `--anchor-checkpoint`
  - `--anchor-policy-kl-weight`
  - `--anchor-value-mse-weight`
  - `--adapter-unfreeze-last-n-blocks`
- Added frozen anchor distillation:
  - legal-masked `KL(anchor_policy || current_policy)`,
  - scalar value MSE to the anchor checkpoint.

## Verification

- Python compile passed for `xiangqi_train.py` and `xiangqi_transformer_model.py`.
- Zero-init equivalence test passed earlier for global adapter: base vs global model max diff was 0.0 before training.
- Backward gradient reached `global_strategy_out`.
- One-step GPU smoke passed for:
  - global strategy anchor,
  - global strategy + last-2-block unfreeze,
  - global strategy + LOS.

## Arena Results

### v12.8D: global strategy, adapter-only, no anchor

- `global_strategy_step296500`: d3 8-9-3 = 47.5%, d4 4-14-2 = 25.0%
- `global_strategy_step297000`: d3 7-9-4 = 45.0%, d4 8-6-6 = 55.0%

Conclusion: global adapter can move d4, but it did not preserve d3.

### v12.8E: global strategy, adapter-only, v12.6 anchor

- `global_strategy_anchor_step296500`: d3 10-5-5 = 62.5%; d4 early-stopped after poor partial result, 0W-7L-3D at 10 games.
- `global_strategy_anchor_step297000_quick`:
  - d3 5-4-3 = 54.2%
  - d4 7-4-1 = 62.5%
  - d5 2-18-0 = 10.0%

Conclusion: this was the best d3/d4 signal, but d5 collapsed. Not a ship candidate.

### v12.8F: global strategy + last two Transformer blocks unfrozen

- `global_strategy_last2_step296500_quick`: d3 6-3-3 = 62.5%, d4 2-6-4 = 33.3%
- `global_strategy_last2_step297000_quick`: d3 3-4-5 = 45.8%, d4 3-7-2 = 33.3%

Conclusion: unfreezing high layers did not stabilize the global signal.

### v12.8G: global strategy + line-of-sight attention bias

- `global_los_step296500_quick`: d3 4-5-3 = 45.8%; d4 early-stopped after poor partial result, 0W-4L-2D at 6 games.
- `global_los_step297000_quick`: d3 5-6-1 = 45.8%, d4 2-5-5 = 37.5%

Conclusion: LOS did not combine constructively with global strategy in this short adapter run.

## Interpretation

- The user's strategic-attention intuition is not disproven. The best v12.8E snapshot did show simultaneous d3/d4 movement.
- The current small adapter learns a style shift faster than it learns deeper tactical robustness.
- d5 is the hard filter. Any v12.8 ship candidate must clear d5, not only d3/d4.
- Increasing global token count now is not the clean next move. The bottleneck looks more like training signal / tactical grounding than raw number of global tokens.

## Recommendation

- Do not ship v12.8D/E/F/G as-is.
- Keep v12.8E `latest_step297000.pt` as an interesting research checkpoint, not a release checkpoint:
  - `/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt`
- Next non-capacity path should be data-targeted:
  - mine v12.8E d5 losses,
  - label with Pika d12/d15 teacher-Q,
  - train the same global adapter with a d5 tactical/regret slice.
- If that still fails, move the global strategic tokens into v13 as a built-in architecture idea instead of a small warm-start adapter.

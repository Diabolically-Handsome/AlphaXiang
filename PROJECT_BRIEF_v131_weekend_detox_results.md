# V13.1 Weekend Detox + Tactical Safety Results

## Summary

- 本輪按「保留 022d step18000、不使用 027 repair、不擴容」執行。
- 數據 detox 已完成，並產生乾淨資料索引；短續訓 028a/028b/028c 均未成為 ship checkpoint。
- 真正突破來自 inference/search side：`022d step18000 + tactical mate-in-1 leaf extension + root mate1 blunder guard`。
- 最終推薦的 V13.1 形態不是新 checkpoint，而是：
  - checkpoint: `/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`
  - inference flags: `--our-tactical-mate1-extension --our-root-mate1-blunder-guard`
  - normal search: `--our-sims 8000 --our-q-weight 1.0 --our-temperature-move 0.02`

## Detox Build

新增工具：

- `tools/v131_weekend_detox_build.py`

輸出根目錄：

- `/home/laure/alphaxiang/v131_weekend_detox`

detox 結果：

| group | clean shards | clean samples | toxic shards | toxic samples | note |
|---|---:|---:|---:|---|
| normal | 38 | 58,430 | 7 | 4,467 | non-canonical oracle policy stripped, not trained |
| d4 | 2 | 2,841 | 0 | 0 | clean |
| d5 | 1 | 155 | 0 | 0 | clean |

Hygiene audit:

- `41` clean shards, `61,426` samples
- `dirty_shards = 0`
- illegal oracle entries: `0`
- illegal teacher_q entries: `0`
- missing `legal_idxs/fens/stm_is_black`: `0`
- report: `/home/laure/alphaxiang/v131_weekend_detox/hygiene_audit_clean.json`

Toxic reasons:

- `7` shards had positive `rep_draw_rate`
- `7` shards had suspicious terminal code `[3]`

Arena repetition replay audit:

- report: `/home/laure/alphaxiang/v131_weekend_detox/arena_repetition_audit.json`
- old recorded `rep=64`, replayed `rep=21`, replayed `longcheck=47`
- supports the decision to treat old repetition/longcheck states as toxic or at least not clean training signal.

## Short Continuation Arms

All arms resumed from:

- `/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`

### 028a Light

- output: `/home/laure/alphaxiang/training_runs/run_028a_v131_detox_light_from022d18000`
- d5 smoke: `0W-3L-3D / 6`, score `25.0%`
- not promoted.

### 028b Strong

- output: `/home/laure/alphaxiang/training_runs/run_028b_v131_detox_strong_from022d18000`
- d3 smoke: `2W-1L-3D / 6`, score `58.3%`
- d4 combined: `6W-6L-8D / 20`, score `50.0%`
- d5 smoke: `1W-2L-3D / 6`, score `41.7%`
- d5 expansion was stopped after terminal evidence reached `0W-6L` in the added run.
- not promoted.

### 028c Pairwise Anchor

- output: `/home/laure/alphaxiang/training_runs/run_028c_v131_pairwise_anchor_from022d18000`
- training mode:
  - `policy_loss_weight=0`
  - `value_loss_weight=0`
  - `teacher_q_loss_weight=0`
  - `teacher_q_pairwise_loss_weight=0.05`
  - `teacher_q_pairwise_use_anchor_reference=True`
  - `teacher_q_pairwise_bad_move_only=True`
  - `anchor_policy_kl_weight=3.0`
  - `anchor_value_mse_weight=1.0`
- d5 smoke: `0W-4L-2D / 6`, score `16.7%`
- not promoted.

## Tactical Safety Candidate

Candidate:

- checkpoint: `022d step18000`
- flags:
  - `--our-tactical-mate1-extension`
  - `--our-root-mate1-blunder-guard`

Results:

| anchor | games | W-L-D | score |
|---|---:|---:|---:|
| Pika d3 | 20 | 11-3-6 | 70.0% |
| Pika d4 | 20 | 9-1-10 | 70.0% |
| Pika d5 | 100 | 36-41-23 | 47.5% |
| Fairy-Stockfish d3 | 20 | 20-0-0 | 100.0% |
| ElephantArt 800 | 20 | 20-0-0 | 100.0% |

Pika d5 evidence files:

- `/home/laure/alphaxiang/v131_weekend_detox/search_safety_022d_step18000/pika_d5_mate1_guard_ext/external_arena_20260509_131525.json`
- `/home/laure/alphaxiang/v131_weekend_detox/search_safety_022d_step18000_expand20/pika_d5_mate1_guard_ext/external_arena_20260509_141148.json`
- `/home/laure/alphaxiang/v131_weekend_detox/search_safety_022d_step18000_final50/pika_d5_mate1_guard_ext/external_arena_20260509_162248.json`
- `/home/laure/alphaxiang/v131_weekend_detox/search_safety_022d_step18000_final100_add50/pika_d5_mate1_guard_ext/external_arena_20260509_203326.json`

Pika d5 symbolic guard summary:

- total root guard events across 100 games: `7`
- most improvement likely comes from `tactical_mate1_extension`, with root guard as occasional safety net.

## Interpretation

The main bottleneck was not fixed by more detox finetuning. The 028 arms show that small continuation can easily fail or become noisy.

The strongest result is that a narrow search-side tactical rule made d5 jump from the previous quick-smoke baseline:

- bare 022d Pika d5 quick smoke: `1W-5L-0D / 6`, score `16.7%`
- 022d + tactical safety Pika d5: `36W-41L-23D / 100`, score `47.5%`

This suggests V13 already has enough global policy strength, but MCTS leaf evaluation was too blind near immediate tactical terminal states. Adding mate-in-1 awareness at neural leaf eval lets search avoid many shallow tactical collapses without retraining the model.

## Recommendation

Ship candidate:

- `V13.1-safety = 022d step18000 + tactical mate1 extension + root mate1 blunder guard`

Do not ship:

- `028a`
- `028b`
- `028c`

Next work, if continuing:

1. Make the safety preset easy to enable in PVP/release scripts.
2. Add a small config name, e.g. `v13_1_safety_8000`, so tests and PVP do not rely on manually remembering flags.
3. Investigate d5 losses under safety; remaining failures likely require mate-in-2 / forcing-check extension, longcheck-aware MCTS state, or tactical loss clustering.
4. Keep detox dataset and report as research artifacts, but do not claim detox finetune itself improved strength.

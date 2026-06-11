# v13 Implementation Ready Brief

## Summary
- v13 兩臂已接入代碼：`v13_200m_dense` 與 `v13_200m_strategy`。
- 兩臂共享 200M 級別主幹、2D relative attention bias、value token pooling、WDL/value/policy 輸出接口。
- strategy arm 額外加入 trunk-native strategic tokens：序列為 `[material_token, strategy_tokens x8, board_tokens x90]`，policy head 只讀 board tokens。
- 根據 Lawrence 的建議，新增短程 v12 teacher bootstrap distillation：v13 仍從 scratch 初始化，但前 50k step 用 v12.6-micro 作行為錨點，之後線性退火到 0，避免長期被 v12 上限綁住。

## Implemented Files
- `xiangqi_transformer_model.py`
  - 新增 `policy_head_dim`，v13 policy low-rank head 使用 384。
  - 新增 `use_trunk_global_strategy_tokens`，strategy arm 將全局戰略 token 放進 Transformer trunk，全層參與 self-attention。
  - 新增 `use_value_token_pooling`，value/WDL 不只讀 material token，而是用 learned query 從 material/strategy/board 全序列池化。
  - 2D relative bias 已支持 prefix tokens；prefix 與 board 的非空間交互保持 0 bias，board-board 仍使用 2D delta。
  - 舊 checkpoint loader 允許缺少新增 v13 權重，保持 v12/v12.8 checkpoint 兼容。
- `xiangqi_train.py`
  - 新增 `--model-preset legacy|v13_200m_dense|v13_200m_strategy`。
  - 新增 `--policy-head-dim`、`--use-trunk-global-strategy-tokens`、`--use-value-token-pooling`。
  - 新增 `--teacher-checkpoint` 作為 `--anchor-checkpoint` 的語義別名。
  - 新增 `--anchor-anneal-steps`，讓 v12 teacher KL/value anchor 可線性退火。
- `tools/v13_impl_smoke.py`
  - 實例化兩個 v13 preset。
  - 跑 forward/backward。
  - 存 checkpoint 並用正式 checkpoint loader roundtrip。
- `tools/_run_v13_200m_train_arm.sh`
  - 單臂訓練入口，`V13_ARM=dense|strategy`。
  - 默認使用 v12.6-micro teacher、human bootstrap、stage2/d4/full-Pika refutation 三類 selfplay source。
- `tools/_run_v13_200m_train_arms_serial.sh`
  - 串行跑 dense baseline 再跑 strategy arm。
- `tools/_run_v13_snapshot_smoke.sh`
  - 對任意 v13 snapshot 跑 Pika d3/d4/d5 smoke panel。
- `tools/_run_v13_after_snapshot_smokes.sh`
  - 等待指定 step snapshot 出現後自動跑 smoke panel。

## Verified
- Python compile:
  - `xiangqi_transformer_model.py`
  - `xiangqi_train.py`
  - `tools/v13_impl_smoke.py`
- Shell syntax:
  - `tools/_run_v13_200m_train_arm.sh`
  - `tools/_run_v13_200m_train_arms_serial.sh`
  - `tools/_run_v13_snapshot_smoke.sh`
  - `tools/_run_v13_after_snapshot_smokes.sh`
- v13 implementation smoke on `cuda:0`:
  - dense params: `198.155M`
  - strategy params: `198.991M`
  - outputs: `policy_logits [1,8100]`, `wdl_logits [1,3]`, `value_scalar [1,1]`
  - both arms have nonzero gradients and checkpoint load roundtrip.
- `ModelOpponent` compatibility:
  - dense roundtrip checkpoint can run 1-sim startpos search.
  - strategy roundtrip checkpoint can run 1-sim startpos search.

## Training Defaults
- Base teacher:
  - `/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt`
- Human data:
  - `/home/laure/alphaxiang/human_bootstrap_data_elite_wdl`
- Selfplay dirs:
  - `/home/laure/alphaxiang/selfplay_runs_stage2_v12`
  - `/home/laure/alphaxiang/v126_day3_d4_slice`
  - `/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses`
- Selfplay sampling ratios:
  - `0.82 0.12 0.06`
- Teacher distillation:
  - `anchor_policy_kl_weight=0.35`
  - `anchor_value_mse_weight=0.03`
  - `anchor_anneal_steps=50000`
- Main training:
  - `max_steps=300000`
  - `micro_batch_size=64`
  - `grad_accum_steps=4`
  - `learning_rate=2e-4`
  - `snapshot_interval_steps=5000`

## Important Caveat
- v12 teacher distillation is intentionally short. Its role is to help v13 leave the random-policy beginner phase faster, not to define the final style.
- Promotion still depends on arena: strategy tokens must beat the dense baseline on d4 or d5 by at least `+5pp` without a meaningful d3 collapse.

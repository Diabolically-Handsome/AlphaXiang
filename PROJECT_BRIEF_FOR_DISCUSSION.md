# AlphaXiang Transformer — 现状与瓶颈

**目的**：把这份文档交给另一位 AI Agent 讨论，目标是**找出"在固定计算预算下还能往上推多少"以及"是否有跨 paradigm 的更好做法"**。

**作者背景**：用户是大二学生，单人开发，硬件 RTX 5090 + RTX 5080，从 2026-04-04 至 2026-04-30 共 ~3.5 周完成 Stage 1 + Stage 2 (v4-v11)。已到自己能想到的极限，希望第三方提新点子。

---

## 1. 项目摘要

中国象棋 (Xiangqi) AlphaZero 风格引擎：

- **架构**：12 层 Transformer，d_model=512，8 head，FFN 2048 → 约 **140M 参数**
- **输入**：115 平面 10×9 棋盘表征（8 步历史快照 + 走子方/计时面）
- **三个 head**：
  - Policy — from×to logits over 8100 moves (90 squares × 90 squares)
  - Value (scalar) — tanh-bounded win-expectation, MSE-trained
  - WDL (3-class) — Win/Draw/Loss probabilities, cross-entropy-trained
- **训练数据来源**：
  - 人类对局监督预训练 (~30 万样本)
  - Pikafish 蒸馏（随机 rollout → Pikafish 给标签）
  - 对抗 Pikafish 自我对弈（我们 MCTS vs Pikafish 在不同深度）
  - **v10 起新增**：Pikafish d=12 给每个位置 calibrated value (oracle value)
  - **v11 新增**：Pikafish d=8 multipv=5 给每个位置 calibrated policy (oracle policy) + hard-position mining
- **MCTS 推理**：自写 C++ extension，python binding；典型 sims=800/move
- **训练对手**：Pikafish (Stockfish for Xiangqi 衍生引擎，公开天梯图 ~3950 Elo)，深度 d=2 到 d=12

---

## 2. 强度演进

| 版本 | 关键策略 | Probe vs Pika d=1+n0.15 | weighted panel Elo |
|---|---|---:|---:|
| v3 (Stage 1 末) | distill + vspika 混合 | 22.7% | ~1387 (panel-rel) |
| v4 | Stage 2 起点，sims=256→800 修复 | 39.0% | ~857–1450 |
| v5 | 升 Pikafish d=4 | 64.0% | ~1500 |
| v6 | 升 d=5（出现 specialization tradeoff） | 54.0% | ~1610 |
| v7 | 混合 d=2/3/4/5 课程 (25/30/30/15) | 71.7% | ~1717 |
| v8 | d=5 占比从 15→25% | (halted) | curriculum saturation |
| v9 | 加 frozen-self self-play vs v7 peak | 58.0% | 显著倒退 |
| **v10** | **oracle Pikafish d=12 value 标注** | 76.7% | **~1879** |
| **v11** | **+ policy oracle + Pika d=12 训练对手 + hard mining** | **90.0%** | **~1990** |

**头对头矩阵（50 games, sims=800）：**
- v10 vs v7: **47-0-3 = 97% / +604 Elo**（oracle 引入的一次性巨变）
- v11 vs v10: **28-11-11 = 67% / +123 Elo**（oracle 之后的常规代际幅度）

**v11 panel 完整数据：**
| Engine (估测公开 Elo) | v10 | v11 |
|---|---:|---:|
| Pikafish d=1+n0.15 (~1600) | 82.0% | **91.0%** (Elo +402) |
| Pikafish d=3 (~2200) | 10.0% | **18.0%** (Elo −263) |
| Fairy-SF d=3 (~2100) | 34.0% | **51.0%** (Elo +7) |
| CNN best (~1500) | 88.0% | **91.0%** (decisive 95.6%) |

---

## 3. 已发现的四条 Lemma（论文级）

**Lemma 1 — Frozen-Self Degeneracy**：用 frozen 旧 peak 当 self-play opponent，且训练起点 = 该 peak 时，information gain ≈ 0。v9 这么做了，结果对 v7 头对头 11-89 大输（v9 反向退化）。

**Lemma 4 — OOD Over-Search Trap**：value head 在分布外 (OOD) 位置上的校准误差，会随 MCTS 仿真次数 B 指数级放大：
```
Δ_score(2B vs B) ∝ −γ · D_KL(opp_dist ‖ train_dist)
```
v7 vs CNN 的 sims=800/1600 上 88% → 67% (-21pp) 是直接证据。**v10 用 oracle value (Pikafish d=12 给每个位置 tanh(cp/500)) 修复了这个问题**，v10 → v11 在 CNN 上保持 91%、决胜 95.6%。

**Lemma — Mixed-Curriculum Saturation**：在课程混合的 plateau 上的模型对 mix 比例的小调整非常脆弱。v8 把 d=5 从 15% 调到 25%（看似温和），cycle 5 sanity probe 从 71.7% 直接跌穿 50% 阈值，自动 halt。

**Lemma — Transitive Elo Overestimate**：通过第三方引擎推算的 panel-implied Elo 系统性高估头对头。v7-v9 panel 算 +510，实测 +363。**但 v11-v10 反向**：panel +108，head-to-head +123。这条 Lemma 只在 OOD 严重时成立。

---

## 4. v11 三个独立 lever 都验证有效

1. **Pikafish d=12 当训练对手** — v10 已给每个位置 oracle value，所以"被强对手暴打"的 sparse-reward 问题被绕开了。模型学的是"如何走得像 Pikafish d=12"，不是"如何赢这个对手"
2. **Policy oracle distillation** — 给每个位置 Pikafish multipv=5 d=8 的 softmax(eval/τ) 当辅助 policy target；和 MCTS visit distribution 用 α=0.5 加权。Mirror v10 的 value oracle 思路
3. **Hard-position mining** — 每 cycle 后用最新模型预测每个 shard 位置的 value，跟 oracle value 对比，top-10% 不一致最大的位置在下一 cycle 加权 3x

---

## 5. 当前瓶颈（核心问题）

### 瓶颈 A — 公开 Elo 锚定差距巨大

我们的强度数字都是 **panel-relative**（基于我们自己估测的对手 Elo）。把 v10 vs ElephantEye 3.31 (depth 10) 跑了一场作绝对锚定，v10 = **0W-47L-0D**。基于 ElephantEye d=10 估测 ~2200-2500 公开 Elo，**v10 公开尺度 ~1500-1800**。v11 没单独跑（用户终止了 v11 vs ElephantEye，因为前 44 局已经 0-44），但 v11-vs-v10 +123 Elo 推算 v11 公开 ~1620-1920。

**也就是说 panel ~2000 但公开尺度可能只有 1700。** 这个 gap 的根因是参考引擎的强度估测不可靠。

### 瓶颈 B — 强 NNUE 引擎仍是天花板

| 对手 | v11 score |
|---|---:|
| Pikafish d=3 (~2200) | **18.0%** ← 历史最高，但仍输 |
| Fairy-SF d=3 (~2100) | **51.0%** ← 首次打平 |
| Pikafish d=15 (天梯图 ~3950) | (没单独测，估计 < 5%) |

强 NNUE engines 在搜索深度 ≥ 5 时基本无法被打破。这是当前架构和算力的硬天花板。

### 瓶颈 C — 已经"修过"的轴不太可能再涨

我们改过的轴（按收益排序）：
1. 修 frozen WDL head（Stage 1 早期）
2. arena temperature 1e-6 → 0.5（修 100% 重复和棋）
3. TRAIN sims 256 → 800（policy 锐度）
4. distill 比例 30% → 75%（Stage 1 sparse reward）
5. mixed Pikafish curriculum (v7)
6. **oracle value labeling (v10)** — 单次最大跃升，+604 Elo
7. **policy oracle + Pika d=12 + hard mining (v11)** — +123 Elo over v10

每条都被薅过一遍了。**这条数据质量路线大致已榨干。**

### 瓶颈 D — 固定参数下的能力上限

12L × 512d × ~140M 参数。在 Pika d=3 上 18% 这个数字，从 v6 的 8% 涨到 v11 的 18%，五个版本只挪 10pp。考虑到我们的乐观投入预算，这条增长曲线在拍平。**怀疑是 model capacity 限制**。

---

## 6. 我们已经放弃 / 跳过的路径

| 路径 | 为什么放弃 |
|---|---|
| 自我对弈 (v7-peak frozen self-play, v9) | Lemma 1 已证伪 |
| 升 d=15 训练对手 | 时间成本 1.5+ 小时/cycle，v11 用 d=12 折中已成功 |
| ElephantArt UCCI 对手 | UCCI 死锁 bug 至今没修 |
| CCZero 集成 | TF1.3 古董，无法 import |
| px0 集成 | 跟 Pikafish 同源，违反 held-out |
| Cross-game leaf batching (C++ rework) | 4-8 小时工程量，5080 利用率从 16% 涨到 70% — ROI 不够 |
| 旋风_2007C anchor (Wine + Windows EXE) | Wine + 2007 老引擎调试代价高，Wine 安装需要 sudo 密码 |

---

## 7. 还没尝试的方向（向另一位 Agent 求建议）

### 数据轴（小风险，固定参数）

- **Auxiliary heads (KataGo 风格)**：score variance head、ply-to-end head、king safety head 等。参数开销 < 1%，但能给主 trunk 更丰富的监督信号
- **Iterative oracle escalation**：当前 oracle 全程 Pikafish d=12。逐 cycle 升 d=10 → d=12 → d=15 怎么样？
- **Self-knowledge distillation**：用 sims=2000 的 MCTS visits 作为额外 policy target（让网络学习"深搜该走哪"）
- **Curriculum on positions**：当前每个 shard 位置等权。按"Pikafish 给的 cp 评分方差"分层训练？
- **Contrastive triplets**：对每个位置生成 (好招, 坏招) 对，loss 最大化它们之间的 value 差

### 架构轴（中风险，需要更换权重）

- **直接扩参数**：12L → 16-20L、d_model 512 → 640-768，~140M → ~280M。预期 +200-400 Elo。但已经撞上算力墙
- **Mixture-of-Experts**：在固定有效参数下扩 capacity？训练复杂度上升
- **Linear attention / FlashAttention**：现在用的是标准 attention，长上下文有压力？8 步历史只有 ~720 token，可能不是瓶颈
- **Dual-head value (KataGo 风格)**：除了点估，还输出 value distribution。可能修 OOD overconfidence 问题

### 范式轴（高风险，可能换路）

- **完全去掉 self-play**：纯 Pikafish-distillation supervised learning，不打自我对弈。会不会更稳？
- **RLHF / DPO 风格**：用 Pikafish 评估对一组候选 game outcomes 排序，用 ranking loss 训练
- **Diffusion-style policy**：把 policy 头换成 conditional diffusion model 去预测下一步分布
- **Tree-of-thought transformer**：encoder 看局面，decoder 自回归生成 PV (principal variation)，然后蒸馏 PV 中每个节点
- **Endgame tablebase 集成**：象棋有完整 6 子残局库，能不能像 KataGo 用 Stockfish 那样在残局段直接查表代替 value head？

### 测量轴（不直接涨 Elo 但帮决策）

- **修 ElephantEye anchor 试验**（depth 8 而非 10）— d=10 太强让我们 0% 没法精确锚定
- **跑 sims=1600 完整 panel**：验证 Lemma 4 是不是真被修了（v11 vs CNN 在 sims=1600 上是不是不再退化）
- **找一个 ~2000 公开 Elo 的引擎做精确锚定** — ElephantArt（如果 UCCI bug 修好）、棋天大圣老版本？

---

## 8. 给另一位 Agent 的具体提问

1. 在固定 12L/140M 参数下，**还有哪些 data quality lever 没被薅过**？特别想知道 KataGo / Lc0 这种现代引擎的训练流程里我们还差什么
2. 如果用户允许扩参数到 280M（v13），最值得花 GPU 时间的**单一最大风险敞口**是什么？是更深的 trunk、更大的 d_model、还是 mixture-of-experts？
3. **有没有一种方法让 panel-Elo 和公开 Elo 重新对齐**？我们怀疑 panel 用的引擎参考 Elo 估错了 200+ 点
4. 有没有听说过象棋/围棋项目里"打不过的师傅 + 偷棋谱"（v11 思路）的类似实验？我们想知道这条路在文献里有没有 prior art
5. **下一步最该做什么**？继续 v12（数据轴边际优化）/ v13（架构扩展）/ 写 paper / 还是别的方向？

---

## 9. 工程设施现状（接手时可以直接用）

代码全部在 `C:\Users\Laure\Desktop\AlphaXiang Transformer\`，训练数据在 WSL `/home/laure/alphaxiang/`：

- `xiangqi_train.py` — 训练循环。支持 oracle_value、oracle_policy（CSR 稀疏存储）、sample_weight 端到端
- `tools/oracle_value_labeler.py` — Pikafish 给每个 shard 位置标 calibrated value
- `tools/oracle_policy_labeler.py` — Pikafish multipv 给每个位置标 top-K policy（v11 新增）
- `tools/hard_position_mining.py` — 找 |oracle - predicted| top-X% 加权（v11 新增）
- `tools/stage1_driver.py` — 自动化 cycle orchestrator，支持 distill / vspika / oracle / mining / 训练 / sanity probe
- `tools/external_arena.py` — vs Pikafish/Fairy-SF/ElephantEye 评测
- `tools/transformer_vs_transformer_arena.py` — 任意两个 ckpt 头对头（v10 vs v7 / v11 vs v10 用过）
- `tools/_run_v11_full_validation.sh` — 完整 panel + 头对头脚本

PEAK 文件：
```
/home/laure/alphaxiang/PEAK_step196000_v4_probe2_score63pct.pt
/home/laure/alphaxiang/PEAK_step204000_v5_probe1_score60pct_d2.pt
/home/laure/alphaxiang/PEAK_step210000_v6_probe2_score65pct_d3.pt
/home/laure/alphaxiang/PEAK_step232500_v7_probe23_score72pct_d1.pt
/home/laure/alphaxiang/PEAK_step255000_v10_probe3_score77pct_d1.pt
/home/laure/alphaxiang/PEAK_step270000_v11_probe2_score90pct_d1.pt  ← 当前 SOTA
```

---

## 10. 感想（用户视角）

我是大二学生，单人项目，~3.5 周从 v3 推到 v11。我自认为已经把"数据质量"这条路榨到极限了。下一步要么是扩参数（v13），要么有什么我没想到的高 ROI 路线。请帮我找出第二条路 — 我相信肯定还有我没注意到的角度。

也许还有一些我现在没意识到、但对您一目了然的"基础 ML 常识级别"的事情我没做？尽管直接告诉我，我不介意被指出基础不足。

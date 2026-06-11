# V12.8 Full-Blood d20 Teacher Data Scaling

## Decision

Do not treat the current V12.8 root-retune probes as a failure of FullPika d20
distillation.  The formal d20/d20 root pool is still far too small, and the old
root-regret shards did not provide `oracle_value`, so value training could fall
back to noisy `scaled_z`.

Formal train/validation labels must be:

- Pikafish root depth `20`
- Pikafish child depth `20`
- MultiPV `6-8`
- legal-masked candidates only
- `oracle_value` coverage `100%` for Pika-root samples

d16/d18/d14/d10 labels are smoke-only.  They must not enter train or validation
sets and must not be used for route-level conclusions.

## Targets

Held-out validation first:

- broad d20 validation: `2k-5k` roots
- pressure d20 validation: `1k-2k` roots
- failure-style d20 validation: `500-1k` roots
- clean-control d20 validation: `1k-2k` roots
- split by game/source to avoid leakage

Training pool scale-up:

- Stage 1: `5k` formal d20/d20 train roots
- Stage 2: `20k` formal d20/d20 train roots
- Stage 3: `50k+` formal d20/d20 train roots

No formal training should start before the held-out validation sets exist and
the Stage 1 train pool is built.  Data-scale ablations must compare roughly
`500`, `5k`, `20k`, and `20k + pressure/failure oversampling`.

## Current Inventory

Latest inventory command:

```bash
V128_SCALE_PHASE=inventory bash tools/_run_v128_d20_root_scaleup.sh
```

Current observed state:

- formal d20/d20 unique roots: `581 / 10000`
- formal d20/d20 records, counting overlapping audits: `1580`
- provisional excluded records: `240`
- observed labeling rate: about `453 roots/hour`
- projected time to `10k`: about `20.8 hours`
- training allowed: `no`

The small-pool probes are therefore only evidence that tiny root-retune /
micro-repair failed.  They do not test full-blood d20 teacher distillation.

## Tooling

New inventory:

- `tools/v128_d20_root_data_inventory.py`

New scale-up runner:

- `tools/_run_v128_d20_root_scaleup.sh`

Important safety behavior:

- runner refuses `root_depth < 20` or `child_depth < 20`
- exported shards are marked `DO_NOT_TRAIN_until_10000_formal_roots`
- validation shards carry `pool_role=val_*` and `validation_holdout=true`
- shard converter now writes `oracle_value` and `oracle_value_coverage`
- read smoke fails unless `oracle_value` coverage is `100%`

## First Active Batch

Completed source collection:

- batch: `batch001_probeB_d6d7_6400_1000roots`
- usable source: d6 only
- path:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/d20_root_scaleup/batches/batch001_probeB_d6d7_6400_1000roots/arena/d6/external_arena_20260526_121607.json`
- config: Probe B, Pika d6, black-side, `12` openings x `2`, `6400` sims
- result: `2W-16L-6D / 24 = 20.8%`
- no gate/verifier/margin edits

The old batch was stopped before continuing into d7 because the new objective
requires held-out validation first.

Active validation batch:

- batch: `val_pressure_probeB_d6_6400_1000roots`
- role: `val_pressure`
- target: `1000` formal d20/d20 roots from the completed d6 source

## Required Metrics Before Arena

Offline metrics must include:

- Pika top-1/top-3 agreement
- candidate ranking accuracy
- mean `regret_cp`
- Q inversion rate
- bad-root repair
- normal-control regression
- mate-risk false negative
- policy entropy collapse
- source ratios
- oracle_value coverage
- teacher_q samples
- Pika-root loss
- human-anchor loss
- clean-control regression

Arena comes only after data-scale ablations pass offline validation.

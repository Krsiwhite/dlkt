# Prespecified first experiment round — 2026-10-13

Purpose: test stable semantic item parameters and train-response difficulty alignment. Original question and concept vectors remain byte-identical. No trainable question, concept, or student ID embedding is introduced in the deployed models.

## Models and contrasts

1. `original`: the local original TextAKT3PL, 128 dimensions, 2 blocks, 8 heads, original dropout settings. Its historical test-selection training loop is replaced by this independent runner.
2. `semantic_mlp`: a shared semantic projection, 128-dimensional one-layer GRU of past responses, and an MLP response head.
3. `decoupled`: the same GRU and projections, with history-only scalar ability and current-semantics-only difficulty/discrimination. `p = sigmoid(a * (theta-b))`. Difficulty is bounded to [-3,3], a > .1, and a weak mean difficulty / log-discrimination anchor is used.
4. `aligned`: exactly the `decoupled` architecture plus reliability-weighted Huber alignment to standardized training-only Rasch difficulty; fixed auxiliary coefficient .05.

Direction 1's controlled contrast is `semantic_mlp` versus `decoupled`. `original` versus either new model also changes the backbone and cannot isolate the effect of decoupling. Direction 2's controlled contrast is `decoupled` versus `aligned`. The IRT output shape and anchoring do not establish true psychological parameter recovery or complete identifiability.

## Data and selection

- Student split: fixed seed 42, 2950 training / 492 validation / 1476 test students.
- Warm: 948 questions; 826,948 training / 140,044 validation / 415,735 test responses.
- Cold: fixed item seed 20261013, disjoint 663 training / 142 validation / 143 test questions. Train on 582,037 responses; score 19,922 validation and 63,732 test queries. All cold-item responses are removed from training and every evaluation history. Each cold query sees up to 200 preceding training-item responses from its held-out student.
- Every target is scored once. Train windows have length 201, stride 201; warm evaluation stride 100, with overlapping history and non-overlapping scored targets. Cold evaluation uses one window per query. Window states reset. This is not the retained ensemble's stride-1/refit protocol.
- Select the checkpoint with maximum validation AUC; evaluate test once after selection. Record AUC, NLL, Brier, ECE (15 equal-width bins), ACC, counts, epochs, elapsed time, and artifact hashes.
- This student test partition has been used in prior project development. It is held out from each new fit/selection process, but must not be described as an untouched external benchmark. Cold item partition is newly established for this experiment round.

## First-round budget, fixed before reading its test results

- Full first comparison: all four models, both protocols, initialization seed 42 (8 runs).
- Repeat the two core direction-2 models (`decoupled`, `aligned`) with seeds 3407 and 2026 on both protocols (8 additional runs). This supplies three-seed evidence for the incremental alignment effect. Original and MLP remain one-seed reference baselines in this first round.
- Maximum 40 epochs, early stopping after 8 epochs without higher validation AUC; AdamW lr .0005, weight decay .0001, batch 128, clip norm 1, cosine schedule with T_max 40, ordinary BCE without label smoothing. No test-selected hyperparameters or seed selection.
- ROCm training may have nondeterministic kernels; seed control is not a bitwise retraining guarantee. Saved checkpoint inference can be checked separately.
- Two-batch, one-epoch smoke runs validate execution only; their metrics are excluded from results.
- No ensemble, test-time mixing, new semantic embeddings, pretrained language-model fine-tuning, or extra architecture search in this round.

## Independent item-parameter diagnostic

After the planned models finish, estimate a separate Rasch reference from held-out test students for descriptive parameter evaluation only. In the cold protocol, fit this reference on their train-item and test-cold-item responses, excluding validation-cold items, then compare semantic difficulty rankings on the 143 test-cold items. These response-derived reference values are never model inputs, alignment supervision, checkpoint-selection criteria, or hyperparameter-selection criteria. Report Spearman and Pearson correlations on all observed evaluation items, with a clearly labelled optional >=30-response subset. The reference is a noisy, model-based proxy; correlation does not establish true psychological difficulty. This diagnostic was specified before the first cold-model test result was available.

## Invocation (existing WSL environment)

```sh
cd /mnt/d/doc/knowledge_trace_graduation_project/dlkt-research
/home/ml/.venvs/pytorch/bin/python -u -m experiments.run --output runs/round1-s42
/home/ml/.venvs/pytorch/bin/python -u -m experiments.run --output runs/round1-repeat --variants decoupled aligned --seeds 3407 2026
```

Weights, per-student prediction arrays, and full training outputs stay local under ignored `runs/`. Aggregate results, source and methodology are committed to GitHub. Original `dlkt` and `dlkt-best-806453` are preserved.

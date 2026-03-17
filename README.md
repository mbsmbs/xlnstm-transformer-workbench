# xLSTM + Modern Transformer Workbench

> This project was implemented and executed in a **Google Colab** environment.

This repository contains one notebook:

- `IFT6135_A2025_HW2_EN_Byungsuk_Min.ipynb`

The notebook is split into two major implementation blocks:

1. `xLSTM` (extended LSTM) on a parity generalization task
2. A modern decoder-only Transformer for modular arithmetic

This README explains in detail what your code is doing in each section.

## Notebook Structure at a Glance

- **Part 1 (xLSTM)**: Builds exponential-gated recurrent modules from scratch, trains/evaluates against baselines, and compares in-distribution vs long-length generalization.
- **Part 2 (Transformer)**: Implements RMSNorm, SwiGLU, RoPE, Grouped-Query Attention, a decoder LM stack, then runs baseline + regularization sweeps + attention visualization.

---

## Part 1: xLSTM Pipeline (Parity Task)

### 1. Imports and Setup

You import:

- `torch`, `torch.nn`, `logsigmoid`
- `numpy`, `matplotlib`
- `tqdm`
- typing utilities

These support model definition, training loops, plotting, and type hints.

### 2. `ExponentialGates` (core gating primitive)

You define an `ExponentialGates` module with two modes:

- **Exponential mode** (`use_exponential=True`):
  - Computes gate pre-activations from `Wx + Ry + b`
  - Splits into input/forget/cell/output channels
  - Uses stabilizer state `m_t`:
    - `m_t = max(f_tilde + m_prev, i_tilde)`
  - Stabilized exponential gates:
    - `i_stab = exp(i_tilde - m_t)`
    - `f_stab = exp(f_tilde + m_prev - m_t)`
  - Updates:
    - `c_t = f_stab * c_prev + i_stab * tanh(z_tilde)`
    - `n_t = f_stab * n_prev + i_stab`
    - `h_t = sigmoid(o_tilde) * (c_t / (n_t + eps))`

- **Sigmoid mode** (`use_exponential=False`):
  - Standard LSTM-like gates `sigmoid/tanh`
  - Uses placeholder `n_t=1`, `m_t=0` to keep interface consistent

It returns both:

- next states `(h_t, c_t, n_t, m_t)`
- gate activations for analysis/debugging

### 3. `sLSTMCell` (scalar-memory recurrence)

`sLSTMCell` wraps the recurrent step logic over full sequences:

- Keeps states as `[4, B, H]` for `(h, c, n, m)`
- Initializes zero states if none provided
- Uses explicit recurrent projections `R_i, R_f, R_z, R_o`
- Adds forget bias initialization (`+1` on forget slice)
- Processes time-major sequence in `forward_sequence`
- Calls selected pointwise function:
  - `forward_pointwise_exp` or `forward_pointwise_sigmoid`

Output:

- full hidden sequence `[B, T, H]`
- final state `[4, B, H]`

### 4. `sLSTMLayer` (input projection + recurrent pass)

This layer:

- Applies input-side gate projections `W_i, W_f, W_z, W_o` to `x`
- Concatenates to `[B, T, 4H]`
- Runs through `sLSTMCell`
- Applies dropout to sequence output

So this is the sequence encoder used inside higher-level xLSTM blocks.

### 5. xLSTM block stack (`LayerNorm`, `GatedMLP`, `xLSTMBlock`, `xLSTM`)

You implement a residual architecture:

- `LayerNorm`: custom residual-weight form with `weight_proxy = 1 + weight`
- `GatedMLP`: `Linear(H -> 4H/3) -> GELU -> Linear(4H/3 -> H)`
- `xLSTMBlock`:
  - Branch 1: `LayerNorm -> sLSTMLayer -> Dropout -> Residual add`
  - Branch 2: `LayerNorm -> GatedMLP -> Dropout -> Residual add`
- `xLSTM`: stacks multiple `xLSTMBlock`s in `nn.ModuleList`

### 6. Data generation (`ParityDataset`)

The parity dataset generator:

- creates binary sequences
- labels each by parity (`sum(bits) % 2`)
- supports variable lengths
- pads sequences and builds masks for valid timesteps

Three dataset modes are used:

- train: lengths 1-40
- in-distribution test: lengths 1-40
- OOD generalization test: lengths 40-256

### 7. Model creation and training logic

`ModelFactory` creates three models:

- `xLSTM (Exponential)`
- `xLSTM (Sigmoid)`
- `Vanilla LSTM` baseline

Each model gets:

- `input_proj: Linear(1 -> hidden_size)`
- `classifier: Linear(hidden_size -> 2)`

`ModelTrainer`:

- uses `AdamW` + `CosineAnnealingLR`
- projects input bits before encoding
- for each sample selects hidden state at last valid position using mask
- classifies parity with cross-entropy
- reports epoch loss/accuracy

### 8. Experiment orchestration (`ExperimentRunner`)

`ExperimentRunner` performs:

- unit-style checks for Part 1.1/1.2/1.3 module outputs and shapes
- full training for all 3 models
- evaluation on ID and OOD sets
- plotting:
  - training loss curves
  - ID accuracy bars
  - OOD accuracy bars
- final summary table

### 9. Part 1 execution config

Main run cell uses:

- `hidden_size=64`, `num_layers=2`
- `train_length=40`, `test_max_length=256`
- `num_samples=10000`, `epochs=20`
- `batch_size=256`, `learning_rate=1e-3`, `weight_decay=0.1`

It runs all tests and the full experiment pipeline end-to-end.

---

## Part 2: Modern Decoder Transformer (Modular Arithmetic)

### 1. Setup

You re-import modules for this section, set seeds (`torch`, `numpy`), and pick device (`cuda` if available).

### 2. Core layers

#### `RMSNorm`

- Implements root-mean-square normalization
- Learnable scale vector only (no bias inside norm computation)

#### `FeedForward` (SwiGLU)

- bias-free `w_gate`, `w_value`, `w_out`
- computes `silu(w_gate(x)) * w_value(x)`
- dropout then output projection

#### RoPE (`rotate_half`, `apply_rope`)

- builds inverse frequencies
- computes sinusoidal rotation angles by position
- rotates first `d_rope` channels of Q/K
- supports partial rotary embedding if `d_rope < d_h`

### 3. `GroupedQueryAttention` (GQA)

Forward flow:

1. project `x` into Q and combined KV
2. reshape Q into `H` heads
3. reshape KV into `G` groups and split K/V
4. repeat-interleave K/V groups to match `H` heads
5. apply RoPE to Q and K
6. compute scaled dot-product scores
7. apply causal mask (upper triangle blocked)
8. softmax + dropout
9. weight V and merge heads
10. output projection back to model dimension

Also returns attention weights for visualization.

### 4. `DecoderBlock` (parallel pre-norm)

For each block:

- `attn_out = Attention(RMSNorm(x))`
- `ffn_out = FFN(RMSNorm(x))`
- `y = x + dropout(attn_out) + dropout(ffn_out)`

This is parallel pre-normalization (both branches see the same normalized input).

### 5. `ModernDecoderLM`

Architecture:

- token embedding (no absolute position embedding)
- stack of decoder blocks
- final RMSNorm
- LM head with **weight tying** to token embedding

Forward returns:

- `logits` of shape `[B, S, vocab_size]`
- stacked hidden states `[L, B, S, d]`
- stacked attention maps `[L, B, H, S, S]`

### 6. Dataset construction (`create_modular_arithmetic_dataset`)

Builds equations over `Z/pZ` (default `p=11`) for both `+` and `*`:

- binary form: `[BOS] a op b [=] r [EOS] [PAD]`
- ternary form: `[BOS] a op b op c [=] r [EOS]`

Vocabulary includes:

- digits `0..p-1`
- `+`, `*`, `[BOS]`, `[EOS]`, `[PAD]`, `[=]`

Then:

- shuffles all generated sequences
- splits train/val
- makes autoregressive pairs:
  - input = all tokens except last
  - target = all tokens except first

### 7. Training and metrics

`compute_metrics` evaluates only tokens **after `[=]`** (excluding `[PAD]` and `[EOS]`):

- masked cross-entropy on RHS tokens
- sequence-level accuracy (all masked positions correct)

`train_model`:

- optimizer: `AdamW`
- scheduler: linear warmup + cosine decay
- gradient clipping
- per-epoch train/val metrics
- optional parameter norm tracking
- saves best checkpoint to `best_model.pt`

### 8. Experiments run in notebook

#### Experiment 1 (baseline)

- config: `d=128`, `L=4`, `H=8`, `G=4`, `d_ff=512`
- `dropout=0.1`, `lr=3e-4`, `weight_decay=1e-4`
- `epochs=100`, `batch_size=64`

Outputs:

- baseline training curves (`exp1_baseline_curves.png`)
- summary metrics table

#### Experiment 2A (dropout sweep)

Runs `dropout in {0.0, 0.2, 0.4}` and compares:

- train/val loss curves
- train/val accuracy curves
- summary table by dropout

Saves:

- `exp2a_dropout_comparison.png`

#### Experiment 2B (weight decay sweep)

Runs `weight_decay in {2.5e-4, 5e-4, 1e-3}` with parameter norm tracking.

Compares:

- parameter norm trajectory
- validation accuracy trajectory
- summary table with best val acc and final norm

Saves:

- `exp2b_weight_decay_analysis.png`

#### Experiment 3 (attention visualization)

- loads best baseline checkpoint
- visualizes final-layer attention heads on 5 standardized arithmetic examples
- saves one figure per example:
  - `attention_binary_add_small.png`
  - `attention_binary_add_carry.png`
  - `attention_binary_mult_small.png`
  - `attention_ternary_add.png`
  - `attention_ternary_mult.png`

### 9. Results serialization

The notebook stores experiment outputs in:

- `HW2_A25_modern_transformer_results.pkl`

Saved payload includes baseline history/metrics/config and sweep histories.

---

## What Your Notebook Is Doing Overall

In short, your code:

- implements two sequence-modeling systems from scratch (xLSTM and modern Transformer),
- validates component correctness with shape/tests,
- trains and compares models on algorithmic tasks,
- studies regularization/optimization effects,
- and produces both quantitative summaries and interpretability plots.

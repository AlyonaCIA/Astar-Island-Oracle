# Bayesian_approach

Standalone Astar Island workspace for a crafted empirical-Bayes model and a stronger hybrid mode that blends the Bayesian posterior with a local CNN prior.

This directory is self-contained and does not require editing the original repo.

## What Is In This Folder

- [`bayes_oracle.py`](/Users/souvikb/EUI_SST/codes/Bayesian_approach/bayes_oracle.py)
  Main entrypoint.
  Supports offline evaluation, live prediction generation, and live submission.

- [`cnn_prior.py`](/Users/souvikb/EUI_SST/codes/Bayesian_approach/cnn_prior.py)
  Minimal local copy of the `unet_cond` prior needed for hybrid mode.

- [`checkpoints/cnn_latest.pt`](/Users/souvikb/EUI_SST/codes/Bayesian_approach/checkpoints/cnn_latest.pt)
  Local refreshed checkpoint bundled into this workspace.

- [`ground_truth`](/Users/souvikb/EUI_SST/codes/Bayesian_approach/ground_truth)
  Cached completed-round ground truth files used to build priors and run offline checks.

- `observations_*.json`
  Cached observation files for completed rounds.

- `round_*.json`
  Cached round metadata files.

- [`check_setup.py`](/Users/souvikb/EUI_SST/codes/Bayesian_approach/check_setup.py)
  Quick workspace validation.

## Model Summary

There are two modes:

- Pure Bayesian
  Feature-conditioned empirical-Bayes priors learned from cached completed rounds, then updated with pooled round observations and direct per-cell counts.

- Hybrid CNN + Bayesian
  Uses the local refreshed CNN checkpoint as a prior and applies Bayesian corrections on top.
  This is the recommended mode.

## Recommended Directory Layout

Anyone running this locally should have:

```text
Bayesian_approach/
  .env
  bayes_oracle.py
  cnn_prior.py
  check_setup.py
  requirements.txt
  checkpoints/
    cnn_latest.pt
  ground_truth/
    r10_s0_....json
    ...
  observations_....json
  round_....json
```

## Step-By-Step Setup

### 1. Clone or copy this folder onto your own branch

From your own repo workflow, create a non-main branch first:

```bash
git checkout -b your-branch-name
```

Then copy this folder into that branch or commit it directly there.

### 2. Create a Python virtual environment

From inside this directory:

```bash
cd Bayesian_approach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Add the API token

Create `.env` from the example:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
ASTAR_TOKEN=your_jwt_token_here
```

### 4. Make sure the local assets are present

You said your friend already has all the data, replays, and observations. For this workspace specifically, the minimum required assets are:

- `ground_truth/`
- `observations_*.json`
- `round_*.json`
- `checkpoints/cnn_latest.pt`

### 5. Validate the workspace

```bash
python check_setup.py
```

You should see:

```text
Workspace check passed.
```

## Offline Usage

### A. Evaluate one cached completed round

Pure Bayesian:

```bash
python bayes_oracle.py round-eval 16 --exclude-round
```

Hybrid mode:

```bash
python bayes_oracle.py round-eval 16 --exclude-round --hybrid-cnn
```

Notes:

- `--exclude-round` means the evaluated round is not used to build priors.
- This is the fairest offline sanity check.

### B. Run leave-one-observed-round-out cross-validation

Pure Bayesian:

```bash
python bayes_oracle.py crossval
```

Hybrid mode:

```bash
python bayes_oracle.py crossval --hybrid-cnn
```

## Live Usage

### Recommended: Dry-run first

This queries the active round, builds predictions, and writes them locally without submitting:

```bash
python bayes_oracle.py live --hybrid-cnn --no-submit
```

This is the safest first command before spending the final submission.

### Submit to the active round

```bash
python bayes_oracle.py live --hybrid-cnn
```

What this does:

1. Loads historical priors from cached completed rounds.
2. Queries the active round using a settlement-focused plan.
3. Pools observations across all 5 seeds to estimate round-level updates.
4. Produces Bayesian predictions.
5. Blends them with the local CNN prior.
6. Submits predictions for all seeds.

### Reuse cached observations instead of querying again

If the round’s observations were already collected and saved locally:

```bash
python bayes_oracle.py live --hybrid-cnn --reuse-only
```

Important:

- `--reuse-only` is only useful if the corresponding `observations_<roundid>.json` file already exists locally.
- If there are no cached observations for the active round, the model will have no live evidence to use.

## Command Reference

### Pure Bayesian

```bash
python bayes_oracle.py crossval
python bayes_oracle.py round-eval 18 --exclude-round
python bayes_oracle.py live --no-submit
python bayes_oracle.py live
```

### Hybrid CNN + Bayesian

```bash
python bayes_oracle.py crossval --hybrid-cnn
python bayes_oracle.py round-eval 18 --exclude-round --hybrid-cnn
python bayes_oracle.py live --hybrid-cnn --no-submit
python bayes_oracle.py live --hybrid-cnn
```

## Practical Recommendation

Use the hybrid mode:

```bash
python bayes_oracle.py live --hybrid-cnn
```

That is the strongest mode currently available in this workspace.

## What To Commit

Commit at least:

- `bayes_oracle.py`
- `cnn_prior.py`
- `check_setup.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`

Do not commit:

- `.env`
- `.venv/`

Commit the checkpoint and cached data only if your branch policy allows large artifacts. If not, keep them local and document that requirement for the person running the workspace.

## Suggested Local Checklist Before Push

Run these in order:

```bash
python check_setup.py
python bayes_oracle.py round-eval 16 --exclude-round --hybrid-cnn
python bayes_oracle.py crossval --hybrid-cnn
python bayes_oracle.py live --hybrid-cnn --no-submit
```

If all of those behave normally, the workspace is ready for another person to run locally.

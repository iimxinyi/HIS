# Hybrid Inference Scheme (HIS)

**Paper:** "A Novel Hybrid Inference Scheme for Diffusion-Based AIGC Services in MEC Networks" — submitted to IEEE Transactions on Mobile Computing (IEEE TMC).

**Goal:** Generate images at varying common-inference-step values, score them with both subjective (image-text alignment) and objective (no-reference IQA) metrics, then fit the image-quality function from Section III of the paper.

**Experimental platform:** Ubuntu 22.04, NVIDIA A800 x 4 / RTX 4090D x 2.

Supports two diffusion backbones with parallel pipelines:

- **Stable Diffusion 3 Medium** (`sd3-medium/`)
- **FLUX.1-dev** (`flux.1-dev/`)

Each model directory has its own entry-point scripts; everything model-agnostic (prompts, metrics, similarity, fitting) lives under `prompts/` and `common/`.

---

## 1. Repository layout

```
HIS/
├── README.md
├── pyproject.toml             # uv-managed dependencies
├── uv.lock
├── Dockerfile                 # patches diffusers + ImageReward + clip imports
├── fit_sigmoid.py             # standalone sigmoid fitter (model-agnostic)
│
├── prompts/                   # single source of truth for prompts
│   ├── group1.py              # 2 public + 20 personal
│   └── group2.py              # DiffusionDB prompts, fill in before running
│
├── common/                    # model-agnostic helpers
│   ├── seed.py                # seed_everywhere(seed)
│   ├── io.py                  # already_done / ensure_parent / reserve_latents_path
│   ├── pretrained_util.py     # PRETRAINED_DIR + HF/torch cache redirection
│   ├── metrics_clip.py        # subjective: CLIP ViT-L/14@336px
│   ├── metrics_image_reward.py# subjective: ImageReward (BLIP, human-preference)
│   ├── metrics_brisque.py     # objective: BRISQUE (piq)
│   ├── metrics_musiq.py       # objective: MUSIQ (pyiqa, SPAQ ckpt)
│   ├── sentence_similarity.py # all-MiniLM-L6-v2 cosine, 0.1 binning
│   └── fitting.py             # legacy sigmoid helpers (still used by fit_sigmoid.py)
│
├── pretrained/                # all caches and downloaded weights live here
│   ├── README.md
│   ├── all-MiniLM-L6-v2/      # sentence-transformer (manually fetched in offline setups)
│   ├── bert-base-uncased/     # ImageReward's BLIP tokenizer (manually fetched)
│   ├── clip/                  # CLIP ViT-L/14@336px (auto-cached on first run)
│   ├── ImageReward-v1.0/      # ImageReward weights (auto)
│   ├── musiq_spaq_ckpt-358bb6af.pth   # MUSIQ weights
│   ├── torch/                 # torch.hub cache (piq BRISQUE SVM weights, etc.)
│   └── huggingface/           # HF hub cache for everything else
│
├── sd3-medium/
│   ├── pipeline_stable_diffusion_3.py # patched (common_step + prompt_unchanged)
│   ├── demo.py
│   ├── generate_similarity.py         # similarity sweep, two modes, resumable
│   ├── generate_sim.py                # SIM sweep across (scale, common_step)
│   ├── evaluate.py                    # all 4 metrics -> Excel
│   ├── fit_similarity.py              # aggregate to (sim_bin, common_step) points
│   └── results/                       # populated by the scripts
│
└── flux.1-dev/                # mirrors sd3-medium/
    ├── pipeline_flux.py       # patched
    ├── demo.py
    ├── generate_similarity.py
    ├── generate_sim.py
    ├── evaluate.py
    ├── fit_similarity.py
    └── results/
```

---

## 2. Setup

### Option A: Docker (recommended)

The Dockerfile builds a CUDA 12.4 + Python 3.10 image, installs everything via
uv, patches the bundled `diffusers` SD3 / FLUX pipelines with the modified
versions in this repo, and applies a few small monkey-patches to `openai-clip`
and `ImageReward` for new-version compatibility.

```bash
# Build
docker build --network=host -t his .

# Run (mounts repo + local model snapshots; persistent caches under pretrained/)
docker run --gpus all -it --rm --network=host \
    -v $PWD:/workspace \
    -v /path/to/Stable-Diffusion-3-Medium:/models/sd3 \
    -v /path/to/FLUX.1-dev:/models/flux \
    -e SD3_MODEL_PATH=/models/sd3 \
    -e FLUX_MODEL_PATH=/models/flux \
    -e HF_HUB_OFFLINE=1 \
    his bash
```

Look for the `patched ...` lines at the end of the build — they confirm both
pipeline files were installed into the bundled diffusers.

### Option B: uv (local)

```bash
uv sync
# Patch diffusers' bundled pipelines once
uv run python - <<'PY'
import pathlib, shutil, diffusers
root = pathlib.Path(diffusers.__file__).resolve().parent
for src, name in [
    ("sd3-medium/pipeline_stable_diffusion_3.py", "pipeline_stable_diffusion_3.py"),
    ("flux.1-dev/pipeline_flux.py", "pipeline_flux.py"),
]:
    dst = sorted(root.glob(f"**/{name}"))[0]
    shutil.copy(src, dst)
    print("patched", dst)
PY
```

---

## 3. Pretrained weights and caches

All caches live under `pretrained/` so they survive container restarts (via `-v $PWD:/workspace`).

| Item | Path |
|---|---|
| SD3-Medium | `$SD3_MODEL_PATH` (env var) |
| FLUX.1-dev | `$FLUX_MODEL_PATH` (env var) |
| CLIP ViT-L/14@336px | `pretrained/clip/` |
| ImageReward | `pretrained/ImageReward-v1.0/` |
| MUSIQ (SPAQ) | `pretrained/musiq_spaq_ckpt-358bb6af.pth` |
| BRISQUE SVM | `pretrained/torch/hub/checkpoints/brisque_svm_weights.pt` |
| bert-base-uncased tokenizer | `pretrained/bert-base-uncased/` |
| all-MiniLM-L6-v2 | `pretrained/all-MiniLM-L6-v2/` |


---

## 4. Experiments

The pipeline has three phases: **generate → evaluate → aggregate → (sigmoid fit)**. Every generator is **resumable** — it skips outputs that already exist, so you can Ctrl+C and rerun without losing progress. Each `(sample, common_step)` is generated with 3 seeds (default `1 2 3`); metrics are averaged across seeds.

### 4.1 SIM experiment

Sweeps the public-prompt guidance scale **for each common_step value**.

```bash
python sd3-medium/generate_sim.py
python flux.1-dev/generate_sim.py
```

Files: `results/sim/Guidance_Scale={s}/scale{s}_step{k}_prompt{j}_seed{r}.png`.

> **Note on FLUX SIM**: FLUX.1-dev is guidance-distilled — `guidance_scale` is
> embedded into the transformer rather than computed as a two-pass classifier-free
> guidance. The sweep still works as an exploration of the embedded-guidance
> value, but the mechanism differs from SD3.

### 4.2 Similarity experiment

Two variants and two anchoring modes are run by default.

**Variants** — different guidance-scale combinations:
- `wSIM`  : `public_scale != personal_scale` (the SIM mechanism is active)
- `woSIM` : `public_scale == personal_scale` (baseline without SIM)

**Anchoring modes** — different ways of choosing the "common phase" prompt:
- `public-anchored`   : `public_prompts[i]` anchors the common phase; every personal[j] swaps in.  
  Files: `Public{i}_Personal{j}_CommonStep{k}_Seed{s}.png`
- `personal-anchored` : `personal_prompts[i]` anchors; every OTHER `personal[j]` (j ≠ i) swaps in.  
  Files: `PAnchor{i}_POther{j}_CommonStep{k}_Seed{s}.png`

Personal-anchored fills in the middle similarity bins that the public-anchored
sweep alone leaves sparse.

```bash
# Both variants, both modes (resumable; default)
python sd3-medium/generate_similarity.py --group 1
python flux.1-dev/generate_similarity.py --group 1

# Just the new mode (existing public-anchored images untouched)
python sd3-medium/generate_similarity.py --group 1 --modes personal-anchored

# Subset the personal-anchored anchors to save GPU time
python sd3-medium/generate_similarity.py --group 1 --modes personal-anchored \
    --cross-anchors 0 5 10 15
```

Outputs land under `results/similarity/group{N}/{variant}/`.


### 4.3 Evaluation

Runs all four metrics on every PNG and writes an Excel per variant (similarity)
or per experiment (SIM):

```bash
# Similarity: writes one xlsx per variant; defaults to both wSIM and woSIM
python sd3-medium/evaluate.py --exp similarity --group 1
python sd3-medium/evaluate.py --exp similarity --group 1 --variants wSIM

# SIM
python sd3-medium/evaluate.py --exp sim
```

Outputs:
- `results/eval/similarity_group{N}_{variant}.xlsx`
- `results/eval/sim.xlsx`

Each Excel has a `raw` sheet (one row per image) plus four per-metric pivot
sheets averaged across seeds.

### 4.4 Aggregate into (sim_bin × common_step) points

`fit_similarity.py` joins each image to its sentence-similarity bin (`cos(anchor,
target)` rounded to nearest 0.1) and averages the metric over all (anchor, target,
seed) tuples in each bin per common_step:

```bash
python sd3-medium/fit_similarity.py --group 1
python flux.1-dev/fit_similarity.py --group 1
```

Output: `results/fitting/group{N}/{variant}_points.xlsx` with four sheets
(`clip`, `image_reward`, `brisque`, `musiq`). Each sheet's index is `sim_bin`,
columns are `common_step`, cells are the mean metric value. This is the input
for the sigmoid-fit step.

### 4.5 Sigmoid fitting (standalone)

```bash
# Fit CLIP curves from SD3 wSIM, group 1, output beside the input file
python fit_sigmoid.py \
    --input sd3-medium/results/fitting/group1/wSIM_points.xlsx \
    --metric clip

# Only fit specific similarity bins
python fit_sigmoid.py --input ... --metric clip --sim-bins 0.2 0.3 0.4 0.5 0.6 0.7

# Change resample step for the curve output (default 0.03)
python fit_sigmoid.py --input ... --metric clip --x-step 0.05
```

For each invocation the script writes a directory `{input_stem}_{metric}_sigmoid/`:
- `plot.png` — scatter + fitted curves, one colour per `sim_bin`
- `params.csv` — fitted `(L, k, x0, C)` per bin
- `curves.xlsx` — resampled curves at the chosen step (`x` column + one column
  per `sim_bin`); the granular y-values you can paste into Origin etc.

The sigmoid form is `y = L / (1 + exp(k * (x - x0))) + C`.

---

## 5. Metrics

| Metric | Type | What it measures | Direction |
|---|---|---|---|
| **CLIP** (ViT-L/14@336px) | Subjective (alignment) | Cosine similarity between image and prompt embeddings | Higher = better |
| **ImageReward** (BLIP) | Subjective (alignment) | Human-preference-trained score (≈ [-3, 3]) | Higher = better |
| **BRISQUE** (piq) | Objective (NR-IQA) | Natural-scene-statistics distortion score | Lower = better |
| **MUSIQ** (SPAQ, pyiqa) | Objective (NR-IQA) | Multi-scale transformer quality, /100 | Higher = better |

ImageReward can be negative — that's the model expressing dislike, not a bug.
BRISQUE can occasionally extrapolate outside [0, 100] for diffusion images that
fall outside its natural-image training distribution; that's a property of the
piq implementation and is left un-clamped.

---

## 6. File-naming reference

SIM (under `results/sim/`):
```
Guidance_Scale={s}/scale{s}_step{k}_prompt{j}_seed{r}.png
  s: public-prompt guidance scale
  k: common inference step
  j: personal prompt index
  r: seed
```

Similarity (under `results/similarity/group{N}/{variant}/`):
```
Public{i}_Personal{j}_CommonStep{k}_Seed{s}.png      # public-anchored mode
PAnchor{i}_POther{j}_CommonStep{k}_Seed{s}.png       # personal-anchored mode
  i, j: prompt indices into prompts/group{N}.py
  k:    common inference step (0..total_step-1)
  s:    seed (default sweep {1, 2, 3}, averaged at evaluation)
```

---

## 7. Adding more prompts from DiffusionDB (Group 2)

Open `prompts/group2.py` and fill in `public_prompts` + `personal_prompts`.
The cost of the similarity experiment is roughly:
```
images = (N_public * N_personal + N_personal * (N_personal - 1))   # both modes
         * total_step * N_seeds * 2 variants
```

Run the same commands with `--group 2`. Each group has independent output
directories, eval Excels, and fitting outputs.

---

## 8. Acknowledgements

- [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [CLIP](https://github.com/openai/CLIP)
- [ImageReward](https://github.com/THUDM/ImageReward)
- [BRISQUE via piq](https://piq.readthedocs.io)
- [MUSIQ via pyiqa](https://github.com/chaofengc/IQA-PyTorch)
- [Sentence-Transformers (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [DistributedDiffusion](https://github.com/HongyangDu/DistributedDiffusion)

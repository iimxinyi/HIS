# Hybrid Inference Scheme (HIS)

**Paper:** "A Novel Hybrid Inference Scheme for Diffusion-Based AIGC Services in MEC Networks" — submitted to IEEE Transactions on Mobile Computing (IEEE TMC).

**Target:** Generate images at varying common-inference-step values, score them with both subjective (image-text alignment) and objective (no-reference IQA) metrics, then fit the image-quality function from Section III of the paper.

**Experimental Platform:** Ubuntu 20.04, Intel Xeon Gold 6248R, NVIDIA A100.

This repository supports two diffusion models in parallel:

- **Stable Diffusion 3 Medium** (`sd3-medium/`)
- **FLUX.1-dev** (`flux.1-dev/`)

Each model directory has its own entry-point scripts but shares prompts (`prompts/`), metrics, and fitting code (`common/`).

---

## 1. Repository layout

```
HIS/
├── README.md
├── pyproject.toml          # uv-managed dependencies
├── uv.lock
├── Dockerfile              # patches diffusers' SD3 and FLUX pipelines
│
├── prompts/                # prompt groups (single source of truth)
│   ├── group1.py           # 2 public + 20 personal (cats + dogs) - filled in
│   └── group2.py           # empty template; fill in before running
│
├── common/                 # model-agnostic helpers
│   ├── seed.py
│   ├── io.py               # skip-if-exists check + temp-latents path
│   ├── metrics_clip.py         # subjective: CLIP image-text alignment
│   ├── metrics_image_reward.py # subjective: ImageReward (BLIP, human-preference)
│   ├── metrics_brisque.py      # objective: BRISQUE
│   ├── metrics_musiq.py        # objective: MUSIQ (SPAQ ckpt)
│   ├── sentence_similarity.py  # all-MiniLM-L6-v2 -> 0.1 bin
│   └── fitting.py              # sigmoid fit + matplotlib helpers
│
├── pretrained/             # local model checkpoints (see pretrained/README.md)
│   └── musiq_spaq_ckpt-358bb6af.pth   # YOU drop this in
│
├── sd3-medium/
│   ├── pipeline_stable_diffusion_3.py  # patched (common_step + prompt_unchanged)
│   ├── demo.py
│   ├── generate_similarity.py          # similarity sweep, resumable
│   ├── generate_sim.py                 # SIM sweep, resumable
│   ├── evaluate.py                     # all 4 metrics -> Excel
│   ├── fit_similarity.py               # sigmoid fit + plots
│   └── results/                        # populated by the scripts
│
└── flux.1-dev/             # mirrors sd3-medium/
    ├── pipeline_flux.py    # patched (common_step + prompt_unchanged)
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

```bash
docker build -t his .
docker run --gpus all -it --rm -v $PWD:/app his bash
```

The Docker build automatically replaces the bundled `diffusers` SD3 and FLUX pipelines with the patched copies in this repo (look for the `patched ...` lines at the end of the build).

### Option B: uv (local)

```bash
uv sync
# manually patch diffusers' pipelines (one-time)
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

### Pretrained weights

Drop `musiq_spaq_ckpt-358bb6af.pth` into `pretrained/`. See `pretrained/README.md` for the source path. ImageReward weights are downloaded automatically on first use.

### Model checkpoints

Set environment variables to point to your local HuggingFace snapshots (otherwise the scripts download from the Hub):

```bash
export SD3_MODEL_PATH=/path/to/Stable-Diffusion-3-Medium
export FLUX_MODEL_PATH=/path/to/FLUX.1-dev
```

---

## 3. Running experiments

Each command below has an SD3 form (under `sd3-medium/`) and an identical FLUX form (under `flux.1-dev/`). All generators are **resumable** — they check whether each output PNG already exists and skip it if so. Kill the script at any time and rerun to continue.

### 3.1 Similarity experiment

Generates one image per `(public_prompt, personal_prompt, common_step)` triple for the selected prompt group. This is the data behind the Section III fitting.

```bash
# Group 1 (default)
python sd3-medium/generate_similarity.py --group 1
python flux.1-dev/generate_similarity.py --group 1

# Group 2 (after you fill in prompts/group2.py)
python sd3-medium/generate_similarity.py --group 2
python flux.1-dev/generate_similarity.py --group 2
```

Outputs land under `results/similarity/group{N}/Public{i}_Personal{j}_CommonStep{k}.png`.

### 3.2 SIM experiment

Sweeps the public-prompt guidance scale (Group 1 only). Useful for validating the Semantic Intensity Modulator design.

```bash
python sd3-medium/generate_sim.py
python flux.1-dev/generate_sim.py
```

Outputs land under `results/sim/Guidance_Scale={i}/scale{i}_prompt{j}_seed{s}.png`.

> Note on FLUX SIM: FLUX.1-dev is guidance-distilled — `guidance_scale` is embedded in the transformer rather than computed as a two-pass classifier-free guidance. The sweep still works as an exploration of the embedded-guidance value, but the result has a different mechanism than on SD3.

### 3.3 Evaluation

Computes CLIP, ImageReward, BRISQUE, and MUSIQ for every generated image and writes an Excel with one sheet per metric:

```bash
python sd3-medium/evaluate.py --exp similarity --group 1
python sd3-medium/evaluate.py --exp sim
# same commands under flux.1-dev/
```

Output: `results/eval/similarity_group{N}.xlsx`, `results/eval/sim.xlsx`.

### 3.4 Fitting

Joins each personal prompt to its sentence-similarity bin (rounded to 0.1), averages each metric per (bin, common_step), fits a sigmoid, and saves a plot per metric plus a CSV of fitted params:

```bash
python sd3-medium/fit_similarity.py --group 1
python flux.1-dev/fit_similarity.py --group 1
```

Output: `results/fitting/group{N}/{metric}.png` + `params.csv`.

---

## 4. What each metric measures

| Metric | Type | What it measures | Direction |
|---|---|---|---|
| **CLIP** (ViT-L/14@336px) | Subjective (alignment) | Cosine similarity between image and prompt embeddings | Higher is better |
| **ImageReward** (BLIP-based) | Subjective (alignment) | Human-preference-trained score, alignment + aesthetics | Higher is better |
| **BRISQUE** (piq) | Objective (NR-IQA) | Natural-scene statistics distortion score | Lower is better |
| **MUSIQ** (SPAQ ckpt, pyiqa) | Objective (NR-IQA) | Multi-scale transformer quality score, normalised to ~[0, 1] | Higher is better |

---

## 5. Adding more prompts (Group 2)

Open `prompts/group2.py` and fill in the two lists:

```python
public_prompts = [
    "...",
    "...",
]
personal_prompts = [
    "...",
    # 10+ prompts; the similarity experiment costs O(N_pub * N_per * 28) images
]
```

Then run the same `--group 2` commands. Group 2 is independent of Group 1: separate output folders, separate Excel, separate fitting plots.

---

## 6. File naming conventions

Similarity experiment:
```
Public{i}_Personal{j}_CommonStep{k}.png
  i: index of the public prompt
  j: index of the personal prompt
  k: number of common inference steps (0..total_step-1)
```

SIM experiment:
```
Guidance_Scale={s}/scale{s}_prompt{j}_seed{r}.png
  s: public-prompt guidance scale
  j: personal prompt index
  r: seed
```

---

## 7. Acknowledgements

- [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [CLIP](https://github.com/openai/CLIP)
- [ImageReward](https://github.com/THUDM/ImageReward)
- [BRISQUE via piq](https://piq.readthedocs.io)
- [MUSIQ via pyiqa](https://github.com/chaofengc/IQA-PyTorch)
- [Sentence-Transformers (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [DistributedDiffusion](https://github.com/HongyangDu/DistributedDiffusion)

# AIUPred Nextflow pipeline

Thin wrapper around the `aiupred` CLI: one **AIUPRED** process per FASTA, outputs `*.aiupred.tsv` under `params.outdir/<sample_id>/`.

Entry points: repository root [`main.nf`](../main.nf) + [`nextflow.config`](../nextflow.config), or nested [`main.nf`](main.nf) here under `nf/`.

## Prerequisites

- [Nextflow](https://www.nextflow.io/) >= 22.10
- **Either** Micromamba/Conda (`-profile conda`) **or** Docker (`-profile docker`)

### GPU vs CPU

By default the workflow does **not** force CPU. AIUPred uses **GPU when PyTorch sees CUDA**; otherwise it **falls back to CPU**. Add `-profile cpu` to always pass `--force-cpu`.

### Parameters (options)

| Parameter | Maps to |
|-----------|---------|
| `input` | FASTA path or glob |
| `outdir` | Published results root |
| `aiupred.predict_binding` | `-b` |
| `aiupred.predict_linker` | `-l` |
| `aiupred.redox` | `-r` |
| `aiupred.gpu` | `-g` (GPU index) |
| `-profile cpu` | `--force-cpu` |

## Quick examples

Local smoke test (Conda):

```bash
nextflow run . -profile test,conda
```

Remote smoke test:

```bash
nextflow run doszilab/AIUPred -r master -profile test,conda
```

Docker smoke test (`docker` uses the CUDA-capable GHCR image; GPU is used only if CUDA is visible inside the container):

```bash
nextflow run . -profile test,docker
```

Binding + linker (Conda):

```bash
nextflow run . -profile conda --input 'data/*.fasta' --outdir results --aiupred.predict_binding true --aiupred.predict_linker true
```

## Profiles (main two)

| Profile | Purpose |
|---------|---------|
| `conda` | Conda env from [`nf/env/aiupred.yml`](env/aiupred.yml) |
| `docker` | Docker + `ghcr.io/doszilab/aiupred:gpu` (GPU if available in container, else CPU) |
| `docker_cpu` | Docker + smaller CPU image + `--force-cpu` |
| `cpu` | Add to any profile to force `--force-cpu` |
| `test` | Bundled FASTA + test `outdir` |

### Extra profiles (optional)

Defined in [`nextflow.config`](nextflow.config): `docker_gpu` / `gpu` (Docker + `--gpus all`), Singularity variants, etc.

## Container images (GHCR)

Published tags (see root `Dockerfile` / `Dockerfile.gpu` to build locally):

```text
ghcr.io/doszilab/aiupred:cpu
ghcr.io/doszilab/aiupred:gpu
```

## GPU on clusters (optional)

For Slurm / Kubernetes / explicit GPU device passthrough, see comments in [`nextflow.config`](nextflow.config) and adapt `docker_gpu` / executor settings to your site.

## Layout

- `../main.nf` — default workflow entry
- `../nextflow.config` — manifest + includes this file
- `main.nf` — alternate entry (`nextflow run nf/main.nf`)
- `modules/local/aiupred.nf` — **AIUPRED** process
- `nextflow.config` — params + profiles
- `conf/base.config` — default CPUs / memory / time
- `env/aiupred.yml` — Conda env

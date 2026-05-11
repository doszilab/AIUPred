# AIUPred Nextflow pipeline

Thin wrapper around the `aiupred` CLI. The workflow reads one or more FASTA paths from `params.input`, runs **AIUPRED** once per file, and publishes `*.aiupred.tsv` under `params.outdir/<sample_id>/`.

The default entry is **repository-root** [`main.nf`](../main.nf) with [`nextflow.config`](../nextflow.config), so `nextflow run doszilab/AIUPred -r main` works without `-main-script`. The copy under [`nf/main.nf`](nf/main.nf) remains valid for nested runs.

## Prerequisites

- [Nextflow](https://www.nextflow.io/) 22.10 or newer
- One execution profile: **conda** (Micromamba recommended), **docker** / **docker_gpu**, or **singularity** / **singularity_gpu**

### Container images (Docker / Singularity)

Build local tags from the repository root (same tags referenced in `nf/nextflow.config`):

```bash
docker build -t aiupred:cpu .
docker build -f Dockerfile.gpu -t aiupred:gpu .
```

For Singularity/Apptainer, you can build from the local Docker daemon, for example:

```bash
singularity build aiupred_cpu.sif docker-daemon://aiupred:cpu
singularity build aiupred_gpu.sif docker-daemon://aiupred:gpu
```

Then point `process.container` in a local config override to the `.sif` path if you do not use `docker://` URIs.

## Quick start (smoke test)

From the repository root, using the bundled test FASTA and a Conda environment that installs this package (first solve can be slow; prefer Micromamba):

```bash
nextflow run . -profile test,conda
```

Remote (no local clone), once the repository is on GitHub:

```bash
nextflow run doszilab/AIUPred -r main -profile test,conda
```

Nested entry (equivalent):

```bash
nextflow run nf/main.nf -profile test,conda
```

CPU-only execution inside Docker after building `aiupred:cpu`:

```bash
nextflow run . -profile test,docker,cpu
```

Custom input and output directory:

```bash
nextflow run . -profile conda --input 'data/*.fasta' --outdir results
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `input` | Glob or path to one FASTA file (required unless `-profile test`) |
| `outdir` | Published results root (default: `results`) |
| `aiupred.binding` | Maps to `-b` |
| `aiupred.linker` | Maps to `-l` |
| `aiupred.redox` | Maps to `-r` |
| `aiupred.gpu` | Maps to `-g` (ignored when `aiupred.force_cpu` is true) |
| `aiupred.force_cpu` | Passes `--force-cpu` (also enabled by `-profile cpu`) |

## Profiles

| Profile | Purpose |
|---------|---------|
| `conda` | Conda env from `nf/env/aiupred.yml` (PyTorch + `pip` install of this repo via relative path) |
| `docker` | CPU container `aiupred:cpu` |
| `docker_gpu` | CUDA container `aiupred:gpu` with `containerOptions = '--gpus all'` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) |
| `singularity` | `docker://aiupred:cpu` |
| `singularity_gpu` | `docker://aiupred:gpu` with `singularity.runOptions = '--nv'` |
| `test` | Sets `input` to `nf/tests/data/test.fasta` and `outdir` to `nf/tests/results` |
| `cpu` | Sets `params.aiupred.force_cpu = true` for portable CPU-only runs |

Combine profiles as needed, for example: `-profile test,docker,cpu`.

## GPU on cluster schedulers (optional)

The default `docker_gpu` / `singularity_gpu` profiles cover workstation-style GPU passthrough. On **Slurm**, you typically request a GPU via the executor instead of Docker options, for example (site-specific; adjust partition and GRES syntax):

```groovy
process {
    executor = 'slurm'
    clusterOptions = '--gres=gpu:1'
    withName: 'AIUPRED' {
        container = 'docker://aiupred:gpu'
    }
}
```

On **Kubernetes** (Nextflow [k8s executor](https://www.nextflow.io/docs/latest/kubernetes.html)), you can request accelerators on the **AIUPRED** process (field names depend on your Nextflow version), for example:

```groovy
process {
    withName: 'AIUPRED' {
        accelerator = [request: 1, type: 'nvidia.com/gpu']
    }
}
```

Validate against your cluster’s device plugin labels and your Nextflow version’s `accelerator` syntax.

## Layout

- `main.nf` (repo root) — default entry workflow for `nextflow run doszilab/AIUPred …`
- `nextflow.config` (repo root) — manifest + includes `nf/nextflow.config`
- `nf/main.nf` — alternate entry (same workflow) when invoked as `nextflow run nf/main.nf`
- `nf/modules/local/aiupred.nf` — **AIUPRED** process
- `nf/nextflow.config` — params and profiles
- `nf/conf/base.config` — default CPUs / memory / time for **AIUPRED**
- `nf/env/aiupred.yml` — Conda specification

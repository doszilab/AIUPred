# AIUPred

[About](#about) | [How to install](#install) | [Command Line Usage](#cli_usage) | [Programmatic Usage (Python API)](#api_usage) | [API Reference](#api_ref) | [Benchmarks](#benchmark) | [License](#license) | [How to cite](#cite)

## <a name="about">About</a>

Intrinsically disordered proteins (IDPs) have no single well-defined tertiary structure under native conditions. AIUPred is a tool that allows to identify disordered protein regions, their binding sites, and flexible linkers, created by Zsuzsanna Dosztányi and Gábor Erdős. 

AIUPred provides both a **standalone command-line interface (CLI)** and an **importable Python library** for seamless integration into bioinformatics pipelines. The library can also be used as a **feature extractor** to generate high-quality structural embeddings for downstream machine learning tasks.

AIUPred is also available as a web server: [https://aiupred.elte.hu/](https://aiupred.elte.hu/)

For more information please refer to the publication: [AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkae385/7673484)

### Requirements
AIUPred requires Python 3.8+ and relies on modern scientific computing libraries (`torch>=2.0.0`, `numpy>=1.21.0`, `scipy>=1.7.0`). These dependencies are **automatically installed** when you install the package via `pip`.

---

## <a name="install">How to install</a>

<b>It is recommended to install AIUPred inside a virtual environment (e.g., Conda or venv).</b>

Because AIUPred is a standard Python package, you can install it directly from GitHub using `pip`:

```bash
pip install git+https://github.com/doszilab/AIUPred.git
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone git@github.com:doszilab/AIUPred.git
cd AIUPred
pip install .
```

---

## <a name="cli_usage">Command Line Usage</a>

Installing AIUPred automatically registers the `aiupred` command in your terminal. You can run it from anywhere on your system.

To carry out a standard disorder analysis on a FASTA file:
```bash
aiupred -i test.fasta
```

To predict **disorder, binding, and flexible linkers** (`-b` and `-l`), and save the results to a file (`-o`):
```bash
aiupred -i test.fasta -o results.tsv -b -l
```

To predict **redox-sensitive disorder** using original and C->S mutant profiles:
```bash
aiupred -i test.fasta -r
```

### Expected Output
```text
# Position	Residue	RedoxPlusDisorder	RedoxMinusDisorder	Region
#sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
1	M	0.8014	0.8014	0
2	E	0.8527	0.8527	0
3	E	0.8157	0.8157	0
4	P	0.8313	0.8313	0
5	Q	0.7959	0.7959	0
6	S	0.7855	0.7855	0
7	D	0.8402	0.8402	0
8	P	0.8788	0.8788	0
...
```

### Available CLI Options
```text
options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file in (multi) FASTA format (Required)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file
  -v, --verbose         Increase output verbosity
  -b, --binding         Predict binding using AIUPred-binding
  -l, --linker          Predict flexible linkers
  -r, --redox           Predict redox-sensitive disorder profiles and binary regions
  -g GPU, --gpu GPU     Index of GPU to use, default=0
  --force-cpu           Force the network to only utilize the CPU. Calculation will be very slow.
```

---

## <a name="api_usage">Programmatic Usage (Python API)</a>

AIUPred is designed to be highly memory-efficient. By initializing the `AIUPred` class, the heavy neural network models are loaded into your GPU (or CPU) exactly **once**, allowing you to process thousands of sequences rapidly.

*Note: AIUPred now **automatically handles memory chunking** for exceptionally long sequences under the hood. You no longer need to call separate "low memory" functions!*

```python
from aiupred import AIUPred, multifasta_reader

# 1. Initialize the predictor (Models are loaded into memory here)
predictor = AIUPred()

sequence = 'THISISATESTSEQUENCE'

# 2. Predict Disorder
disorder_propensities = predictor.predict_disorder(sequence)

# 3. Predict Binding
binding_propensities = predictor.predict_binding(sequence)

# 4. Predict Flexible Linkers
# Tip: Passing the pre-calculated arrays skips redundant neural network inference!
linker_propensities = predictor.predict_linker(
    sequence, 
    disorder_pred=disorder_propensities, 
    binding_pred=binding_propensities
)

# 5. Predict Redox-Sensitive Disorder (original vs C->S mutant)
redox_plus_disorder, redox_minus_disorder = predictor.predict_redox_profiles(sequence)
redox_regions_binary = predictor.predict_redox_region_binary(
    sequence,
    redox_plus_disorder=redox_plus_disorder,
    redox_minus_disorder=redox_minus_disorder
)

# 6. Extract Feature Embeddings
# Returns a 2D numpy array of shape (L, 32)
features_2d = predictor.get_embedding(sequence, center_only=True)

# Returns a 3D numpy array of shape (L, 101, 32) containing the full context windows
features_3d = predictor.get_embedding(sequence, center_only=False)
```

### Processing a Multi-FASTA File
```python
from aiupred import AIUPred, multifasta_reader

predictor = AIUPred()
sequences = multifasta_reader('test.fasta')

for header, seq in sequences.items():
    print(f"Processing {header}...")
    disorder = predictor.predict_disorder(seq)
    binding = predictor.predict_binding(seq)
    linker = predictor.predict_linker(seq, disorder_pred=disorder, binding_pred=binding)
```

---

## <a name="api_ref">API Reference</a>

### `class AIUPred(force_cpu=False, gpu_num=0)`
The core predictor class. Initializes the neural networks and manages the computation device.
- `force_cpu` *(bool)*: Force the models to run on CPU. (Default: `False`)
- `gpu_num` *(int)*: The index of the CUDA device to use. (Default: `0`)

#### `predict_disorder(sequence: str, apply_smoothing: bool = True) -> numpy.ndarray`
Predicts disorder propensities for a given amino acid sequence.
- `sequence` *(str)*: The amino acid sequence.
- `apply_smoothing` *(bool)*: Applies Savitzky-Golay filtering to the output if the sequence is >10 residues. (Default: `True`)

#### `predict_binding(sequence: str, apply_smoothing: bool = True) -> numpy.ndarray`
Predicts binding propensities for a given amino acid sequence.
- `sequence` *(str)*: The amino acid sequence.
- `apply_smoothing` *(bool)*: Applies Savitzky-Golay filtering to the output. (Default: `True`)

#### `predict_linker(sequence: str, apply_smoothing: bool = True, disorder_pred: Optional[numpy.ndarray] = None, binding_pred: Optional[numpy.ndarray] = None) -> numpy.ndarray`
Predicts flexible linker propensities by mathematically combining disorder and binding scores.
- `sequence` *(str)*: The amino acid sequence.
- `apply_smoothing` *(bool)*: Applies Savitzky-Golay filtering. (Default: `True`)
- `disorder_pred` *(Optional[numpy.ndarray])*: Pre-calculated disorder array to save computation time.
- `binding_pred` *(Optional[numpy.ndarray])*: Pre-calculated binding array to save computation time.

#### `mutate_c_to_s(sequence: str) -> str`
Returns the C->S mutated sequence used for redox-minus calculations.
- `sequence` *(str)*: The amino acid sequence.

#### `get_redox_regions(redox_plus_values: numpy.ndarray, redox_minus_values: numpy.ndarray) -> dict`
Detects redox-sensitive region boundaries using differences between original and C->S mutant disorder profiles.
- `redox_plus_values` *(numpy.ndarray)*: Original sequence disorder propensities.
- `redox_minus_values` *(numpy.ndarray)*: C->S mutant disorder propensities.
- **Returns:** A dictionary of `{region_start: region_end}` boundaries (0-based, inclusive).

#### `predict_redox_profiles(sequence: str) -> tuple[numpy.ndarray, numpy.ndarray]`
Predicts redox plus/minus disorder profiles.
- `sequence` *(str)*: The amino acid sequence.
- **Returns:** `(redox_plus_disorder, redox_minus_disorder)` where plus is original and minus is C->S mutant.

#### `predict_redox_region_binary(sequence: str, redox_plus_disorder: Optional[numpy.ndarray] = None, redox_minus_disorder: Optional[numpy.ndarray] = None) -> numpy.ndarray`
Predicts a binary redox region annotation per residue (`1` in region, else `0`).
- `sequence` *(str)*: The amino acid sequence.
- `redox_plus_disorder` *(Optional[numpy.ndarray])*: Optional pre-calculated original disorder profile.
- `redox_minus_disorder` *(Optional[numpy.ndarray])*: Optional pre-calculated C->S mutant disorder profile.

#### `get_embedding(sequence: str, center_only: bool = True, chunk_len: int = 1000) -> numpy.ndarray`
Extracts high-dimensional feature embeddings from the sequence.
- `sequence` *(str)*: The amino acid sequence.
- `center_only` *(bool)*: If `True`, returns the center residue's 32-dimensional embedding with shape `(L, 32)`. If `False`, returns the full sliding window context with shape `(L, 101, 32)`. (Default: `True`)
- `chunk_len` *(int)*: Sequence chunk length to prevent out-of-memory errors on large proteins. (Default: `1000`)

### Helper Functions

#### `multifasta_reader(file_path: str) -> dict`
Utility function to parse a FASTA file.
- **Returns:** A dictionary mapping `>header` strings to their corresponding `sequence` strings.

---

## <a name="benchmark">Benchmarks</a>

|     | Type            | Single sequence | Human proteome          |   
|-----|-------------|-----------------|-------------------------|
| GPU | 1080 Ti 12G  | 3 sec           | **100 proteins/second** |
| CPU | Xeon E3-1270 v5 | **1.7 sec** | 3.5 proteins/second     | 

**GPU memory usage:**

| VRAM (GB) | Residues |   
|-----------|----------|
| 2         | 3000     |
| 6         | 8000     | 
| 12        | 16000    |  

---

##  <a name="license">License</a>

This project is licensed under the MIT License. See the LICENSE file for details.

---

## <a name="cite">How to cite</a>

### AIUPred

<pre>@article{10.1093/nar/gkae385,
    author = {Erdős, Gábor and Dosztányi, Zsuzsanna},
    title = {AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder},
    journal = {Nucleic Acids Research},
    volume = {52},
    number = {W1},
    pages = {W176-W181},
    year = {2024},
    month = {05},
    abstract = {Intrinsically disordered proteins and protein regions (IDPs/IDRs) carry out important biological functions without relying on a single well-defined conformation. As these proteins are a challenge to study experimentally, computational methods play important roles in their characterization. One of the commonly used tools is the IUPred web server which provides prediction of disordered regions and their binding sites. IUPred is rooted in a simple biophysical model and uses a limited number of parameters largely derived on globular protein structures only. This enabled an incredibly fast and robust prediction method, however, its limitations have also become apparent in light of recent breakthrough methods using deep learning techniques. Here, we present AIUPred, a novel version of IUPred which incorporates deep learning techniques into the energy estimation framework. It achieves improved performance while keeping the robustness of the original method. Based on the evaluation of recent benchmark datasets, AIUPred scored amongst the top three single sequence based methods. With a new web server we offer fast and reliable visual analysis for users as well as options to analyze whole genomes in mere seconds with the downloadable package. AIUPred is available at https://aiupred.elte.hu.},
    issn = {0305-1048},
    doi = {10.1093/nar/gkae385},
    url = {https://doi.org/10.1093/nar/gkae385},
    eprint = {https://academic.oup.com/nar/article-pdf/52/W1/W176/58435879/gkae385.pdf},
}</pre>

### AIUPred-binding

<pre>@article{ERDOS2025169071,
title = {AIUPred – Binding: Energy Embedding to Identify Disordered Binding Regions},
journal = {Journal of Molecular Biology},
volume = {437},
number = {15},
pages = {169071},
year = {2025},
note = {Computation Resources for Molecular Biology},
issn = {0022-2836},
doi = {https://doi.org/10.1016/j.jmb.2025.169071},
url = {https://www.sciencedirect.com/science/article/pii/S0022283625001378},
author = {Gábor Erdős and Norbert Deutsch and Zsuzsanna Dosztányi},
keywords = {Protein disorder, Functional disorder, Disorder binding prediction, Functional disorder prediction, Energy embedding},
abstract = {Intrinsically disordered regions (IDRs) play critical roles in various cellular processes, often mediating interactions through disordered binding regions that transition to ordered states. Experimental characterization of these functional regions is highly challenging, underscoring the need for fast and accurate computational tools. Despite their importance, predicting disordered binding regions remains a significant challenge due to limitations in existing datasets and methodologies. In this study, we introduce AIUPred-binding, a novel prediction tool leveraging a high dimensional mathematical representation of structural energies – we call energy embedding – and pathogenicity scores from AlphaMissense. By employing a transfer learning approach, AIUPred-binding demonstrates improved accuracy in identifying functional sites within IDRs. Our results highlight the tool’s ability to discern subtle features within disordered regions, addressing biases and other challenges associated with manually curated datasets. We present AIUPred-binding integrated into the AIUPred web framework as a versatile and efficient resource for understanding the functional roles of IDRs. AIUPred-binding is freely accessible at https://aiupred.elte.hu.}
}</pre>

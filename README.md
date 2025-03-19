# AIUPred

[About](#about)

[How to install](#install)

[How to run](#single_pred)

[AIUPred as an importable library](#multi_pred)

[AIUPred-binding][#binding]

[Examples](#examples)

[Functions](#functions)

[Benchmarks](#benchmark)

[License](#license)


## <a name="about">About</a>

Intrinsically disordered proteins (IDPs) have no single well-defined tertiary structure under native conditions. AIUPred is a tool that allows to identify disordered protein regions created by Zsuzsanna Dosztányi and Gábor Erdős. 

AIUPred contains a standalone executable script as well as and importable python library.

AIUPred is also available as a webserver (https://aiupred.elte.hu/)

For more information please refer to the publication: [AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkae385/7673484)

### Requirements

```
torch~=2.0.1
numpy~=1.26.0
scipy~=1.13.0
```

## <a name="install">How to install</a>

Clone the repository: `git clone https://github.com/doszilab/AIUPred`

Change the working directory to the downloaded folder

<b>It is recommended to use a virtual environment</b>

After the environment is ready install the required libraries:

`pip3 install -r requirements.txt`

## <a name="single_pred">How to run</a>

AIUPred contains an executable python script which calls the supplied library as 
well as a FASTA formatted file containing two protein sequences.

In order to carry out an analysis use the following command

`python3 aiupred.py -i test.fasta`

Expected output:

```
>sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
1       M       0.8014
2       E       0.8527
3       E       0.8157
4       P       0.8313
5       Q       0.7959
6       S       0.7855
7       D       0.8402
8       P       0.8788
...
```

Available options for the executable are the following:

```
-h, --help            show this help message and exit
-i INPUT_FILE, --input_file INPUT_FILE
Input file in (multi) FASTA format
-o OUTPUT_FILE, --output_file OUTPUT_FILE
Output file
-b, --binding         Predict binding using AIUPred-binding
-v, --verbose         Increase output verbosity
-g GPU, --gpu GPU     Index of GPU to use, default=0
--force-cpu           Force the network to only utilize the CPU. Calculation will be very slow, not recommended
```

## <a name="multi_pred">Programmatic usage</a>

AIUPred contains a loadable python library. The following section gives some tips how to use the importable library. First download and extract AIUPred and add its location to your PYTHONPATH environment variable (assuming standard bash shell)

`export PYTHONPATH="${PYTHONPATH}:/path/to/aiupred/folder"`

After reloading the shell AIUPred will be importable in your python scripts.

The simplest way to execute AIUPred from the library is to call the `aiupred_disorder()` function. This loads all the required network data and executes the prediction.

```python
import aiupred_lib
sequence = 'THISISATESTSEQENCE'
# Predict the disorder profile using AIUPred
aiupred_lib.aiupred_disorder(sequence)
```

In order to analyze multiple sequences it is recommended to load the network data into memory and keep it there.

```python
import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models('disorder')
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.predict_disorder(sequence, embedding_model, regression_model, device,
                                          smoothing=True)
```

The bottleneck of the method in terms of speed is the loading of `embedding_model` and `regression_model` 
so by keeping them in memory we can speed up the analysis significantly for future proteins. 

Depending on the length of the sequence AIUPred can use an 
excessive amount of memory. In case you run out of memory AIUPred
is supplied with an approximation function which uses variable length
chunks of the input protein to save memory, then concatenates the results

```python
import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models('disorder')
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.low_memory_predict_disorder(sequence, embedding_model, regression_model, device,
                                          smoothing=True)
```
Please note, that the results of the low memory version differ from the original!

## <a name="binding">AIUPred-binding</a>

AIUPred-binding is also a part of the downloadable package. For command line usage use the `-b` flag:

`python3 aiupred.py -i test.fasta -b`

Expected output:

```
>sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
1       M       0.8669
2       E       0.6927
3       E       0.5133
4       P       0.3790
5       Q       0.3011
6       S       0.2671
7       D       0.2229
8       P       0.2326
...
```

### AIUPred-binding programmatic usage:

In order to use AIUPred-binding programmatically the functions and their usage is identical to disorder prediction:


```python
import aiupred_lib
sequence = 'THISISATESTSEQENCE'
# Predict the disorder profile using AIUPred
aiupred_lib.aiupred_binding(sequence)
```

In order to analyze multiple sequences it is recommended to load the network data into memory and keep it there.

```python
import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models('binding')
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.predict_binding(sequence, embedding_model, regression_model, device,
                                          smoothing=True)
```

For low memory prediction use:

```python
import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models('binding')
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.low_memory_predict_binding(sequence, embedding_model, regression_model, device,
                                          smoothing=True)
```
Please note, that the results of the low memory version differ from the original!

## <a name="examples">Examples</a>

Example scripts can be found in the 'examples' library

## <a name="functions">Functions</a>

`aiupred_disorder(sequence, force_cpu=False, gpu_num=0)`

Predicts disorder propensities for a given amino acid sequence.

- `sequence`: Amino acid sequence as a string.
- `force_cpu`: Force the method to run on CPU only mode (default: False).
- `gpu_num`: Index of the GPU to use (default: 0).

`aiupred_binding(sequence, force_cpu=False, gpu_num=0)`

Predicts binding propensities for a given amino acid sequence.

- `sequence`: Amino acid sequence as a string.
- `force_cpu`: Force the method to run on CPU only mode (default: False).
- `gpu_num`: Index of the GPU to use (default: 0).

`main(multifasta_file, force_cpu=False, gpu_num=0, binding=False)`

Predicts disorder or binding propensities for sequences in a FASTA file.

- `multifasta_file`: Location of (multi) FASTA formatted sequences.
- `force_cpu`: Force the method to run on CPU only mode (default: False).
- `gpu_num`: Index of the GPU to use (default: 0).
- `binding`: Predict binding propensities if True, otherwise predict disorder propensities (default: False).

### Models

`TransformerModel`

Transformer model to estimate positional contact potential from an amino acid sequence.

`BindingTransformerModel`

Transformer model for binding prediction.

`BindingDecoderModel`

Decoder model for binding prediction.

`DecoderModel`

Regression model to estimate disorder propensity from an energy tensor.

### Helper Functions

`tokenize(sequence, device)`

Tokenizes an amino acid sequence.

`predict_disorder(sequence, energy_model, regression_model, device, smoothing=None)`

Predicts disorder propensity from a sequence using a transformer and a regression model.

`calculate_energy(sequence, energy_model, device)`

Calculates residue energy from a sequence using a transformer network.

`predict_binding(sequence, embedding_model, decoder_model, device, smoothing=None, energy_only=False, binding=False)`

Predicts binding propensity from a sequence using a transformer and a decoder model.

`low_memory_predict_disorder(sequence, embedding_model, decoder_model, device, smoothing=None, chunk_len=1000)`

Predicts disorder propensity for long sequences using a low memory approach.

`low_memory_predict_binding(sequence, embedding_model, decoder_model, device, smoothing=None, chunk_len=1000)`

Predicts binding propensity for long sequences using a low memory approach.

`binding_transform(prediction, smoothing=True)`

Transforms binding predictions.

`multifasta_reader(file_handler)`

Reads sequences from a (multi) FASTA file.

`init_models(prediction_type, force_cpu=False, gpu_num=0)`

Initializes networks and device to run on.


## <a name="benchmark">Benchmarks</a>

|     | Type            | Single sequence | Human proteome          |   
|-----|-------------|-----------------|-------------------------|
| GPU | 1080 Ti 12G  | 3 sec           | **100 proteins/second** |
| CPU | Xeon E3-1270 v5 | **1.7 sec**     | 3.5 proteins/second     | 


GPU memory usage:

| VRAM (GB) | Residues |   
|-----------|----------|
| 2         | 3000     |
| 6         | 8000     | 
| 12        | 16000    |  

##  <a name="license">License</a>

This project is licensed under the MIT License. See the LICENSE file for details.
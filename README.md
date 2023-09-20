# AIUPred disorder prediction method v0.9

For in-house use only

## Architecture

AIUPred utilizes the original IUPred force-field.
The force-field is than used to calculate the energies 
of globular proteins (17282 proteins with 4276509 residues), which is than 
estimated using an encoder-only transformer network with
a fully connected decoder. 

The input of the transformer is an integer tokenized sequence
in a -50+50 window. To account for N and C terminal positions
an X encoding residues is added. All non-standard residues are also
substituted to X.

The estimated energies are than used to predict disorder propensity
using the DisProt database and a set of fully connected layers. 
The energy values are padded with zeroes and unfolded into
100 long windows. 


## Requirements

```
torch~=2.0.1
numpy~=1.26.0
```

## Installation

It is recommended to use a virtual environment

Install the required libraries:

`pip3 install -r requirements.txt`

## Analysis

`python3 aiupred.py -i test.fasta`

Expected output:

```
# AIUPred v0.9
# Gabor Erdos, Zsuzsanna Dosztanyi
# For in house use only

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

## Options

```
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file in (multi) FASTA format
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file
  -v, --verbose         Increase output verbosity
  -g GPU, --gpu GPU     Index of GPU to use, default=0
  --force-cpu           Force the network to only utilize the CPU. Calculation will be very slow, not recommended

```

## Benchmarks

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

## Testing

![plot](./imgs/P04637.png)

![plot](./imgs/P35222.png)

v0.9 has been tested on the following interpreters:

```
python 3.10.12
```
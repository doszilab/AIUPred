# AIUPred disorder prediction method v0.9

For in-house use only

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
  -v, --verbose         Increase output verbosity
  -g GPU, --gpu GPU     Index of GPU to use, default=0
  --force-cpu           Force the network to only utilize the CPU. Calculation will be very slow, not recommended
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file in (multi) FASTA format
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file
```

## Testing

![plot](./imgs/P04637.png)

![plot](./imgs/P35222.png)

v0.9 has been tested on the following interpreters:

```
python 3.10.12
```
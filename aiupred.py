import argparse
import logging
import aiupred_lib

parser = argparse.ArgumentParser(
    description='AIUPred disorder prediction method v0.9\n'
                'Developed by Gabor Erdos and Zsuzsanna Dosztanyi',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("-i", "--input_file",
                    help="Input file in (multi) FASTA format",
                    required=True)
parser.add_argument("-o", "--output_file",
                    help="Output file")
parser.add_argument("-v", "--verbose",
                    help="Increase output verbosity",
                    action="store_true")
parser.add_argument("-g", "--gpu",
                    help="Index of GPU to use, default=0",
                    default=0)
parser.add_argument("--force-cpu",
                    help="Force the network to only utilize the CPU. Calculation will be very slow, not recommended",
                    action="store_true")

args = parser.parse_args()
# logging.basicConfig(format='%(asctime)s | %(message)s')
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='# %(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

try:
    open(args.input_file)
except FileNotFoundError as exc:
    print('{}: {}'.format(type(exc).__name__, exc))


output_str = '''# AIUPred v0.9
# Gabor Erdos, Zsuzsanna Dosztanyi
# For in house use only\n\n'''
for ident, results in aiupred_lib.main(args.input_file,
                                       force_cpu=args.force_cpu,
                                       gpu_num=args.gpu).items():
    output_str += ident + '\n'
    for pos, value in enumerate(results['aiupred']):
        output_str += f'{pos+1}\t{results["sequence"][pos]}\t{value:.4f}\n'
    output_str += '\n\n'

if args.output_file:
    with open(args.output_file, 'w') as file_handler:
        file_handler.write(output_str.strip())
else:
    print(output_str.strip())

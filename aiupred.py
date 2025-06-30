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
                    required=True, type=argparse.FileType('r', encoding='UTF-8'))
parser.add_argument("-o", "--output_file",
                    help="Output file")
parser.add_argument("-v", "--verbose",
                    help="Increase output verbosity",
                    action="store_true")
parser.add_argument("-b", "--binding",
                    help="Predict binding using AIUPred-binding",
                    action="store_true")
parser.add_argument("-g", "--gpu",
                    help="Index of GPU to use, default=0",
                    default=0)
parser.add_argument("--force-cpu",
                    help="Force the network to only utilize the CPU. Calculation will be very slow, not recommended",
                    action="store_true")

args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='# %(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


output_str = r'''#             _____ _    _ _____              _ 
#       /\   |_   _| |  | |  __ \            | |
#      /  \    | | | |  | | |__) | __ ___  __| |
#     / /\ \   | | | |  | |  ___/ '__/ _ \/ _` |
#    / ____ \ _| |_| |__| | |   | | |  __/ (_| |
#   /_/    \_\_____|\____/|_|   |_|  \___|\__,_|
#
# Gabor Erdos, Zsuzsanna Dosztanyi
# Version 2.0
# AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder
# Nucleic Acids Res. 2024 Jul 5;52(W1):W176-W181.doi: 10.1093/nar/gkae385. 
#
# AIUPred - Binding: energy embedding to identify disordered binding regions
# Journal of Molecular Biology 2025
'''
print(output_str)
if not args.output_file:
    output_str = ''
output_str += "# Position\tResidue\tDisorder"
if args.binding:
    output_str += '\tBinding'
output_str += '\n'
result_text = []
logging.info('Starting analysis')
for ident, results in aiupred_lib.main(args.input_file,
                                       force_cpu=args.force_cpu,
                                       gpu_num=args.gpu,
                                       binding=args.binding).items():
    result_text.append('#' + ident + '\n')
    for pos, value in enumerate(results['aiupred']):
        result_text.append(f'{pos+1}\t{results["sequence"][pos]}\t{value:.4f}')
        if args.binding:
            result_text.append(f'\t{results["aiupred-binding"][pos]:.4f}')
        result_text.append('\n')
    result_text.append('\n\n')
logging.info('Analysis done, writing output')
if args.output_file:
    with open(args.output_file, 'w') as file_handler:
        file_handler.write(output_str.strip() + '\n' + ''.join(result_text))
else:
    print(output_str.strip() + '\n' + ''.join(result_text))

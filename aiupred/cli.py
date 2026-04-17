import argparse
import logging
import sys
from .predictor import AIUPred
from .utils import multifasta_reader

def main():
    parser = argparse.ArgumentParser(
        description='AIUPred disorder prediction method v2.0\n'
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
    parser.add_argument("-l", "--linker",
                        help="Predict flexible linkers",
                        action="store_true")
    parser.add_argument("-r", "--redox",
                        help="Predict redox-sensitive disorder profiles and binary regions",
                        action="store_true")
    parser.add_argument("-g", "--gpu",
                        help="Index of GPU to use, default=0",
                        type=int, default=0)
    parser.add_argument("--force-cpu",
                        help="Force the network to only utilize the CPU. Calculation will be very slow.",
                        action="store_true")

    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='# %(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format='# %(message)s')

    banner = r'''#             _____ _    _ _____              _ 
#       /\   |_   _| |  | |  __ \            | |
#      /  \    | | | |  | | |__) | __ ___  __| |
#     / /\ \   | | | |  | |  ___/ '__/ _ \/ _` |
#    / ____ \ _| |_| |__| | |   | | |  __/ (_| |
#   /_/    \_\_____|\____/|_|   |_|  \___|\__,_|
#
#
# Gabor Erdos, Zsuzsanna Dosztanyi
# Version 2.1.0
# AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder
# Nucleic Acids Res. 2024 Jul 5;52(W1):W176-W181.doi: 10.1093/nar/gkae385. 
#
# AIUPred - Binding: energy embedding to identify disordered binding regions
# Journal of Molecular Biology 2025
'''
    print(banner)

    # 1. Read sequences
    logging.info('Reading FASTA file...')
    sequences = multifasta_reader(args.input_file)
    if not sequences:
        logging.error("FASTA file is empty or invalid.")
        sys.exit(1)
    
    # 2. Initialize the Predictor ONCE
    logging.info('Initializing AIUPred networks...')
    predictor = AIUPred(force_cpu=args.force_cpu, gpu_num=args.gpu)

    # 3. Setup output formatting
    output_lines = []
    if args.redox:
        header_line = "# Position\tResidue\tRedoxPlusDisorder\tRedoxMinusDisorder\tRegion"
    else:
        header_line = "# Position\tResidue\tDisorder"

    if args.binding:
        header_line += "\tBinding"

    if args.linker:
        header_line += "\tLinker"

    if args.output_file:
        output_lines.append(header_line)

    # 4. Run predictions
    logging.info(f'Starting analysis on {len(sequences)} sequences...')
    for num, (ident, sequence) in enumerate(sequences.items(), start=1):
        logging.debug(f'Processing {num}/{len(sequences)}: {ident}')
        
        # Clean sequence (remove newlines if any snuck in)
        sequence = "".join(sequence.split())
        
        disorder_preds = None
        redox_plus_preds = None
        redox_minus_preds = None
        redox_region_binary = None
        if args.redox:
            redox_plus_preds, redox_minus_preds = predictor.predict_redox_profiles(sequence)
            disorder_preds = redox_plus_preds
            redox_region_binary = predictor.predict_redox_region_binary(
                sequence,
                redox_plus_disorder=redox_plus_preds,
                redox_minus_disorder=redox_minus_preds
            )
        else:
            disorder_preds = predictor.predict_disorder(sequence)
        
        binding_preds = None
        if args.binding:
            binding_preds = predictor.predict_binding(sequence)

        # Calculate linker (Passes the pre-calculated arrays to save huge amounts of time!)
        linker_preds = None
        if args.linker:
            linker_preds = predictor.predict_linker(
                sequence, 
                disorder_pred=disorder_preds, 
                binding_pred=binding_preds
            )

        output_lines.append(f'#{ident}')
        
        for pos, res in enumerate(sequence):
            if args.redox:
                line = (
                    f'{pos+1}\t{res}\t{redox_plus_preds[pos]:.4f}\t'
                    f'{redox_minus_preds[pos]:.4f}\t{int(redox_region_binary[pos])}'
                )
            else:
                line = f'{pos+1}\t{res}\t{disorder_preds[pos]:.4f}'
            if args.binding:
                line += f'\t{binding_preds[pos]:.4f}'
            if args.linker:
                line += f'\t{linker_preds[pos]:.4f}'
            output_lines.append(line)
            
        output_lines.append('') # Add a blank line between sequences

    # 5. Write or Print Output
    logging.info('Analysis done, writing output...')
    final_output = "\n".join(output_lines)
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(banner + "\n" + final_output + "\n")
    else:
        print(header_line)
        print(final_output)

if __name__ == '__main__':
    main()

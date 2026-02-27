import os
from aiupred import AIUPred, multifasta_reader

def main():
    print("Initializing AIUPredictor...")
    # 1. Initialize the class. Models are loaded into memory exactly ONCE here.
    # Set force_cpu=True if you are testing on a laptop without a GPU
    predictor = AIUPred(force_cpu=False) 

    # ==========================================
    # Example 1: Single Sequence Analysis (P53)
    # ==========================================
    print("\n--- Example 1: Single Sequence Analysis ---")
    sequence = """MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
    DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
    SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
    RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS
    SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP
    PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG
    GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"""
    
    # Clean the sequence (removes spaces and newlines)
    sequence = "".join(sequence.split())
    
    print("Predicting disorder for P53...")
    disorder_propensities = predictor.predict_disorder(sequence)
    print(f"Disorder propensities (first 5 AA): {disorder_propensities[:5]}")

    print("Predicting binding for P53...")
    binding_propensities = predictor.predict_binding(sequence)
    print(f"Binding propensities (first 5 AA):  {binding_propensities[:5]}")

    # ==========================================
    # Example 2: Multi-FASTA Batch Processing
    # ==========================================
    print("\n--- Example 2: Multi-FASTA Batch Processing ---")
    
    # Let's create a temporary dummy FASTA file for this example to run
    dummy_fasta_path = "test_sequences.fasta"
    with open(dummy_fasta_path, "w") as f:
        f.write(">Seq1_Short\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">Seq2_P53_Snippet\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVL\n")

    print(f"Reading sequences from {dummy_fasta_path}...")
    fasta_dict = multifasta_reader(dummy_fasta_path)

    for header, seq in fasta_dict.items():
        print(f"\nProcessing {header} (Length: {len(seq)})...")
        
        # Notice how fast this is because we don't have to reload the models!
        disorder_preds = predictor.predict_disorder(seq)
        binding_preds = predictor.predict_binding(seq)
        
        # Just printing the average score for the sequence as an example
        print(f"  -> Average Disorder: {disorder_preds.mean():.4f}")
        print(f"  -> Average Binding:  {binding_preds.mean():.4f}")

    # Clean up the dummy file
    if os.path.exists(dummy_fasta_path):
        os.remove(dummy_fasta_path)
        print("\nCleaned up test FASTA file.")

if __name__ == '__main__':
    main()

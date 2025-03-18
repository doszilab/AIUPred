import aiupred_lib

# Sequence of human P53 protein
sequence = """MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE
RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS
SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP
PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG
GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"""
sequence = "".join(sequence.splitlines())

analysis_type = 'disorder' # Change this to 'binding' for binding prediction
# Create the models and store them so they are only calculated once
embedding_model, reg_model, device = aiupred_lib.init_models(analysis_type)

# Use low_memory_predict_binding for AIUPred-binding
prediction = aiupred_lib.low_memory_predict_disorder(sequence, embedding_model, reg_model, device)
print(prediction)
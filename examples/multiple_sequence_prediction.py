import aiupred_lib
from aiupred_lib import aiupred_binding

# Read the test sequences provided
with open("test.fasta") as file_handler:
    sequences = aiupred_lib.multifasta_reader(file_handler)

analysis_type = 'disorder' # Change this to 'binding' for binding prediction
# Create the models and store them so they are only calculated once
embedding_model, reg_model, device = aiupred_lib.init_models(analysis_type)

for header, sequence in sequences.items():
    # Use predict_binding for AIUPred-binding prediction
    prediction = aiupred_lib.predict_disorder(sequence, embedding_model, reg_model, device)
    print(header, prediction)

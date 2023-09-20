import logging

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad
import math
import os

PATH = os.path.dirname(os.path.realpath(__file__))
AA_CODE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
WINDOW = 100


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer network
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer model to estimate positional contact potential from an amino acid sequence
    """
    def __init__(self):
        super().__init__()
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 256, 0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (Batch x Window+1 x Embed_dim
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = torch.flatten(embedding, 1)
        output = self.decoder(output)
        return torch.squeeze(output)


class RegModel(nn.Module):
    """
    Regression model to estimate disorder propensity from and energy tensor
    """
    def __init__(self):
        super().__init__()
        self.forward_layer = nn.Linear((WINDOW + 1), 8)
        self.fc_2 = nn.Linear(8, 4)
        self.fc_3 = nn.Linear(4, 1)

    def forward(self, src: Tensor) -> Tensor:
        x = torch.relu(self.forward_layer(src))
        x = torch.relu(self.fc_2(x))
        output = torch.sigmoid(self.fc_3(x))
        return torch.squeeze(output)


def tokenize(sequence, device):
    """
    Tokenize an amino acid sequence. Non-standard amino acids are treated as X
    :param sequence: Amino acid sequence in string
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tokenized tensors
    """
    return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence]).to(device)


def predict_disorder(sequence, energy_model, regression_model, device):
    """
    Predict disorder propensity from a sequence using a transformer and a regression model
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param regression_model: regression model
    :param device: Device to run on. CUDA{x} or CPU
    :return:
    """
    predicted_energies = calculate_energy(sequence, energy_model, device)
    padded_energies = pad(predicted_energies, (WINDOW // 2, WINDOW // 2), 'constant', 0)
    unfolded_energies = padded_energies.unfold(0, WINDOW + 1, 1)
    return regression_model(unfolded_energies).detach().cpu().numpy()


def calculate_energy(sequence, energy_model, device):
    """
    Calculates residue energy from a sequence using a transformer network
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tensor of energy values
    """
    tokenized_sequence = tokenize(sequence, device)
    padded_token = pad(tokenized_sequence, (WINDOW // 2, WINDOW // 2), 'constant', 20)
    unfolded_tokens = padded_token.unfold(0, WINDOW + 1, 1)
    return energy_model(unfolded_tokens)


def multifasta_reader(file_location):
    """
    (multi) FASTA reader function
    :param file_location: Location of (multi) FASTA formatted file
    :return: Dictionary with header -> sequence mapping from the file
    """
    sequence_dct = {}
    header = None
    with open(file_location) as file_handler:
        for line in file_handler:
            if line.startswith('>'):
                header = line.strip()
                sequence_dct[header] = ''
            elif line.strip():
                sequence_dct[header] += line.strip()
    return sequence_dct


def aiupred_disorder(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Numpy array with disorder propensities for each position
    """
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    if force_cpu:
        device = 'cpu'
    logging.debug(f'Running on {device}')
    if device == 'cpu':
        print('# Warning: No GPU found, running on CPU. It is advised to run AIUPred on a GPU')

    embedding_model = TransformerModel()
    embedding_model.load_state_dict(torch.load(f'{PATH}/data/embedding.pt', map_location=device))
    embedding_model.to(device)
    embedding_model.eval()

    reg_model = RegModel()
    reg_model.load_state_dict(torch.load(f'{PATH}/data/regression.pt', map_location=device))
    reg_model.to(device)
    reg_model.eval()

    return predict_disorder(sequence, embedding_model, reg_model, device)


def main(multifasta_file, force_cpu=False, gpu_num=0):
    """
    Main function to be called from aiupred.py
    :param multifasta_file: Location of (multi) FASTA formatted sequences
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Dictionary with parsed sequences and predicted results
    """
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    if force_cpu:
        device = 'cpu'
    logging.debug(f'Running on {device}')
    if device == 'cpu':
        print('# Warning: No GPU found, running on CPU. It is advised to run AIUPred on a GPU')

    embedding_model = TransformerModel()
    embedding_model.load_state_dict(torch.load(f'{PATH}/data/embedding.pt', map_location=device))
    embedding_model.to(device)
    embedding_model.eval()

    reg_model = RegModel()
    reg_model.load_state_dict(torch.load(f'{PATH}/data/regression.pt', map_location=device))
    reg_model.to(device)
    reg_model.eval()

    logging.debug("Networks initialized")
    sequences = multifasta_reader(multifasta_file)
    logging.debug("Sequences read")
    if not sequences:
        raise ValueError("FASTA file is empty")
    results = {}
    logging.StreamHandler.terminator = ""
    for num, (ident, sequence) in enumerate(sequences.items()):
        results[ident] = {}
        results[ident]['aiupred'] = predict_disorder(sequence, embedding_model, reg_model, device)
        results[ident]['sequence'] = sequence
        logging.debug(f'{num}/{len(sequences)} sequences done...\r')
    logging.StreamHandler.terminator = '\n'
    logging.debug(f'Analysis done, writing output')
    return results

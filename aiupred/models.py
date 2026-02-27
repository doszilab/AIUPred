import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


AA_CODE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
WINDOW = 100
class PositionalEncoding(nn.Module):
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

    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 64, 0, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(len(AA_CODE), self.d_model)
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


class DecoderModel(nn.Module):
    def __init__(self, pred_type):
        super().__init__()
        input_dim = (WINDOW + 1) * (WINDOW + 1) * 32
        output_dim = 1
        current_dim = input_dim
        if pred_type == 'disorder':
            layer_architecture = [64, 64, 64, 64, 16]
        else:
            layer_architecture = [64, 64, 64, 16]
        self.layers = nn.ModuleList()
        for hdim in layer_architecture:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.layers[-1](x))
        return torch.squeeze(output)


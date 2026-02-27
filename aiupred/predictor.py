import os
import torch
import numpy as np
import logging
from torch.nn.functional import pad
from .models import TransformerModel, DecoderModel, AA_CODE, WINDOW

# Define the path to the weights relative to this file
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODULE_DIR, 'data')

class AIUPred:
    def __init__(self, force_cpu: bool = False, gpu_num: int = 0):
        self._setup_device(force_cpu, gpu_num)
        self._load_models()

    def _setup_device(self, force_cpu: bool, gpu_num: int):
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            if not force_cpu:
                logging.warning('No GPU found. Running on CPU is slower.')
        else:
            self.device = torch.device(f'cuda:{gpu_num}')

    def _load_models(self):
        self.embedding_model = TransformerModel().to(self.device)
        self.embedding_model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'embedding.pt'), map_location=self.device, weights_only=True))
        self.embedding_model.eval()

        self.disorder_decoder = DecoderModel('disorder').to(self.device)
        self.disorder_decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'disorder_decoder.pt'), map_location=self.device, weights_only=True))
        self.disorder_decoder.eval()

        self.binding_decoder = DecoderModel('binding').to(self.device)
        self.binding_decoder.load_state_dict(torch.load(os.path.join(DATA_DIR, 'binding_decoder.pt'), map_location=self.device, weights_only=True))
        self.binding_decoder.eval()

    def predict_disorder(self, sequence: str, smoothing=None) -> np.ndarray:
        return self._run_inference(sequence, self.disorder_decoder, smoothing)

    def predict_binding(self, sequence: str, smoothing=None) -> np.ndarray:
        return self._run_inference(sequence, self.binding_decoder, smoothing)

    def _tokenize(self, sequence: str) -> torch.Tensor:
        # Replaces your standalone tokenize function
        return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence], device=self.device)

    @torch.no_grad()
    def _forward_pass(self, sequence: str, decoder_model: torch.nn.Module, smoothing=None) -> np.ndarray:
        """The core tensor operations for a single chunk."""
        _tokens = self._tokenize(sequence)
        _padded_token = pad(_tokens, (WINDOW // 2, WINDOW // 2), 'constant', 0)
        _unfolded_tokens = _padded_token.unfold(0, WINDOW + 1, 1)
        
        _token_embedding = self.embedding_model(_unfolded_tokens, embed_only=True)
        
        _padded_embed = pad(_token_embedding, (0, 0, 0, 0, WINDOW // 2, WINDOW // 2), 'constant', 0)
        _unfolded_embedding = _padded_embed.unfold(0, WINDOW + 1, 1)
        _decoder_input = _unfolded_embedding.permute(0, 2, 1, 3)
        
        _prediction = decoder_model(_decoder_input).cpu().numpy()
        
        if smoothing:
            _prediction = smoothing(_prediction)
        return _prediction

    def _run_inference(self, sequence: str, decoder_model: torch.nn.Module, smoothing=None, chunk_len=1000) -> np.ndarray:
        """Handles routing to standard or low-memory execution automatically."""
        overlap = 100
        if len(sequence) <= chunk_len:
            return self._forward_pass(sequence, decoder_model, smoothing)
        
        # Low-memory logic automatically kicks in for long sequences
        overlapping_predictions = []
        for chunk_start in range(0, len(sequence), chunk_len - overlap):
            chunk_seq = sequence[chunk_start:chunk_start + chunk_len]
            overlapping_predictions.append(self._forward_pass(chunk_seq, decoder_model, smoothing=None))
            
        prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
        
        if smoothing:
            prediction = smoothing(prediction)
        return prediction

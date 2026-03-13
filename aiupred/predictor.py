import os
import torch
import numpy as np
import logging
from torch.nn.functional import pad
from .models import TransformerModel, DecoderModel, AA_CODE, WINDOW
from scipy.signal import savgol_filter

# Define the path to the weights relative to this file
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODULE_DIR, 'data')

class AIUPred:
    def __init__(self, force_cpu: bool = False, gpu_num: int = 0):
        self._setup_device(force_cpu, gpu_num)
        self._load_models()
        self._load_binding_transform()

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

    @staticmethod
    def _apply_disorder_cutoff(raw_score: np.ndarray) -> np.ndarray:
        """
        Transforms a raw prediction score into a calibrated probability 
        where 0.5 is the optimal binarization cutoff.
        """
        beta_0 = -6.04370
        beta_1 = 11.41083
        z = beta_0 + (beta_1 * raw_score)
        return 1 / (1 + np.exp(-z))

    def _load_binding_transform(self):
        """Loads the binding transform mapping into memory once."""
        self.transform_dict = {}
        transform_path = os.path.join(DATA_DIR, 'binding_transform')
        
        with open(transform_path) as fn:
            for line in fn:
                key, value = line.strip().split()
                self.transform_dict[int(float(key) * 1000)] = float(value)
                
        # Create a vectorized function for blazingly fast numpy array mapping
        self.vectorized_transform = np.vectorize(self.transform_dict.get)

    def _apply_binding_transform(self, prediction: np.ndarray, apply_smoothing: bool) -> np.ndarray:
        """Applies the lookup table mapping and optional Savitzky-Golay filtering."""
        rounded_pred = np.rint(prediction * 1000)
        transformed_pred = self.vectorized_transform(rounded_pred)
        
        if not apply_smoothing:
            return transformed_pred
            
        # Apply smoothing
        pred = savgol_filter(transformed_pred, 11, 5)
        pred[pred > 1] = 1.0  # Cap the maximum value at 1.0
        return pred

    def predict_disorder(self, sequence: str) -> np.ndarray:
        # 1. Get the raw neural network output
        raw_pred = self._run_inference(sequence, self.disorder_decoder)
        
        # 2. Apply the logistic calibration
        calibrated_pred = self._apply_disorder_cutoff(raw_pred)
        
        return calibrated_pred

    def predict_binding(self, sequence: str, apply_smoothing: bool = True) -> np.ndarray:
        # 1. Get raw predictions
        raw_pred = self._run_inference(sequence, self.binding_decoder)
        
        # 2. Apply the dictionary transform and smoothing
        return self._apply_binding_transform(raw_pred, apply_smoothing)

    def predict_linker(self, sequence: str, apply_smoothing: bool = True, disorder_pred: np.ndarray = None, binding_pred: np.ndarray = None) -> np.ndarray:
        """
        Predicts flexible linker propensities by combining disorder and binding scores.
        You can pass pre-calculated disorder/binding arrays to save computation time.
        """
        threshold = 0.5
        penalty_factor = 0.1
        # Calculate disorder if not provided
        if disorder_pred is None:
            disorder_pred = self.predict_disorder(sequence)
            
        # Calculate binding if not provided
        if binding_pred is None:
            binding_pred = self.predict_binding(sequence, apply_smoothing)
            
        # Apply the linker equation
        linker_pred = (disorder_pred ** 0.215) * ((1.0 - binding_pred) ** 0.967)
        
        seq_len = len(disorder_pred)
        
        # 1. Find the end of the N-terminal disordered tail
        n_idx = 0
        while n_idx < seq_len and disorder_pred[n_idx] >= threshold:
            n_idx += 1
            
        # 2. Find the start of the C-terminal disordered tail
        c_idx = seq_len - 1
        while c_idx >= 0 and disorder_pred[c_idx] >= threshold:
            c_idx -= 1
            
        # 3. Apply the penalty to the identified terminal regions
        # Only apply if the protein isn't 100% disordered (handled by the c_idx >= n_idx check)
        if n_idx > 0:
            linker_score[:n_idx] *= penalty_factor
            
        if c_idx < seq_len - 1 and c_idx >= n_idx:
            linker_score[c_idx + 1:] *= penalty_factor
            
        return linker_score

    def _tokenize(self, sequence: str) -> torch.Tensor:
        # Replaces your standalone tokenize function
        return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence], device=self.device)

    @torch.no_grad()
    def _forward_pass(self, sequence: str, decoder_model: torch.nn.Module) -> np.ndarray:
        """The core tensor operations for a single chunk. STRICTLY no smoothing here."""
        _tokens = self._tokenize(sequence)
        _padded_token = pad(_tokens, (WINDOW // 2, WINDOW // 2), 'constant', 0)
        _unfolded_tokens = _padded_token.unfold(0, WINDOW + 1, 1)
        
        _token_embedding = self.embedding_model(_unfolded_tokens, embed_only=True)
        
        _padded_embed = pad(_token_embedding, (0, 0, 0, 0, WINDOW // 2, WINDOW // 2), 'constant', 0)
        _unfolded_embedding = _padded_embed.unfold(0, WINDOW + 1, 1)
        _decoder_input = _unfolded_embedding.permute(0, 2, 1, 3)
        
        return decoder_model(_decoder_input).cpu().numpy()

    def _run_inference(self, sequence: str, decoder_model: torch.nn.Module, smoothing=None, chunk_len=1000) -> np.ndarray:
        """Handles routing to standard or low-memory execution automatically."""
        overlap = 100
        
        # Get raw prediction (chunked or not)
        if len(sequence) <= chunk_len:
            prediction = self._forward_pass(sequence, decoder_model)
        else:
            # Low-memory logic automatically kicks in for long sequences
            overlapping_predictions = []
            for chunk_start in range(0, len(sequence), chunk_len - overlap):
                chunk_seq = sequence[chunk_start:chunk_start + chunk_len]
                overlapping_predictions.append(self._forward_pass(chunk_seq, decoder_model))
                
            prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
        
        # Apply smoothing exactly once to the final array
        if smoothing:
            prediction = smoothing(prediction)
            
        return prediction

    @torch.no_grad()
    def _forward_embedding(self, sequence: str, center_only: bool) -> np.ndarray:
        """Extracts embeddings. Returns (L, 32) if center_only, else (L, 101, 32)."""
        _tokens = self._tokenize(sequence)
        _padded_token = pad(_tokens, (WINDOW // 2, WINDOW // 2), 'constant', 0)
        _unfolded_tokens = _padded_token.unfold(0, WINDOW + 1, 1)
        
        # Generates the full (L, 101, 32) tensor
        _token_embedding = self.embedding_model(_unfolded_tokens, embed_only=True)
        
        if center_only:
            # Slice out the center residue's 32-dim vector
            center_idx = WINDOW // 2
            return _token_embedding[:, center_idx, :].cpu().numpy()
        else:
            # Return the full context window for each position
            return _token_embedding.cpu().numpy()

    def get_embedding(self, sequence: str, center_only: bool = True, chunk_len: int = 1000) -> np.ndarray:
        """
        Extracts sequence embeddings. 
        If center_only=True, returns shape (L, 32).
        If center_only=False, returns shape (L, 101, 32).
        """
        overlap = 100
        
        if len(sequence) <= chunk_len:
            return self._forward_embedding(sequence, center_only)
            
        # Low-memory chunking logic
        overlapping_embeddings = []
        for chunk_start in range(0, len(sequence), chunk_len - overlap):
            chunk_seq = sequence[chunk_start:chunk_start + chunk_len]
            overlapping_embeddings.append(self._forward_embedding(chunk_seq, center_only))
            
        # Concatenate along the sequence length dimension (axis=0)
        # This works perfectly for both 2D and 3D arrays!
        embedding = np.concatenate(
            (overlapping_embeddings[0], *[x[overlap:] for x in overlapping_embeddings[1:]]), 
            axis=0
        )
        
        return embedding

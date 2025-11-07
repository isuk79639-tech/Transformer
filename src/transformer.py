import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components import (
    MultiHeadAttention,
    FeedForward, LayerNorm, PositionalEncoding, create_look_ahead_mask
)

class EncoderLayer(nn.Module):
    """Single Encoder Layer"""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # Self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """Single Decoder Layer with state caching"""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, i=0):
        super(DecoderLayer, self).__init__()
        self.i = i  # Layer index for state caching
        
        # Self-attention mechanism
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Cross-attention mechanism
        self.cross_attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, state=None):
        # State caching for self-attention
        if state is None or state[self.i] is None:
            # Training phase or first inference step: use current input
            key_values = x
        else:
            # Inference phase: concatenate with cached key-values
            key_values = torch.cat((state[self.i], x), dim=1)
        
        # Update state cache
        if state is not None:
            state[self.i] = key_values
        
        # Self-attention with cached key-values
        attn_output, _ = self.self_attention(x, key_values, key_values, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (encoder_output is fixed, no caching needed)
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, state

class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        x = self.norm(x) #确实是有的
        
        return x

class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout, i)
            for i in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
    def init_state(self, encoder_output):
        """Initialize state for incremental decoding"""
        return [None] * self.num_layers
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, state=None):
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Initialize state if not provided
        if state is None:
            state = self.init_state(encoder_output)
        
        # Pass through decoder layers with state caching
        for layer in self.layers:
            x, state = layer(x, encoder_output, src_mask, tgt_mask, state)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, state

class Transformer(nn.Module):
    """Complete Transformer Model"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Scale embeddings
        self.embedding_scale = math.sqrt(d_model)
        
        # Encoder and Decoder
        self.encoder = Encoder(
            d_model, nhead, num_encoder_layers, d_ff, dropout, max_len
        )
        
        self.decoder = Decoder(
            d_model, nhead, num_decoder_layers, d_ff, dropout, max_len
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():#递归遍历所有参数
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embeddings
        src_emb = self.src_embedding(src) * self.embedding_scale
        tgt_emb = self.tgt_embedding(tgt) * self.embedding_scale
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decoder (training mode: no state caching)
        decoder_output, _ = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask, state=None)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        src_emb = self.src_embedding(src) * self.embedding_scale
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None, state=None):
        """Decode target sequence with optional state caching"""
        tgt_emb = self.tgt_embedding(tgt) * self.embedding_scale
        decoder_output, state = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask, state)
        output = self.output_projection(decoder_output)
        return output, state
    
    def generate(self, src, src_mask, max_length=100, start_token=2, end_token=3):
        self.eval()
        
        # Ensure batch_size is 1
        if src.size(0) != 1:
            raise ValueError(f"generate() only accepts single sequences. Got batch_size={src.size(0)}")
        
        device = src.device
        
        # Encode source (only once)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize state for incremental decoding
        state = self.decoder.init_state(encoder_output)
        
        # Initialize target with start token (shape: [1, 1])
        next_token = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        # Store generated sequence tokens (list for easier termination)
        generated_tokens = [start_token]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Create target mask for current sequence length
                tgt_mask = create_look_ahead_mask(next_token.size(1)).to(device)
                
                # Decode with state caching (only current token)
                output, state = self.decode(next_token, encoder_output, src_mask, tgt_mask, state)
                
                # Get next token (single token ID)
                next_token_id = output[0, -1, :].argmax(dim=-1).item()
                generated_tokens.append(next_token_id)
                
                # Check if sequence has ended - stop immediately when end_token is generated
                if next_token_id == end_token:
                    break
                
                # Update next_token for next iteration
                next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        
        # Return as 1D tensor [seq_len] on the correct device
        return torch.tensor(generated_tokens, dtype=torch.long, device=device)
    
    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


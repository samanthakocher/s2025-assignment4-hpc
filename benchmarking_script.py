#!/usr/bin/env python3
"""
Transformer Model Benchmarking Script

This script initializes a custom Transformer model with given hyperparameters,
generates random batch data, and benchmarks forward and backward passes.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import timeit
import argparse
import time
import sys
import statistics
from typing import Dict, List, Tuple, Optional
import math
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Custom CUDA synchronize function (mock implementation)
def cuda_synchronize():
    """Simulates torch.cuda.synchronize() by adding a small delay"""
    time.sleep(0.001)  # Simulate GPU synchronization delay

@dataclass
class TransformerConfig:
    """Configuration class for Transformer model hyperparameters"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 10000

class LayerNorm:
    """Layer Normalization"""
    def __init__(self, features: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        
        # For backward pass
        self.x = None
        self.mean = None
        self.var = None
        self.normalized = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for layer normalization"""
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.normalized + self.beta
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for layer normalization"""
        # Simplified backward pass
        batch_size, seq_len, d_model = self.x.shape
        
        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * self.normalized, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))
        
        # Gradient w.r.t. input x (simplified)
        grad_input = self.gamma * grad_output / np.sqrt(self.var + self.eps)
        return grad_input

class MultiHeadAttention:
    """Multi-Head Attention layer"""
    def __init__(self, config: TransformerConfig):
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Query, Key, Value projections and output projection
        self.W_q = np.random.randn(config.d_model, config.d_model) * 0.01
        self.W_k = np.random.randn(config.d_model, config.d_model) * 0.01
        self.W_v = np.random.randn(config.d_model, config.d_model) * 0.01
        self.W_o = np.random.randn(config.d_model, config.d_model) * 0.01
        
        # For backward pass
        self.q = None
        self.k = None
        self.v = None
        self.attention_weights = None
        self.x = None
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None
        
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (n_heads, d_k)"""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_k)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine the heads back"""
        batch_size, n_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch_size, seq_len, n_heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass for multi-head attention"""
        self.x = (q, k, v)
        batch_size = q.shape[0]
        
        # Linear projections
        q_proj = np.matmul(q, self.W_q)  # (batch_size, seq_len, d_model)
        k_proj = np.matmul(k, self.W_k)  # (batch_size, seq_len, d_model)
        v_proj = np.matmul(v, self.W_v)  # (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        self.q = self.split_heads(q_proj)  # (batch_size, n_heads, seq_len, d_k)
        self.k = self.split_heads(k_proj)  # (batch_size, n_heads, seq_len, d_k)
        self.v = self.split_heads(v_proj)  # (batch_size, n_heads, seq_len, d_k)
        
        # Compute attention scores
        scores = np.matmul(self.q, self.k.transpose(0, 1, 3, 2))  # (batch_size, n_heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Apply softmax
        self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights = self.attention_weights / (np.sum(self.attention_weights, axis=-1, keepdims=True) + 1e-9)
        
        # Apply attention to values
        context = np.matmul(self.attention_weights, self.v)  # (batch_size, n_heads, seq_len, d_k)
        
        # Combine heads
        context = self.combine_heads(context)  # (batch_size, seq_len, d_model)
        
        # Final projection
        output = np.matmul(context, self.W_o)  # (batch_size, seq_len, d_model)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for multi-head attention"""
        q, k, v = self.x
        batch_size, seq_len, _ = grad_output.shape
        
        # Gradient w.r.t. output projection
        grad_context = np.matmul(grad_output, self.W_o.T)  # (batch_size, seq_len, d_model)
        self.grad_W_o = np.matmul(self.combine_heads(np.matmul(self.attention_weights, self.v)).transpose(0, 2, 1), grad_output)
        
        # Split grad_context into heads
        grad_context_heads = self.split_heads(grad_context)  # (batch_size, n_heads, seq_len, d_k)
        
        # Gradient w.r.t. values
        grad_v = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), grad_context_heads)  # (batch_size, n_heads, seq_len, d_k)
        grad_v = self.combine_heads(grad_v)  # (batch_size, seq_len, d_model)
        
        # Gradient w.r.t. attention weights (simplified)
        grad_weights = np.matmul(grad_context_heads, self.v.transpose(0, 1, 3, 2))  # (batch_size, n_heads, seq_len, seq_len)
        
        # Gradient w.r.t. scores (simplified softmax gradient)
        grad_scores = grad_weights * self.attention_weights  # (batch_size, n_heads, seq_len, seq_len)
        
        # Gradient w.r.t. query and key
        grad_q = np.matmul(grad_scores, self.k)  # (batch_size, n_heads, seq_len, d_k)
        grad_k = np.matmul(grad_scores.transpose(0, 1, 3, 2), self.q)  # (batch_size, n_heads, seq_len, d_k)
        
        # Combine heads for gradients
        grad_q = self.combine_heads(grad_q)  # (batch_size, seq_len, d_model)
        grad_k = self.combine_heads(grad_k)  # (batch_size, seq_len, d_model)
        
        # Gradient w.r.t. input projections
        self.grad_W_q = np.matmul(q.transpose(0, 2, 1), grad_q)
        self.grad_W_k = np.matmul(k.transpose(0, 2, 1), grad_k)
        self.grad_W_v = np.matmul(v.transpose(0, 2, 1), grad_v)
        
        # Gradient w.r.t. inputs
        grad_q_input = np.matmul(grad_q, self.W_q.T)
        grad_k_input = np.matmul(grad_k, self.W_k.T)
        grad_v_input = np.matmul(grad_v, self.W_v.T)
        
        return grad_q_input, grad_k_input, grad_v_input

class PositionWiseFeedForward:
    """Position-wise Feed Forward Network"""
    def __init__(self, config: TransformerConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        # Parameters
        self.W1 = np.random.randn(config.d_model, config.d_ff) * 0.01
        self.b1 = np.zeros(config.d_ff)
        self.W2 = np.random.randn(config.d_ff, config.d_model) * 0.01
        self.b2 = np.zeros(config.d_model)
        
        # For backward pass
        self.x = None
        self.h = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for position-wise feed-forward network"""
        self.x = x
        self.h = np.maximum(0, np.matmul(x, self.W1) + self.b1)  # ReLU activation
        return np.matmul(self.h, self.W2) + self.b2
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for position-wise feed-forward network"""
        # Gradient w.r.t. W2 and b2
        self.grad_W2 = np.matmul(self.h.transpose(0, 2, 1), grad_output)
        self.grad_b2 = np.sum(grad_output, axis=(0, 1))
        
        # Gradient w.r.t. hidden layer
        grad_h = np.matmul(grad_output, self.W2.T)
        
        # Gradient through ReLU
        grad_h[self.h <= 0] = 0
        
        # Gradient w.r.t. W1 and b1
        self.grad_W1 = np.matmul(self.x.transpose(0, 2, 1), grad_h)
        self.grad_b1 = np.sum(grad_h, axis=(0, 1))
        
        # Gradient w.r.t. input
        grad_input = np.matmul(grad_h, self.W1.T)
        
        return grad_input

class EncoderLayer:
    """Transformer Encoder Layer"""
    def __init__(self, config: TransformerConfig):
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.d_model)
        self.ffn = PositionWiseFeedForward(config)
        self.norm2 = LayerNorm(config.d_model)
        self.dropout = config.dropout
        
        # For backward pass
        self.x = None
        self.attention_output = None
        self.norm1_output = None
        self.ffn_output = None
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Forward pass for encoder layer"""
        self.x = x
        
        # Self-attention
        self.attention_output = self.attention.forward(x, x, x, mask)
        
        # Dropout (simulated by scaling)
        if self.dropout > 0:
            self.attention_output *= (1.0 - self.dropout)
        
        # Add & Norm
        self.norm1_output = self.norm1.forward(x + self.attention_output)
        
        # Feed-forward
        self.ffn_output = self.ffn.forward(self.norm1_output)
        
        # Dropout (simulated by scaling)
        if self.dropout > 0:
            self.ffn_output *= (1.0 - self.dropout)
        
        # Add & Norm
        output = self.norm2.forward(self.norm1_output + self.ffn_output)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for encoder layer"""
        # Through norm2
        grad_norm2 = self.norm2.backward(grad_output)
        
        # Split gradient for residual connection
        grad_ffn = grad_norm2
        grad_norm1_residual = grad_norm2
        
        # Through feed-forward
        grad_norm1 = self.ffn.backward(grad_ffn)
        
        # Add gradient from residual connection
        grad_norm1 = grad_norm1 + grad_norm1_residual
        
        # Through norm1
        grad_norm1 = self.norm1.backward(grad_norm1)
        
        # Split gradient for residual connection
        grad_attention = grad_norm1
        grad_x_residual = grad_norm1
        
        # Through attention
        grad_x, _, _ = self.attention.backward(grad_attention)
        
        # Add gradient from residual connection
        grad_x = grad_x + grad_x_residual
        
        return grad_x

class Transformer:
    """Simple Transformer Model"""
    def __init__(self, config: TransformerConfig):
        self.config = config
        
        # Embeddings
        self.token_embedding = np.random.randn(config.vocab_size, config.d_model) * 0.01
        self.position_embedding = self.create_position_embedding(config.max_seq_len, config.d_model)
        
        # Encoder layers
        self.encoder_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        
        # Final layer norm
        self.norm = LayerNorm(config.d_model)
        
        # For backward pass
        self.input_ids = None
        self.embeddings = None
        self.encoder_outputs = []
        
    def create_position_embedding(self, max_seq_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal position embeddings"""
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_embedding = np.zeros((max_seq_len, d_model))
        pos_embedding[:, 0::2] = np.sin(position * div_term)
        pos_embedding[:, 1::2] = np.cos(position * div_term)
        
        return pos_embedding
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass for transformer"""
        self.input_ids = input_ids
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_emb = self.token_embedding[input_ids]  # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding[:seq_len]  # (seq_len, d_model)
        
        # Combine embeddings
        self.embeddings = token_emb + pos_emb  # (batch_size, seq_len, d_model)
        
        # Create attention mask (allow all connections)
        mask = np.zeros((batch_size, 1, 1, seq_len))
        
        # Pass through encoder layers
        x = self.embeddings
        self.encoder_outputs = []
        
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)
            self.encoder_outputs.append(x)
        
        # Final layer norm
        output = self.norm.forward(x)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass for transformer"""
        # Through final layer norm
        grad_encoder = self.norm.backward(grad_output)
        
        # Through encoder layers in reverse
        for i in range(len(self.encoder_layers) - 1, -1, -1):
            grad_encoder = self.encoder_layers[i].backward(grad_encoder)
        
        # We'd compute gradients for embeddings here, but we'll skip for simplicity
        # as they're not needed for timing the backward pass

def generate_random_data(batch_size: int, seq_len: int, vocab_size: int) -> np.ndarray:
    """Generate random input data"""
    return np.random.randint(0, vocab_size, size=(batch_size, seq_len))

def run_benchmark(
    batch_size: int = 8, 
    seq_len: int = 128, 
    d_model: int = 512, 
    n_heads: int = 8, 
    n_layers: int = 6, 
    d_ff: int = 2048, 
    warmup_steps: int = 1, 
    measure_steps: int = 5, 
    backward: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Run benchmarking for transformer model
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        d_ff: Feed-forward dimension
        warmup_steps: Number of warmup steps
        measure_steps: Number of measurement steps
        backward: Whether to run backward pass
        
    Returns:
        Tuple of forward and backward timings
    """
    logger.info(f"Initializing model with {n_layers} layers, {n_heads} heads, d_model={d_model}, d_ff={d_ff}")
    
    # Initialize model
    config = TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=10000,
        max_seq_len=seq_len
    )
    model = Transformer(config)
    
    # Generate random data
    logger.info(f"Generating random batch of data: batch_size={batch_size}, seq_len={seq_len}")
    input_ids = generate_random_data(batch_size, seq_len, config.vocab_size)
    
    # Warmup steps
    logger.info(f"Running {warmup_steps} warmup steps")
    for _ in range(warmup_steps):
        output = model.forward(input_ids)
        if backward:
            # Create random gradients for the output
            grad_output = np.random.randn(*output.shape)
            model.backward(grad_output)
    
    # Measurement steps
    logger.info(f"Running {measure_steps} measurement steps")
    forward_timings = []
    backward_timings = []
    
    for i in range(measure_steps):
        # Forward pass timing
        start_time = timeit.default_timer()
        output = model.forward(input_ids)
        cuda_synchronize()  # Simulate CUDA synchronization
        forward_time = timeit.default_timer() - start_time
        forward_timings.append(forward_time)
        
        # Backward pass timing (if requested)
        if backward:
            # Create random gradients for the output
            grad_output = np.random.randn(*output.shape)
            
            start_time = timeit.default_timer()
            model.backward(grad_output)
            cuda_synchronize()  # Simulate CUDA synchronization
            backward_time = timeit.default_timer() - start_time
            backward_timings.append(backward_time)
        
        logger.info(f"Step {i+1}: Forward={forward_time:.4f}s, Backward={backward_time:.4f}s" if backward else f"Step {i+1}: Forward={forward_time:.4f}s, Backward=N/A")
    
    return forward_timings, backward_timings

def run_model_size_benchmarks(with_warmup: bool = True):
    """Run benchmarks for specified model sizes"""
    # Model sizes based on section 2.1.2 (assumed parameters)
    model_sizes = [
        {
            "name": "Small",
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048
        },
        {
            "name": "Medium",
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "d_ff": 3072
        },
        {
            "name": "Large",
            "d_model": 1024, 
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 4096
        }
    ]

    warmup_text = "WITH" if with_warmup else "WITHOUT"
    print(f"\n======= BENCHMARKING {warmup_text} WARMUP =======")
    
    all_results = {}
    
    for model in model_sizes:
        print(f"\nBenchmarking {model['name']} model...")
        print("-" * 50)
        
        # Run benchmark
        warmup_steps = 1 if with_warmup else 0
        forward_timings, backward_timings = run_benchmark(
            batch_size=8,
            seq_len=128,
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            n_layers=model["n_layers"],
            d_ff=model["d_ff"],
            warmup_steps=warmup_steps,
            measure_steps=5,
            backward=True
        )
        
        # Calculate statistics
        forward_avg = sum(forward_timings) / len(forward_timings)
        forward_std = statistics.stdev(forward_timings) if len(forward_timings) > 1 else 0
        backward_avg = sum(backward_timings) / len(backward_timings)
        backward_std = statistics.stdev(backward_timings) if len(backward_timings) > 1 else 0
        
        print("\n===== Benchmark Results =====")
        print(f"Model: {model['n_layers']} layers, {model['n_heads']} heads, d_model={model['d_model']}, d_ff={model['d_ff']}")
        print(f"Steps: {warmup_steps} warmup, 5 measured")
        print(f"Forward pass: {forward_avg:.6f}s ± {forward_std:.6f}s")
        print(f"Backward pass: {backward_avg:.6f}s ± {backward_std:.6f}s")
        print(f"Total (fwd+bwd): {forward_avg + backward_avg:.6f}s")
        
        all_results[model["name"]] = {
            "forward_avg": forward_avg,
            "forward_std": forward_std,
            "backward_avg": backward_avg,
            "backward_std": backward_std
        }
    
    return all_results

def run_step(model, inputs, grad_output, enable_backward=True):
    with record_function('forward_pass'):
        output = model.forward(inputs)

    if enable_backward:
        with record_function('backward_pass'):
            model.backward(grad_output)

        with record_function('optimizer'):
            pass

    return output

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transformer Model Benchmarking")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--measure_steps", type=int, default=5, help="Number of measurement steps")
    parser.add_argument("--forward_only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--no_warmup", action="store_true", help="Skip warmup steps")
    parser.add_argument("--all_model_sizes", action="store_true", help="Benchmark all model sizes from section 2.1.2")
    parser.add_argument("--compare_warmup", action="store_true", help="Compare with and without warmup")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for XL model")
    
    args = parser.parse_args()

    if args.profile:
        # Only run profiling if --profile is passed
        print("Profiling XL model...")

        # Use XL config
        config = TransformerConfig(
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            vocab_size=10000,
            max_seq_len=128
        )
        model = Transformer(config)
        input_ids = generate_random_data(8, 128, 10000)
        grad_output = np.random.randn(8, 128, 1024)

        # Warmup run
        print("Running 1 warmup step...")
        model.forward(input_ids)
        model.backward(grad_output)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                run_step(model, input_ids, grad_output, enable_backward=True)
                prof.step()

        prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
        print("\n===== Profiling Summary (sorted by CUDA timme) =====")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        return 
    
    # Run all model sizes benchmark if requested
    if args.all_model_sizes:
        with_warmup_results = run_model_size_benchmarks(with_warmup=True)
        
        # If comparing warmup vs no warmup, run again without warmup
        if args.compare_warmup:
            without_warmup_results = run_model_size_benchmarks(with_warmup=False)
            
            # Print comparison
            print("\n===== WARMUP vs NO WARMUP COMPARISON =====")
            for model_name in with_warmup_results:
                with_warmup = with_warmup_results[model_name]
                without_warmup = without_warmup_results[model_name]
                
                fwd_diff = (without_warmup["forward_avg"] - with_warmup["forward_avg"]) / with_warmup["forward_avg"] * 100
                bwd_diff = (without_warmup["backward_avg"] - with_warmup["backward_avg"]) / with_warmup["backward_avg"] * 100
                
                print(f"\n{model_name} Model:")
                print(f"Forward pass: {fwd_diff:.2f}% {'slower' if fwd_diff > 0 else 'faster'} without warmup")
                print(f"Backward pass: {bwd_diff:.2f}% {'slower' if bwd_diff > 0 else 'faster'} without warmup")
        
        return
    
    # Run single benchmark with specified parameters
    warmup_steps = 0 if args.no_warmup else args.warmup_steps
    
    # Run benchmark
    forward_timings, backward_timings = run_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        warmup_steps=warmup_steps,
        measure_steps=args.measure_steps,
        backward=not args.forward_only
    )
    
    # Calculate statistics
    forward_avg = sum(forward_timings) / len(forward_timings)
    forward_std = statistics.stdev(forward_timings) if len(forward_timings) > 1 else 0
    
    print("\n===== Benchmark Results =====")
    print(f"Model: {args.n_layers} layers, {args.n_heads} heads, d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"Data: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"Steps: {warmup_steps} warmup, {args.measure_steps} measured")
    print(f"Forward pass: {forward_avg:.6f}s ± {forward_std:.6f}s")
    
    if not args.forward_only and backward_timings:
        backward_avg = sum(backward_timings) / len(backward_timings)
        backward_std = statistics.stdev(backward_timings) if len(backward_timings) > 1 else 0
        print(f"Backward pass: {backward_avg:.6f}s ± {backward_std:.6f}s")
        print(f"Total (fwd+bwd): {forward_avg + backward_avg:.6f}s")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Transformer Model Benchmarking Script

This script initializes a custom Transformer model with given hyperparameters,
generates random batch data, and benchmarks forward and backward passes.
"""

import argparse
import statistics
import timeit
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity
import os

# Triton support
try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Check for torch.compile availability
try:
    HAS_COMPILE = True
except ImportError:
    HAS_COMPILE = False
    logger.warning("torch.compile not available in this PyTorch version")

class RMSNormTorch(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x / torch.sqrt(norm + self.eps)
    
def create_compiled_rmsnorm(backend="inductor"):
    """Create a compiled version of RMSNorm"""
    if not HAS_COMPILE:
        logger.warning("torch.compile not available, returning uncompiled version")
        return RMSNormTorch
    
    class CompiledRMSNormTorch(RMSNormTorch):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__(hidden_size, eps)
            self.compiled_forward = torch.compile(super().forward, backend=backend)
            
        def forward(self, x):
            return self.compiled_forward(x)
    
    return CompiledRMSNormTorch

# Create RMSNorm implementations
CompiledRMSNormTorch = create_compiled_rmsnorm()

if has_triton:
    @triton.jit
    def rmsnorm_kernel(X_ptr, W_ptr, Y_ptr, eps, N, BLOCK_SIZE: tl.constexpr):
        row_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load data and weights
        x = tl.load(X_ptr + row_idx * N + offsets, mask=mask, other=0.0)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0)

        # Compute RMSNorm
        x_squared = x * x
        mean_squared = tl.sum(x_squared, axis=0) / N
        inv_rms = 1.0 / tl.sqrt(mean_squared + eps)

        # Normalize and scale
        y = x * inv_rms * w

        # Store result
        tl.store(Y_ptr + row_idx * N + offsets, y, mask=mask)

    # Forward implementation
    def rmsnorm_triton_forward(x, weight, eps=1e-6):
        batch_size, seq_len, hidden_size = x.shape
        # Reshape input for the kernel
        x_reshaped = x.view(-1, hidden_size)
        # Create output tensor
        output = torch.empty_like(x_reshaped)
        
        # Configure grid and launch kernel
        grid = (x_reshaped.shape[0],)
        rmsnorm_kernel[grid](
            x_reshaped, weight, output, eps, 
            hidden_size, BLOCK_SIZE=min(1024, hidden_size)
        )
        
        # Reshape output back
        return output.view(batch_size, seq_len, hidden_size)
    
    # RMSNorm as a drop-in replacement using Triton
    class RMSNormTriton(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            
        def forward(self, x):
            # Call our Triton implementation
            return rmsnorm_triton_forward(x, self.weight, self.eps)

else:
    # Placeholder class if Triton is not available
    class RMSNormTriton(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            
        def forward(self, x):
            logger.warning("Triton not available - using PyTorch RMSNorm instead")
            norm = x.pow(2).mean(dim=-1, keepdim=True)
            return self.weight * x / torch.sqrt(norm + self.eps)

def benchmark_rmsnorm(backward=True, iterations=100, device="cuda"):
        logger.info("Benchmarking LayerNorm vs RMSNorm (PyTorch) vs RMSNorm (Triton)")
        sizes = [1024, 2048, 4096, 8192]
        num_rows = 128  # Batch size * sequence length
        hidden_dim = 1024  # Dimension of embeddings
        
        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        logger.info(f"Running on {device}")

        if not has_triton and device == "cuda":
            logger.warning("Triton not available. Triton RMSNorm will be skipped.")

        results = {}

        for size in sizes:
            print(f"\nBenchmarking for dimension size: {size}")
            
            # Initialize models
            layernorm = nn.LayerNorm(size).to(device)
            rmsnorm_torch = RMSNormTorch(size).to(device)
            rmsnorm_compiled = CompiledRMSNormTorch(size).to(device)
            
            if has_triton and device == "cuda":
                rmsnorm_triton = RMSNormTriton(size).to(device)
            
            # Warm-up
            for _ in range(10):
                x = torch.randn(num_rows, size, device=device, requires_grad=backward)
                x_torch = x.clone().detach().requires_grad_(backward)
                x_compiled = x.clone().detach().requires_grad_(backward)
                
                # Forward
                ln_out = layernorm(x)
                rms_torch_out = rmsnorm_torch(x_torch)
                rms_compiled_out = rmsnorm_compiled(x_compiled)
                
                if has_triton and device == "cuda":
                    x_triton = x.clone().detach().requires_grad_(backward)
                    rms_triton_out = rmsnorm_triton(x_triton)
                
                # Backward (if enabled)
                if backward:
                    dy = torch.randn_like(ln_out)
                    ln_out.backward(dy)
                    
                    dy_torch = dy.clone()
                    rms_torch_out.backward(dy_torch)

                    dy_compiled = dy.clone()
                    rms_compiled_out.backward(dy_compiled)
                    
                    if has_triton and device == "cuda":
                        dy_triton = dy.clone()
                        rms_triton_out.backward(dy_triton)
            
            # Benchmark LayerNorm
            ln_times = []
            for _ in range(iterations):
                x = torch.randn(num_rows, size, device=device, requires_grad=backward)
                
                # Forward
                torch.cuda.synchronize() if device == "cuda" else None
                start = timeit.default_timer()
                ln_out = layernorm(x)
                
                # Backward (if enabled)
                if backward:
                    dy = torch.randn_like(ln_out)
                    ln_out.backward(dy)
                    
                torch.cuda.synchronize() if device == "cuda" else None
                ln_times.append(timeit.default_timer() - start)
            
            # Benchmark RMSNorm (PyTorch)
            rms_torch_times = []
            for _ in range(iterations):
                x_torch = torch.randn(num_rows, size, device=device, requires_grad=backward)
                
                # Forward
                torch.cuda.synchronize() if device == "cuda" else None
                start = timeit.default_timer()
                rms_torch_out = rmsnorm_torch(x_torch)
                
                # Backward (if enabled)
                if backward:
                    dy_torch = torch.randn_like(rms_torch_out)
                    rms_torch_out.backward(dy_torch)
                    
                torch.cuda.synchronize() if device == "cuda" else None
                rms_torch_times.append(timeit.default_timer() - start)

            # Benchmark RMSNorm (Compiled PyTorch)
            rms_compiled_times = []
            for _ in range(iterations):
                x_compiled = torch.randn(num_rows, size, device=device, requires_grad=backward)
                
                # Forward
                torch.cuda.synchronize() if device == "cuda" else None
                start = timeit.default_timer()
                rms_compiled_out = rmsnorm_compiled(x_compiled)
                
                # Backward (if enabled)
                if backward:
                    dy_compiled = torch.randn_like(rms_compiled_out)
                    rms_compiled_out.backward(dy_compiled)
                    
                torch.cuda.synchronize() if device == "cuda" else None
                rms_compiled_times.append(timeit.default_timer() - start)
            
            # Benchmark RMSNorm (Triton)
            if has_triton and device == "cuda":
                rms_triton_times = []
                for _ in range(iterations):
                    x_triton = torch.randn(num_rows, size, device=device, requires_grad=backward)
                    
                    # Forward
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    rms_triton_out = rmsnorm_triton(x_triton)
                    
                    # Backward (if enabled)
                    if backward:
                        dy_triton = torch.randn_like(rms_triton_out)
                        rms_triton_out.backward(dy_triton)
                        
                    torch.cuda.synchronize()
                    rms_triton_times.append(timeit.default_timer() - start)
            
            # Calculate statistics
            ln_avg = sum(ln_times) / len(ln_times)
            ln_std = statistics.stdev(ln_times) if len(ln_times) > 1 else 0
            
            rms_torch_avg = sum(rms_torch_times) / len(rms_torch_times)
            rms_torch_std = statistics.stdev(rms_torch_times) if len(rms_torch_times) > 1 else 0

            rms_compiled_avg = sum(rms_compiled_times) / len(rms_compiled_times)
            rms_compiled_std = statistics.stdev(rms_compiled_times) if len(rms_compiled_times) > 1 else 0
            
            # Print results
            pass_type = "Forward+Backward" if backward else "Forward only"
            print(f"Size {size}, {pass_type}:")
            print(f"  LayerNorm (PyTorch):         {ln_avg*1000:.3f}ms ± {ln_std*1000:.3f}ms")
            print(f"  RMSNorm (PyTorch):           {rms_torch_avg*1000:.3f}ms ± {rms_torch_std*1000:.3f}ms")
            print(f"  RMSNorm (Compiled PyTorch):  {rms_compiled_avg*1000:.3f}ms ± {rms_compiled_std*1000:.3f}ms")
            print(f"  Speedup (RMS vs LN):         {ln_avg/rms_torch_avg:.2f}x")
            print(f"  Speedup (Compiled vs PyTorch RMS): {rms_torch_avg/rms_compiled_avg:.2f}x")
            print(f"  Speedup (Compiled vs LN):    {ln_avg/rms_compiled_avg:.2f}x")
            
            if has_triton and device == "cuda":
                rms_triton_avg = sum(rms_triton_times) / len(rms_triton_times)
                rms_triton_std = statistics.stdev(rms_triton_times) if len(rms_triton_times) > 1 else 0
                print(f"  RMSNorm (Triton):            {rms_triton_avg*1000:.3f}ms ± {rms_triton_std*1000:.3f}ms")
                print(f"  Speedup (Triton vs PyTorch RMS): {rms_torch_avg/rms_triton_avg:.2f}x")
                print(f"  Speedup (Triton vs Compiled): {rms_compiled_avg/rms_triton_avg:.2f}x")
                print(f"  Total speedup (Triton vs LN): {ln_avg/rms_triton_avg:.2f}x")
            
            # Store results
            results[size] = {
                'layernorm': (ln_avg, ln_std),
                'rmsnorm_torch': (rms_torch_avg, rms_torch_std),
                'rmsnorm_compiled': (rms_compiled_avg, rms_compiled_std)
            }
            if has_triton and device == "cuda":
                results[size]['rmsnorm_triton'] = (rms_triton_avg, rms_triton_std)
        
        # Return results in DataFrame format for easy analysis
        try:
            import pandas as pd
            rows = []
            for size in results:
                row = {'Size': size}
                for impl, (avg, std) in results[size].items():
                    row[f"{impl}_ms"] = avg * 1000
                    row[f"{impl}_std"] = std * 1000
                rows.append(row)
            
            results_df = pd.DataFrame(rows)
            print("\nSummary Table (time in ms):")
            print(results_df.to_string())
            return results_df
        except ImportError:
            return results

def benchmark_transformer_with_rmsnorm(use_triton=False):
    """Benchmark the transformer with different RMSNorm implementations"""
    # Create a modified transformer model that uses RMSNorm instead of LayerNorm
    
    class EncoderLayerWithRMSNorm(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm', use_triton=False, use_compiled=False):
            super().__init__()
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
            
            # Choose normalization layer based on type
            if norm_type == 'layernorm':
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
            elif norm_type == 'rmsnorm':
                if use_triton and has_triton and torch.cuda.is_available():
                    self.norm1 = RMSNormTriton(d_model)
                    self.norm2 = RMSNormTriton(d_model)
                elif use_compiled and HAS_COMPILE:
                    self.norm1 = CompiledRMSNormTorch(d_model)
                    self.norm2 = CompiledRMSNormTorch(d_model)
                else:
                    self.norm1 = RMSNormTorch(d_model)
                    self.norm2 = RMSNormTorch(d_model)
                    
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            attn_output = self.dropout1(self.attn(x, x, x, mask))
            x = self.norm1(x + attn_output)
            ffn_output = self.dropout2(self.ffn(x))
            return self.norm2(x + ffn_output)
    
    class TransformerWithRMSNorm(nn.Module):
        def __init__(self, d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len, 
                 dropout=0.1, norm_type='layernorm', use_triton=False, use_compiled=False):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            self.encoder = nn.ModuleList([
                EncoderLayerWithRMSNorm(d_model, n_heads, d_ff, dropout, norm_type, use_triton, use_compiled)
                for _ in range(n_layers)
            ])
            
            # Choose final normalization layer based on type
            if norm_type == 'layernorm':
                self.norm = nn.LayerNorm(d_model)
            elif norm_type == 'rmsnorm':
                if use_triton and has_triton and torch.cuda.is_available():
                    self.norm = RMSNormTriton(d_model)
                elif use_compiled and HAS_COMPILE:
                    self.norm = CompiledRMSNormTorch(d_model)
                else:
                    self.norm = RMSNormTorch(d_model)

        def forward(self, input_ids):
            x = self.token_embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1)]
            mask = torch.ones(input_ids.size(0), 1, 1, input_ids.size(1), device=input_ids.device)
            for layer in self.encoder:
                x = layer(x, mask)
            return self.norm(x)

    # Define the MultiHeadAttention and PositionWiseFeedForward classes
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def split_heads(self, x):
            bsz, seq_len, _ = x.shape
            return x.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        def combine_heads(self, x):
            bsz, n_heads, seq_len, d_k = x.shape
            return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        def forward(self, q, k, v, mask=None):
            q = self.split_heads(self.W_q(q))
            k = self.split_heads(self.W_k(k))
            v = self.split_heads(self.W_v(v))
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = self.dropout(F.softmax(scores, dim=-1))
            context = torch.matmul(weights, v)
            return self.W_o(self.combine_heads(context))
    
    class PositionWiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.fc2(self.dropout(F.relu(self.fc1(x))))
    
    # Benchmark settings
    batch_size = 16
    seq_len = 128
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    vocab_size = 10000
    warmup = 3
    steps = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for CUDA availability
    if device != "cuda" and use_triton:
        logger.warning("Triton requires CUDA. Cannot benchmark Triton on CPU.")
        return None
    
    logger.info(f"Benchmarking Transformer with {'Triton' if use_triton else 'PyTorch'} RMSNorm")
    
    # Initialize models
    impl_name = "Triton" if use_triton else "PyTorch"
    model = TransformerWithRMSNorm(
        d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, 
        dropout=0.1, use_triton=use_triton
    ).to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(warmup):
        output = model(input_ids)
        loss = output.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    # Benchmark forward pass
    torch.cuda.synchronize() if device == "cuda" else None
    fwd_times = []
    for _ in range(steps):
        torch.cuda.synchronize() if device == "cuda" else None
        start = timeit.default_timer()
        output = model(input_ids)
        torch.cuda.synchronize() if device == "cuda" else None
        fwd_times.append(timeit.default_timer() - start)
    
    # Benchmark backward pass
    bwd_times = []
    for _ in range(steps):
        output = model(input_ids)
        loss = output.mean()
        optim.zero_grad()
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize() if device == "cuda" else None
        bwd_times.append(timeit.default_timer() - start)
        
        optim.step()
    
    # Calculate statistics
    fwd_avg = sum(fwd_times) / len(fwd_times)
    fwd_std = statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0
    
    bwd_avg = sum(bwd_times) / len(bwd_times)
    bwd_std = statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0
    
    # Print results
    print(f"\nTransformer with {impl_name} RMSNorm:")
    print(f"  Forward pass:  {fwd_avg*1000:.3f}ms ± {fwd_std*1000:.3f}ms")
    print(f"  Backward pass: {bwd_avg*1000:.3f}ms ± {bwd_std*1000:.3f}ms")
    print(f"  Total time:    {(fwd_avg + bwd_avg)*1000:.3f}ms")
    
    return {
        'forward': (fwd_avg, fwd_std),
        'backward': (bwd_avg, bwd_std)
    }

def benchmark_transformer_model(model_size='small', norm_type='layernorm', use_compile=False, 
                                device="cuda", warmup=5, steps=10, amp=False):
    """
    Benchmark different sized transformer models with different normalization layers
    
    model_size: 'small', 'medium', or 'large'
    norm_type: 'layernorm' or 'rmsnorm'
    use_compile: whether to use torch.compile for the entire model
    """
    # Define model configurations
    model_configs = {
        'small': {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048
        },
        'medium': {
            'd_model': 768,
            'n_heads': 12,
            'n_layers': 12,
            'd_ff': 3072
        },
        'large': {
            'd_model': 1024,
            'n_heads': 16,
            'n_layers': 24,
            'd_ff': 4096
        }
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_configs.keys())}")
    
    config = model_configs[model_size]
    
    # Common parameters
    batch_size = 16
    seq_len = 128
    vocab_size = 10000
    
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Benchmarking {model_size.capitalize()} Transformer with {norm_type}")
    logger.info(f"Config: {config}")
    logger.info(f"Using compilation: {use_compile}")
    logger.info(f"Device: {device}")
    
    # Create the model
    model = TransformerWithRMSNorm(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout=0.1,
        norm_type=norm_type,
        use_triton=False,
        use_compiled=False  # We'll use torch.compile for the whole model instead
    ).to(device)
    
    # Apply torch.compile if requested
    if use_compile and HAS_COMPILE:
        model = torch.compile(model)
    
    # Optimizer setup
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Generate random data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Context for AMP
    context = autocast(device_type=device) if amp else nullcontext()
    
    # Warmup
    for _ in range(warmup):
        with context:
            output = model(input_ids)
            loss = output.mean()
        optim.zero_grad()
        with context:
            loss.backward()
        optim.step()
    
    # Benchmark forward pass
    torch.cuda.synchronize() if device == "cuda" else None
    fwd_times = []
    for _ in range(steps):
        torch.cuda.synchronize() if device == "cuda" else None
        start = timeit.default_timer()
        with context:
            output = model(input_ids)
        torch.cuda.synchronize() if device == "cuda" else None
        fwd_times.append(timeit.default_timer() - start)
    
    # Benchmark backward pass
    bwd_times = []
    for _ in range(steps):
        with context:
            output = model(input_ids)
            loss = output.mean()
        optim.zero_grad()
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = timeit.default_timer()
        with context:
            loss.backward()
        torch.cuda.synchronize() if device == "cuda" else None
        bwd_times.append(timeit.default_timer() - start)
        
        optim.step()
    
    # Benchmark full step (forward + backward + optimizer)
    full_times = []
    for _ in range(steps):
        torch.cuda.synchronize() if device == "cuda" else None
        start = timeit.default_timer()
        
        with context:
            output = model(input_ids)
            loss = output.mean()
        optim.zero_grad()
        with context:
            loss.backward()
        optim.step()
        
        torch.cuda.synchronize() if device == "cuda" else None
        full_times.append(timeit.default_timer() - start)
    
    # Calculate statistics
    fwd_avg = sum(fwd_times) / len(fwd_times)
    fwd_std = statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0
    
    bwd_avg = sum(bwd_times) / len(bwd_times)
    bwd_std = statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0
    
    full_avg = sum(full_times) / len(full_times)
    full_std = statistics.stdev(full_times) if len(full_times) > 1 else 0
    
    # Return results
    return {
        'model_size': model_size,
        'norm_type': norm_type,
        'compiled': use_compile,
        'forward': (fwd_avg, fwd_std),
        'backward': (bwd_avg, bwd_std),
        'full_step': (full_avg, full_std)
    }

def benchmark_all_transformer_models(device="cuda", amp=False):
    """Run benchmarks for all model sizes and configurations"""
    results = []
    
    model_sizes = ['small', 'medium', 'large']
    norm_types = ['layernorm', 'rmsnorm']
    compile_options = [False, True]
    
    for model_size in model_sizes:
        for norm_type in norm_types:
            for use_compile in compile_options:
                if use_compile and not HAS_COMPILE:
                    continue  # Skip if torch.compile not available
                
                # More warmup/steps for small models, fewer for large
                warmup = 10 if model_size == 'small' else (5 if model_size == 'medium' else 3)
                steps = 20 if model_size == 'small' else (10 if model_size == 'medium' else 5)
                
                try:
                    result = benchmark_transformer_model(
                        model_size=model_size,
                        norm_type=norm_type,
                        use_compile=use_compile,
                        device=device,
                        warmup=warmup,
                        steps=steps,
                        amp=amp
                    )
                    results.append(result)
                    
                    # Print current result
                    compiled_str = "Compiled" if use_compile else "Vanilla"
                    fwd_ms = result['forward'][0] * 1000
                    bwd_ms = result['backward'][0] * 1000
                    full_ms = result['full_step'][0] * 1000
                    
                    print(f"\n{model_size.capitalize()} - {norm_type} - {compiled_str}:")
                    print(f"  Forward:   {fwd_ms:.3f}ms")
                    print(f"  Backward:  {bwd_ms:.3f}ms")
                    print(f"  Full step: {full_ms:.3f}ms")
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {model_size} with {norm_type} (compiled={use_compile}): {e}")
    
    # Create comparison table
    try:
        import pandas as pd
        
        data = []
        for r in results:
            data.append({
                'Model Size': r['model_size'].capitalize(),
                'Norm Type': r['norm_type'],
                'Compiled': 'Yes' if r['compiled'] else 'No',
                'Forward (ms)': r['forward'][0] * 1000,
                'Forward Std': r['forward'][1] * 1000,
                'Backward (ms)': r['backward'][0] * 1000, 
                'Backward Std': r['backward'][1] * 1000,
                'Full Step (ms)': r['full_step'][0] * 1000,
                'Full Step Std': r['full_step'][1] * 1000
            })
        
        df = pd.DataFrame(data)
        print("\n=== Full Benchmark Results ===")
        print(df.to_string(index=False))
        
        # Compute speedups
        speedups = []
        for size in model_sizes:
            for norm in norm_types:
                # Get baseline (vanilla) performance
                baseline = df[(df['Model Size'] == size.capitalize()) & 
                              (df['Norm Type'] == norm) & 
                              (df['Compiled'] == 'No')]
                
                if baseline.empty:
                    continue
                    
                # Get compiled performance
                compiled = df[(df['Model Size'] == size.capitalize()) & 
                              (df['Norm Type'] == norm) & 
                              (df['Compiled'] == 'Yes')]
                              
                if compiled.empty:
                    continue
                
                # Calculate speedups
                forward_speedup = baseline['Forward (ms)'].values[0] / compiled['Forward (ms)'].values[0]
                backward_speedup = baseline['Backward (ms)'].values[0] / compiled['Backward (ms)'].values[0]
                full_speedup = baseline['Full Step (ms)'].values[0] / compiled['Full Step (ms)'].values[0]
                
                speedups.append({
                    'Model Size': size.capitalize(),
                    'Norm Type': norm,
                    'Forward Speedup': forward_speedup,
                    'Backward Speedup': backward_speedup,
                    'Full Step Speedup': full_speedup
                })
        
        # Print speedup comparisons
        if speedups:
            speedups_df = pd.DataFrame(speedups)
            print("\n=== Compilation Speedups (higher is better) ===")
            print(speedups_df.to_string(index=False))
        
        # Normalization type comparison (LayerNorm vs RMSNorm)
        norm_speedups = []
        for size in model_sizes:
            for compile_opt in ['Yes', 'No']:
                # Get LayerNorm performance (baseline)
                ln_perf = df[(df['Model Size'] == size.capitalize()) & 
                             (df['Norm Type'] == 'layernorm') & 
                             (df['Compiled'] == compile_opt)]
                             
                if ln_perf.empty:
                    continue
                    
                # Get RMSNorm performance
                rms_perf = df[(df['Model Size'] == size.capitalize()) & 
                              (df['Norm Type'] == 'rmsnorm') & 
                              (df['Compiled'] == compile_opt)]
                              
                if rms_perf.empty:
                    continue
                
                # Calculate speedups
                forward_speedup = ln_perf['Forward (ms)'].values[0] / rms_perf['Forward (ms)'].values[0]
                backward_speedup = ln_perf['Backward (ms)'].values[0] / rms_perf['Backward (ms)'].values[0]
                full_speedup = ln_perf['Full Step (ms)'].values[0] / rms_perf['Full Step (ms)'].values[0]
                
                norm_speedups.append({
                    'Model Size': size.capitalize(),
                    'Compiled': compile_opt,
                    'Forward Speedup': forward_speedup,
                    'Backward Speedup': backward_speedup,
                    'Full Step Speedup': full_speedup
                })
        
        # Print normalization comparison
        if norm_speedups:
            norm_speedups_df = pd.DataFrame(norm_speedups)
            print("\n=== RMSNorm vs LayerNorm Speedups (higher is better) ===")
            print(norm_speedups_df.to_string(index=False))
            
        return df
    except ImportError:
        logger.warning("pandas not available, returning raw results")
        return results

def plot_benchmark_results(results_df):
    """Create visualization plots for the benchmark results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Forward pass comparison
        plt.figure(figsize=(12, 8))
        g = sns.catplot(
            data=results_df,
            x="Model Size", y="Forward (ms)",
            hue="Norm Type", col="Compiled",
            kind="bar", height=6, aspect=0.8
        )
        g.set_axis_labels("Model Size", "Forward Pass Time (ms)")
        g.fig.suptitle("Forward Pass Performance Comparison", y=1.05)
        plt.tight_layout()
        plt.savefig("forward_pass_comparison.png")
        
        # 2. Full step comparison
        plt.figure(figsize=(12, 8))
        g = sns.catplot(
            data=results_df,
            x="Model Size", y="Full Step (ms)",
            hue="Norm Type", col="Compiled",
            kind="bar", height=6, aspect=0.8
        )
        g.set_axis_labels("Model Size", "Full Step Time (ms)")
        g.fig.suptitle("Full Step Performance Comparison", y=1.05)
        plt.tight_layout()
        plt.savefig("full_step_comparison.png")
        
        # 3. Compilation speedup for each model size and norm type
        # Reshape data for plot
        plot_data = []
        for index, row in results_df.iterrows():
            # Find the matching non-compiled version
            baseline = results_df[
                (results_df['Model Size'] == row['Model Size']) &
                (results_df['Norm Type'] == row['Norm Type']) &
                (results_df['Compiled'] == 'No')
            ]
            
            if baseline.empty or row['Compiled'] == 'No':
                continue
                
            # Calculate speedups
            fwd_speedup = baseline['Forward (ms)'].values[0] / row['Forward (ms)']
            bwd_speedup = baseline['Backward (ms)'].values[0] / row['Backward (ms)']
            full_speedup = baseline['Full Step (ms)'].values[0] / row['Full Step (ms)']
            
            plot_data.append({
                'Model Size': row['Model Size'],
                'Norm Type': row['Norm Type'],
                'Phase': 'Forward',
                'Speedup': fwd_speedup
            })
            plot_data.append({
                'Model Size': row['Model Size'],
                'Norm Type': row['Norm Type'],
                'Phase': 'Backward',
                'Speedup': bwd_speedup
            })
            plot_data.append({
                'Model Size': row['Model Size'],
                'Norm Type': row['Norm Type'],
                'Phase': 'Full Step',
                'Speedup': full_speedup
            })
            
        if plot_data:
            speedup_df = pd.DataFrame(plot_data)
            plt.figure(figsize=(12, 8))
            g = sns.catplot(
                data=speedup_df,
                x="Model Size", y="Speedup",
                hue="Phase", col="Norm Type",
                kind="bar", height=6, aspect=0.8
            )
            g.set_axis_labels("Model Size", "Speedup Factor (higher is better)")
            g.fig.suptitle("Compilation Speedup Factors", y=1.05)
            plt.tight_layout()
            plt.savefig("compilation_speedup.png")
        
        logger.info("Plots saved to current directory")
        
    except ImportError:
        logger.warning("matplotlib or seaborn not available, skipping plots")

def analyze_rmsnorm_implementations():
    """Run comprehensive analysis of different RMSNorm implementations"""
    # Run forward-only benchmarks
    print("\n=== RMSNorm Forward Pass Benchmarks ===")
    forward_results = benchmark_rmsnorm(backward=False)
    
    # Run forward+backward benchmarks
    print("\n=== RMSNorm Forward+Backward Pass Benchmarks ===")
    full_results = benchmark_rmsnorm(backward=True)
    
    return forward_results, full_results
    
def analyze_transformer_models(amp=False):
    """Run comprehensive analysis of Transformer models with different normalization layers"""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        logger.warning("CUDA not available, using CPU for benchmarks")
        device = "cpu"
    
    print("\n=== Transformer Model Benchmarks ===")
    results = benchmark_all_transformer_models(device=device, amp=amp)
    
    try:
        # Save results to CSV
        results.to_csv("transformer_benchmarks.csv", index=False)
        logger.info("Results saved to transformer_benchmarks.csv")
        
        # Create plots
        plot_benchmark_results(results)
    except:
        logger.warning("Failed to save results or create plots")
    
    return results

class TorchBenchmark:
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def split_heads(self, x):
            bsz, seq_len, _ = x.shape
            return x.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        def combine_heads(self, x):
            bsz, n_heads, seq_len, d_k = x.shape
            return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        def forward(self, q, k, v, mask=None):
            q = self.split_heads(self.W_q(q))
            k = self.split_heads(self.W_k(k))
            v = self.split_heads(self.W_v(v))
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = self.dropout(F.softmax(scores, dim=-1))
            context = torch.matmul(weights, v)
            return self.W_o(self.combine_heads(context))
        
    class PositionWiseFeedForward(nn.Module):
            def __init__(self, d_model, d_ff, dropout=0.1):
                super().__init__()
                self.fc1 = nn.Linear(d_model, d_ff)
                self.fc2 = nn.Linear(d_ff, d_model)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                return self.fc2(self.dropout(F.relu(self.fc1(x))))
            
            class MultiHeadAttention(nn.Module):
                def __init__(self, d_model, n_heads, dropout=0.1):
                    super().__init__()
                    self.d_model = d_model
                    self.n_heads = n_heads
                    self.d_k = d_model // n_heads
                    self.W_q = nn.Linear(d_model, d_model, bias=False)
                    self.W_k = nn.Linear(d_model, d_model, bias=False)
                    self.W_v = nn.Linear(d_model, d_model, bias=False)
                    self.W_o = nn.Linear(d_model, d_model, bias=False)
                    self.dropout = nn.Dropout(dropout)

                def split_heads(self, x):
                    bsz, seq_len, _ = x.shape
                    return x.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

                def combine_heads(self, x):
                    bsz, n_heads, seq_len, d_k = x.shape
                    return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

                def forward(self, q, k, v, mask=None):
                    q = self.split_heads(self.W_q(q))
                    k = self.split_heads(self.W_k(k))
                    v = self.split_heads(self.W_v(v))
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
                    if mask is not None:
                        scores = scores.masked_fill(mask == 0, -1e9)
                    weights = self.dropout(F.softmax(scores, dim=-1))
                    context = torch.matmul(weights, v)
                    return self.W_o(self.combine_heads(context))


    class EncoderLayer(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = TorchBenchmark.MultiHeadAttention(d_model, n_heads, dropout)
            self.ffn = TorchBenchmark.PositionWiseFeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = x + self.dropout1(self.attn(x, x, x, mask))
            x = self.norm1(x)
            x = x + self.dropout2(self.ffn(x))
            return self.norm2(x)

    class Transformer(nn.Module):
        def __init__(self, d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len, dropout=0.1):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            self.encoder = nn.ModuleList([
                TorchBenchmark.EncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, input_ids):
            x = self.token_embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1)]
            mask = torch.ones(input_ids.size(0), 1, 1, input_ids.size(1), device=input_ids.device)
            for layer in self.encoder:
                x = layer(x, mask)
            return self.norm(x)

    @staticmethod
    def generate_random_data(batch_size, seq_len, vocab_size, device):
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    @staticmethod
    def run_benchmark(batch_size=8, seq_len=128, d_model=512, n_heads=8, n_layers=6, d_ff=2048,
            warmup=1, steps=5, device="cuda", use_amp=False):
        logger.info("Running PyTorch benchmark")
        model = TorchBenchmark.Transformer(d_model, n_heads, n_layers, d_ff, 10000, seq_len).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        input_ids = TorchBenchmark.generate_random_data(batch_size, seq_len, 10000, device)
        context = autocast(device_type=device) if use_amp else nullcontext()
        for _ in range(warmup):
            with context:
                loss = model(input_ids).mean()
                optim.zero_grad(); loss.backward(); optim.step()
        fwd_times, bwd_times = [], []
        for _ in range(steps):
            torch.cuda.synchronize() if device == 'cuda' else None
            start = timeit.default_timer()
            with context:
                output = model(input_ids)
            torch.cuda.synchronize() if device == 'cuda' else None
            fwd_times.append(timeit.default_timer() - start)
            with context:
                loss = output.mean()
            optim.zero_grad()
            torch.cuda.synchronize() if device == 'cuda' else None
            start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize() if device == 'cuda' else None
            bwd_times.append(timeit.default_timer() - start)
            optim.step()
        return fwd_times, bwd_times

def generate_numpy_data(batch_size, seq_len, vocab_size):
    return np.random.randint(0, vocab_size, size=(batch_size, seq_len))

class LayerNorm:
    def __init__(self, features: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
            
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
    
def run_benchmark(batch_size=8, seq_len=128, d_model=512, n_heads=8, n_layers=6, d_ff=2048, warmup_steps=1, measure_steps=5, backward=True):
    logger.info("Running NumPy benchmark")
    input_ids = generate_numpy_data(batch_size, seq_len, 10000)
    dummy_output_shape = (batch_size, seq_len, d_model)

    for _ in range(warmup_steps):
        _ = np.random.randn(*dummy_output_shape)

    forward_timings, backward_timings = [], []

    for _ in range(measure_steps):
        start = timeit.default_timer()
        _ = np.random.randn(*dummy_output_shape)
        forward_timings.append(timeit.default_timer() - start)

        if backward:
            start = timeit.default_timer()
            _ = np.random.randn(*dummy_output_shape)  # mock backward
            backward_timings.append(timeit.default_timer() - start)

    return forward_timings, backward_timings

def safe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark_norm_layers(backward=True, iterations=100):
    logger.info("Benchmarking LayerNorm vs RMSNorm")
    sizes = [1024, 2048, 4096, 8192]
    num_rows = 1024

    timings = {"LayerNorm": [], "RMSNorm": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")

    results = {}

    for size in sizes:
        print(f"\nBenchmarking for dimension size: {size}")

        # Initialize models
        layernorm = nn.LayerNorm(size, elementwise_affine=True).to(device)
        rmsnorm = RMSNormTorch(size).to(device)
        
        # Warm-up
        for _ in range(10):
            # Create fresh inputs for each warm-up iteration
            x_ln = torch.randn(num_rows, size, device=device, requires_grad=backward)
            x_rms = x_ln.clone().detach().requires_grad_(backward)
            
            # Forward
            ln_out = layernorm(x_ln)
            rms_out = rmsnorm(x_rms)
            
            # Backward (if enabled)
            if backward:
                dy = torch.randn_like(ln_out)
                ln_out.backward(dy)
                rms_out.backward(dy.clone())
        
        # Benchmark LayerNorm
        ln_times = []
        for _ in range(iterations):
            # Create fresh inputs for each benchmark iteration
            x_ln = torch.randn(num_rows, size, device=device, requires_grad=backward)
            if backward and x_ln.grad is not None:
                x_ln.grad = None
                
            # Forward
            safe_synchronize()
            start = timeit.default_timer()
            ln_out = layernorm(x_ln)
            
            # Backward (if enabled)
            if backward:
                dy = torch.randn_like(ln_out)
                ln_out.backward(dy)
                
            safe_synchronize()
            ln_times.append(timeit.default_timer() - start)
        
        # Benchmark RMSNorm
        rms_times = []
        for _ in range(iterations):
            # Create fresh inputs for each benchmark iteration
            x_rms = torch.randn(num_rows, size, device=device, requires_grad=backward)
            if backward and x_rms.grad is not None:
                x_rms.grad = None
                
            # Forward
            safe_synchronize()
            start = timeit.default_timer()
            rms_out = rmsnorm(x_rms)
            
            # Backward (if enabled)
            if backward:
                dy = torch.randn_like(rms_out)
                rms_out.backward(dy)
                
            safe_synchronize()
            rms_times.append(timeit.default_timer() - start)
        
        # Calculate statistics
        ln_avg = sum(ln_times) / len(ln_times)
        ln_std = statistics.stdev(ln_times) if len(ln_times) > 1 else 0
        
        rms_avg = sum(rms_times) / len(rms_times)
        rms_std = statistics.stdev(rms_times) if len(rms_times) > 1 else 0
        
        # Print results
        pass_type = "Forward+Backward" if backward else "Forward only"
        print(f"Size {size}, {pass_type}:")
        print(f"  LayerNorm: {ln_avg*1000:.3f}ms ± {ln_std*1000:.3f}ms")
        print(f"  RMSNorm:   {rms_avg*1000:.3f}ms ± {rms_std*1000:.3f}ms")
        print(f"  Speedup:   {ln_avg/rms_avg:.2f}x")
        
        # Store results
        if size not in results:
            results[size] = {}
        results[size]['layernorm'] = (ln_avg, ln_std)
        results[size]['rmsnorm'] = (rms_avg, rms_std)
    
    return results

def benchmark_rmsnorm_torch():
    print("Benchmarking RMSNormTorch")
    sizes = [1024, 2048, 4096, 8192]
    num_rows = 8192
    iterations = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for size in sizes:
        x = torch.randn(num_rows, size, device=device)
        weight = torch.randn(size, device=device)
        model = RMSNormTorch(size).to(device)
        model.weight.data = weight.clone()

        for _ in range(10):
            model(x)  # warm-up

        start = timeit.default_timer()
        for _ in range(iterations):
            model(x)
        total_time = timeit.default_timer() - start

        print(f"Size {size}: PyTorch RMSNorm {total_time:.4f}s")

def export_flame_graph(model, input_ids, device):
    logger.info("Generating profiler trace for flame graph...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            with record_function("forward_pass"):
                output = model(input_ids)
            with record_function("backward_pass"):
                loss = output.mean()
                model.zero_grad()
                loss.backward()
            prof.step()

    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    logger.info("Flamegraph stacks and trace saved.")

def profile_memory_run(forward_only=True, use_amp=False):
    import torch.profiler
    from contextlib import nullcontext
    config = MODEL_2_7B_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TorchBenchmark.Transformer(
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config["d_ff"],
        config["vocab_size"],
        config["max_seq_len"]
    ).to(device)

    input_ids = TorchBenchmark.generate_random_data(1, 512, config["vocab_size"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    context = autocast(device_type=device) if use_amp else nullcontext()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(5):
            with context:
                outputs = model(input_ids)
                loss = outputs.mean()
            if not forward_only:
                optimizer.zero_grad()
                with context:
                    loss.backward()
                optimizer.step()
            prof.step()
    
    # Export trace for PyTorch memory visualizer
    output_dir = f"./memory_profile_forward" if forward_only else f"./memory_profile_full"
    os.makedirs(output_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))

MODEL_2_7B_CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,
    "vocab_size": 50000,
    "max_seq_len": 512
}

def run_memory_profiles_for_model_sizes():
    import shutil
    from pathlib import Path

    sizes = [
        ("Small", 512, 8, 6, 2048),
        ("Medium", 768, 12, 12, 3072),
        ("Large", 1024, 16, 24, 4096)
    ]
    base_config = {
        "vocab_size": 50000,
        "max_seq_len": 512
    }

    for name, d_model, n_heads, n_layers, d_ff in sizes:
        print(f"\n=== Profiling {name} Model ===")

        MODEL_2_7B_CONFIG.update({
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            **base_config
        })

        for mode in ["forward", "full"]:
            dir_path = Path(f"./memory_profile_{mode}")
            if dir_path.exists():
                shutil.rmtree(dir_path)

        print("Profiling forward pass...")
        profile_memory_run(forward_only=True)
        dest = f"./memory_profile_forward_{name}"
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(f"./memory_profile_forward", dest)

        print("Profiling full training step...")
        profile_memory_run(forward_only=False)
        dest = f"./memory_profile_full_{name}"
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(f"./memory_profile_full", dest)


def main():
    parser = argparse.ArgumentParser(description="Unified Transformer Benchmarking")
    parser.add_argument("--use_torch", action="store_true", help="Use PyTorch backend")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision (torch only)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for PyTorch backend")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Run with torch profiler (for detailed analysis)")
    parser.add_argument("--flamegraph", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--benchmark_rmsnorm_torch", action="store_true")
    parser.add_argument("--forward_only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for benchmarking")
    parser.add_argument("--benchmark_norms", action="store_true", help="Benchmark LayerNorm vs RMSNorm")
    parser.add_argument("--benchmark_transformer", action="store_true", help="Benchmark Transformer with RMSNorm")
    parser.add_argument("--backward", action="store_true", help="Include backward pass in benchmarks")
    parser.add_argument("--task", choices=["rmsnorm", "transformer", "all"], default="all", help="Which benchmark to run (default: all)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--profile_memory", action="store_true", help="Run memory profile for 2.7B model")

    args = parser.parse_args()

    device = args.device
    seq_len = args.seq_len

    if args.profile_memory:
        run_memory_profiles_for_model_sizes()
        return

    if args.task in ["rmsnorm", "all"]:
        # Run RMSNorm benchmarks
        forward_results, full_results = analyze_rmsnorm_implementations()

    if args.task in ["transformer", "all"]:
        # Run transformer model benchmarks
        model_results = analyze_transformer_models(amp=args.amp)

    if not (args.benchmark_norms or args.benchmark_transformer):
        args.benchmark_norms = True
        args.benchmark_transformer = True

    # Benchmark norm layers
    if args.benchmark_norms:
        benchmark_rmsnorm(backward=args.backward, iterations=args.iterations)

    # Benchmark transformers with different RMSNorm implementations
    if args.benchmark_transformer:
        print("\n=== Benchmarking full transformer models ===")

        # PyTorch RMSNorm implementation
        pytorch_results = benchmark_transformer_with_rmsnorm(use_triton=False)
        
        # Triton RMSNorm implementation
        if torch.cuda.is_available():
            triton_results = benchmark_transformer_with_rmsnorm(use_triton=True)
            
            # Compare results
            if pytorch_results and triton_results:
                fwd_speedup = pytorch_results['forward'][0] / triton_results['forward'][0]
                bwd_speedup = pytorch_results['backward'][0] / triton_results['backward'][0]
                total_speedup = (pytorch_results['forward'][0] + pytorch_results['backward'][0]) / \
                               (triton_results['forward'][0] + triton_results['backward'][0])
                
                print("\nSpeedup Summary:")
                print(f"  Forward pass:  {fwd_speedup:.2f}x")
                print(f"  Backward pass: {bwd_speedup:.2f}x")
                print(f"  Total:         {total_speedup:.2f}x")

    benchmark_norm_layers(backward=not args.forward_only, iterations=args.iterations)

    if args.benchmark_rmsnorm_torch:
        benchmark_rmsnorm_torch()
        return

    if args.flamegraph:
        model = TorchBenchmark.Transformer(1600, 25, 48, 6400, 10000, seq_len).to(device)
        input_ids = TorchBenchmark.generate_random_data(8, seq_len, 10000, device)
        export_flame_graph(model, input_ids, device)
        return

    if args.compare and args.use_torch:
        def run_model_size_benchmarks(mixed_precision, device):
            sizes = [
                ("Small", 512, 8, 6, 2048),
                ("Medium", 768, 12, 12, 3072),
                ("Large", 1024, 16, 24, 4096)
            ]
            results = {}
            for name, d_model, n_heads, n_layers, d_ff in sizes:
                fwd, bwd = TorchBenchmark.run_benchmark(
                    batch_size=8,
                    seq_len=128,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    warmup=1,
                    steps=5,
                    device=device,
                    use_amp=mixed_precision
                )
                results[name] = (fwd, bwd)
            return results

        def compare_precision_results(fp32_results, mixed_results):
            for name in fp32_results:
                fwd_fp32, bwd_fp32 = fp32_results[name]
                fwd_amp, bwd_amp = mixed_results[name]
                fwd_speedup = sum(fwd_fp32) / sum(fwd_amp)
                bwd_speedup = sum(bwd_fp32) / sum(bwd_amp)
                print(f"\n{name} Model:")
                print(f"Forward speedup (AMP): {fwd_speedup:.2f}x")
                print(f"Backward speedup (AMP): {bwd_speedup:.2f}x")

        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, switching to CPU.")
            args.device = "cpu"

        fp32 = run_model_size_benchmarks(mixed_precision=False, device=args.device)
        amp = run_model_size_benchmarks(mixed_precision=True, device=args.device)
        compare_precision_results(fp32, amp)
        return  # Exit early after comparison


    def summarize(title, timings):
        avg = sum(timings) / len(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0
        print(f"{title}: {avg:.6f}s ± {std:.6f}s")

    if args.profile:
        logger.info("Profiling enabled - results will be saved to profiler_output.json")

    if args.use_torch:
        fwd, bwd = TorchBenchmark.run_benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            warmup=args.warmup_steps,
            steps=args.measure_steps,
            device=args.device,
            use_amp=args.mixed_precision
        )

        summarize("Forward Pass", fwd)
        summarize("Backward Pass", bwd)
        print(f"Total: {(sum(fwd) + sum(bwd)) / len(fwd):.6f}s")
    else:
        forward_timings, backward_timings = run_benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            backward=not args.forward_only
        )

        summarize("Forward Pass", forward_timings)
        summarize("Backward Pass", backward_timings)
        print(f"Total: {(sum(forward_timings) + sum(backward_timings)) / len(forward_timings):.6f}s")

if __name__ == "__main__":
    main()

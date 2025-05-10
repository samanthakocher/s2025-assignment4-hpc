#!/usr/bin/env python3
from __future__ import annotations

from typing import Type

import torch

from torch.autograd import Function

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def get_rmsnorm_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm.
    The expectation is that this class will implement RMSNorm
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyRMSNormAutogradFunctionClass
    class RMSNormFunction(Function):
        @staticmethod
        def forward(ctx, x, weight):
            """
            Forward pass for RMSNorm.
            
            Args:
                ctx: Context for saving variables for backward pass
                x: Input tensor of shape (..., H) where H is the hidden dimension
                weight: Learnable weight parameter of shape (H,)
                
            Returns:
                Normalized tensor of the same shape as x
            """
            # Calculate RMS along the last dimension
            # Keep dimensions for broadcasting
            variance = torch.mean(x**2, dim=-1, keepdim=True)
            rms = torch.sqrt(variance + 1e-8)  # Adding epsilon for numerical stability
            
            # Normalize
            x_normalized = x / rms
            
            # Scale with learnable parameters
            result = x_normalized * weight
            
            # Save variables needed for backward pass
            ctx.save_for_backward(x, rms, weight, x_normalized)
            
            return result
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass for RMSNorm.
            
            Args:
                ctx: Context with saved variables from forward pass
                grad_output: Gradient of the loss with respect to output
                
            Returns:
                Gradients with respect to input and weight
            """
            x, rms, weight, x_normalized = ctx.saved_tensors
            H = x.shape[-1] 

            # Gradient w.r.t. weight g
            grad_weight = torch.sum(grad_output * x_normalized, dim=tuple(range(grad_output.dim() - 1)))

            # Gradient w.r.t. x
            # Step 1: dy/dx = g / rms
            gx = grad_output * weight  # shape (..., H)

            # Step 2: dot product term
            dot = torch.sum(gx * x, dim=-1, keepdim=True) 

            # Step 3: final grad_input
            grad_input = (gx - x * dot / (rms**2 * H)) / rms

            return grad_input, grad_weight
    
    return RMSNormFunction


def get_rmsnorm_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_rmsnorm_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    if triton is None:
        raise ImportError("Triton not available.")

        # Define the Triton kernel for forward pass
    @triton.jit
    def rmsnorm_forward_kernel(
        x_ptr,  # Pointer to the input tensor
        weight_ptr,  # Pointer to the weight parameter 
        output_ptr,  # Pointer to the output tensor
        n_cols,  # Hidden dimension size
        stride_x_row,  # Stride for rows in input tensor
        stride_x_col,  # Stride for columns in input tensor
        stride_out_row,  # Stride for rows in output tensor
        stride_out_col,  # Stride for columns in output tensor
        BLOCK_SIZE: tl.constexpr  # Block size for processing
    ):
        """
        Triton kernel for RMSNorm forward pass.
        """
        # Get program ID
        row_idx = tl.program_id(0)
        
        # Calculate offsets
        x_row_offset = row_idx * stride_x_row
        out_row_offset = row_idx * stride_out_row
        
        # Create block-level offset arrays
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Calculate input and output pointers for current block
        x_ptrs = x_row_offset + col_offsets * stride_x_col
        
        # Load input values
        x_vals = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
        
        # Calculate squared values
        x_squared = x_vals * x_vals
        
        # Compute sum of squares
        sum_squares = tl.sum(x_squared, axis=0)
        
        # Calculate RMS (Root Mean Square)
        rms = tl.sqrt(sum_squares / n_cols + 1e-8)
        
        # Normalize
        x_normalized = x_vals / rms
        
        # Load weight values
        weight_vals = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
        
        # Apply weight
        output = x_normalized * weight_vals
        
        # Store output
        output_ptrs = out_row_offset + col_offsets * stride_out_col
        tl.store(output_ptr + output_ptrs, output, mask=mask)

    @triton.jit
    def rmsnorm_backward_kernel(
        x_ptr,           # Pointer to input x
        weight_ptr,      # Pointer to weight (g)
        grad_output_ptr, # Pointer to grad_output
        grad_input_ptr,  # Pointer to output grad_input
        grad_weight_ptr, # Pointer to output grad_weight
        n_cols,          # Number of features (H)
        stride_x_row,
        stride_x_col,
        stride_grad_out_row,
        stride_grad_out_col,
        stride_grad_in_row,
        stride_grad_in_col,
        BLOCK_SIZE: tl.constexpr
    ):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        # Compute memory addresses
        x_ptrs = x_ptr + row_id * stride_x_row + offsets * stride_x_col
        go_ptrs = grad_output_ptr + row_id * stride_grad_out_row + offsets * stride_grad_out_col
        gi_ptrs = grad_input_ptr + row_id * stride_grad_in_row + offsets * stride_grad_in_col

        # Load data
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        go = tl.load(go_ptrs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

        # Recompute RMS and normalized x
        rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + 1e-8)
        x_hat = x / rms

        # Compute grad_weight contribution
        grad_weight = go * x_hat
        tl.atomic_add(grad_weight_ptr + offsets, grad_weight, mask=mask)

        # Compute dot product of grad * x
        gx = go * w
        dot = tl.sum(gx * x, axis=0)

        # grad_input formula
        grad_input = (gx - x * dot / (rms * rms * n_cols)) / rms
        tl.store(gi_ptrs, grad_input, mask=mask)

    # Define the autograd Function class
    class RMSNormTritonFunction(Function):
        @staticmethod
        def forward(ctx, x, weight):
            """
            Forward pass for RMSNorm using Triton.
            
            Args:
                ctx: Context for saving variables for backward pass
                x: Input tensor of shape (..., H) where H is the hidden dimension
                weight: Learnable weight parameter of shape (H,)
                
            Returns:
                Normalized tensor of the same shape as x
            """
            # Get the input shape
            orig_shape = x.shape
            hidden_dim = orig_shape[-1]
            
            # Reshape input for triton kernel if needed
            if len(orig_shape) > 2:
                x = x.reshape(-1, hidden_dim)
            
            # Get tensor dimensions
            batch_size, n_cols = x.shape
            
            # Create output tensor
            output = torch.empty_like(x)
            
            # Calculate strides
            stride_x_row = x.stride(0)
            stride_x_col = x.stride(1) if x.stride(1) else 1
            stride_out_row = output.stride(0)
            stride_out_col = output.stride(1) if output.stride(1) else 1
            
            # Choose block size (usually a power of 2 less than or equal to n_cols)
            BLOCK_SIZE = triton.next_power_of_2(n_cols)
            
            # Launch kernel
            grid = (batch_size,)
            rmsnorm_forward_kernel[grid](
                x.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                n_cols,
                stride_x_row,
                stride_x_col,
                stride_out_row,
                stride_out_col,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Reshape output back to original shape if needed
            if len(orig_shape) > 2:
                output = output.reshape(orig_shape)

            x_flat = x.reshape(-1, hidden_dim)
            rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + 1e-8)
            x_normalized = x_flat / rms   

            # Save tensors needed for backward pass
            ctx.save_for_backward(x, weight, x_normalized)
            ctx.hidden_dim = hidden_dim
            ctx.orig_shape = orig_shape
            
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass for RMSNorm using Triton.
            
            Args:
                ctx: Context with saved variables from forward pass
                grad_output: Gradient of the loss with respect to output
                
            Returns:
                Gradients with respect to input and weight
            """        
            # Retrieve saved tensors and context
            x, weight, x_normalized = ctx.saved_tensors
            hidden_dim = ctx.hidden_dim
            orig_shape = ctx.orig_shape
            
            # Reshape grad_output if necessary
            if len(orig_shape) > 2:
                grad_output = grad_output.reshape(-1, hidden_dim)
            
            # Get tensor dimensions
            batch_size, n_cols = x_normalized.shape
            
            # Create output tensors for gradients
            grad_input = torch.empty_like(x_normalized)
            grad_weight = torch.zeros_like(weight)
            
            # Calculate strides
            stride_x_row = x_normalized.stride(0)
            stride_x_col = x_normalized.stride(1) if x_normalized.stride(1) else 1
            stride_grad_out_row = grad_output.stride(0)
            stride_grad_out_col = grad_output.stride(1) if grad_output.stride(1) else 1
            stride_grad_in_row = grad_input.stride(0)
            stride_grad_in_col = grad_input.stride(1) if grad_input.stride(1) else 1
            
            # Choose block size
            BLOCK_SIZE = triton.next_power_of_2(n_cols)
            
            # Launch backward kernel
            grid = (batch_size,)
            rmsnorm_backward_kernel[grid](
                x.data_ptr(),
                weight.data_ptr(),
                grad_output.data_ptr(),
                grad_input.data_ptr(),
                grad_weight.data_ptr(),
                n_cols,
                stride_x_row,
                stride_x_col,
                stride_grad_out_row,
                stride_grad_out_col,
                stride_grad_in_row,
                stride_grad_in_col,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            # Reshape grad_input back to original shape if needed
            if len(orig_shape) > 2:
                grad_input = grad_input.reshape(orig_shape)
            
            return grad_input, grad_weight
    
    return RMSNormTritonFunction


def rmsnorm_backward_g_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to g.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to g. Shape: (H,)
    """
    # Calculate the RMS (Root Mean Square) along the last dimension
    # Keep dimensions for proper broadcasting
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-8)
    
    # Normalized input
    x_norm = x / rms
    
    # For each element of g, the gradient is the sum of grad_output * x_norm
    # over all dimensions except the last one
    grad_g = torch.sum(grad_output * x_norm, dim=tuple(range(grad_output.dim() - 1)))
    
    return grad_g


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to x.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to x. Shape: (*, H)
    """
    # Get the shape of the last dimension
    feat_dim = x.shape[-1]
    
    # Calculate mean of squares
    ms = torch.mean(x * x, dim=-1, keepdim=True)
    
    # Add epsilon for numerical stability
    eps = 1e-8
    rsqrt_ms = torch.rsqrt(ms + eps)  # 1/sqrt(ms + eps)
    
    # Calculate normalized x
    normalized_x = x * rsqrt_ms
    
    # Gradient with respect to normalized_x
    grad_normalized_x = grad_output * g.view(*([1] * (x.dim() - 1)), feat_dim)
    
    # Gradient with respect to x
    # Let's break down the terms:
    # 1. First term: direct contribution via the chain rule
    first_term = grad_normalized_x * rsqrt_ms
    
    # 2. Second term: contribution via the effect on the normalization factor
    # ∂(ms)/∂(x_i) = 2*x_i/feat_dim
    # ∂(rsqrt_ms)/∂(ms) = -0.5 * (ms + eps)^(-3/2)
    # ∂(normalized_x)/∂(rsqrt_ms) = x
    second_term_factor = -0.5 * rsqrt_ms / (ms + eps)
    second_term = torch.sum(grad_normalized_x * x, dim=-1, keepdim=True) * second_term_factor * x * (2.0 / feat_dim)
    
    # Final gradient
    return first_term + second_term


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError

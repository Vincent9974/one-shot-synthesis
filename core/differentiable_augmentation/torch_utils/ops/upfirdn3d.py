# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 3d images."""

import os
import warnings
import numpy as np
import torch
import traceback

from .. import custom_ops
from .. import misc
from . import conv3d_gradfix

#----------------------------------------------------------------------------

_inited = False
_plugin = None

def _init():
    global _inited, _plugin
    if not _inited:
        sources = ['upfirdn3d.cpp', 'upfirdn3d.cu']
        sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
        try:
            _plugin = custom_ops.get_plugin('upfirdn3d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
        except:
            warnings.warn('Failed to build CUDA kernels for upfirdn3d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
    return _plugin is not None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy ,sz = scaling
    assert sx >= 1 and sy >= 1  and sz >= 1
    return sx, sy, sz

def _parse_padding(padding):
    #print(padding)
    if isinstance(padding, int):
        #print("padding is (int)")
        padding = [padding, padding, padding, padding, padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 3:
        padx, pady , padz = padding
        padding = [padx, padx, pady, pady, padz, padz]
        #print(padding)
    if len(padding) == 4:
        padx, pady , padz = padding[:3]
        padding = [padx, padx, pady, pady, padz, padz]
        #print(padding)
    #print(padding)
    padx0, padx1, pady0, pady1, padz0, padz1 = padding
    return padx0, padx1, pady0, pady1, padz0, padz1

def _get_filter_size(f):
    #print(f.shape)#12
    #print("f.shape:")
    #print(f.shape[-1])#12
    #print(f.shape[1])none
    #print(f.shape[0])12
    #print("f.ndim:")
    #print(f.ndim)
    if f is None:
        return 1, 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2, 3]
    fw = f.shape[-1]
    fh = f.shape[0]
    fd = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
        fd = int(fd)
    misc.assert_shape(f, [fd, fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1 and fd >= 1
    return fd, fw, fh,

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 3d FIR filter for `upfirdn3d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)
    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))

    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    #print("f:")
    #print(f)
    return f

#----------------------------------------------------------------------------

def upfirdn3d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 3d images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 3d FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        print("impl == 'cuda' and x.device.type == 'cuda' and _init()")
        return _upfirdn3d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn3d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn3d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn3d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 5
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    # print("x.shape")
    # print(x.shape) #[5,3,34,36,31]
    batch_size, num_channels, in_depth, in_height, in_width = x.shape
    upx, upy, upz = _parse_scaling(up)
    downx, downy, downz = _parse_scaling(down)
    #padx0, padx1, pady0, pady1 = _parse_padding(padding)
    padx0, padx1, pady0, pady1,padz0, padz1 = _parse_padding(padding)

    # Upsample by inserting zeros. 插值
    x = x.reshape([batch_size, num_channels, in_depth, 1, in_height, 1, in_width, 1])
    #print(in_depth,in_height,in_width)
    # print("x:")
    # print(x.shape) 【5，3，44，1，44，1，44，1】
    # print(upx-1,upy-1,upz-1)
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1, 0, 0, 0, upz-1])
    # print(x.shape)
    x = x.reshape([batch_size, num_channels,in_depth * upz, in_height * upy, in_width * upx])
    # print(x.shape) [5,3,44,44,44]

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0), max(padz0, 0), max(padz1, 0)])
    x = x[:, :, max(-pady0, 0): x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0), max(-padz0, 0): x.shape[4] - max(-padz1, 0)]
    #print("x.shape")
    #print(x.shape)

    # Setup filter.
    # print(f)
    f = f * (gain ** (f.ndim / 2))
    # print(f)
    # print(f)
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    # print(f)
    # Convolve with the filter.
    # print(f.ndim)
    # print(f[np.newaxis, np.newaxis, np.newaxis].shape)
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    # print(f.unsqueeze(2).shape)
    # print(f.unsqueeze(3).shape)
    # print(f.ndim)
    if f.ndim == 5:
        x = conv3d_gradfix.conv3d(input=x, weight=f, groups=num_channels)
    else:
        # print(f.shape)
        x = conv3d_gradfix.conv3d(input=x, weight=f.unsqueeze(2).unsqueeze(3), groups=num_channels)
        x = conv3d_gradfix.conv3d(input=x, weight=f.unsqueeze(2).unsqueeze(4), groups=num_channels)
        x = conv3d_gradfix.conv3d(input=x, weight=f.unsqueeze(3).unsqueeze(4), groups=num_channels)

    # print(x.shape)
    # Downsample by throwing away pixels.
    # print("downz")
    # print(downz)
    x = x[:, :, ::downz, ::downy, ::downx]
    # print(x.shape)
    # x = x[:, :, :16, :16, :16]
    # print('x.shape')
    # print(x.shape)
    return x

#----------------------------------------------------------------------------

_upfirdn3d_cuda_cache = dict()

def _upfirdn3d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn3d()` using custom ops.
    """
    # Parse arguments.
    upx, upy, upz = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, upz, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn3d_cuda_cache:
        return _upfirdn3d_cuda_cache[key]

    # Forward op.
    class Upfirdn3dCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn3d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn3d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
                y = _plugin.upfirdn3d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, np.sqrt(gain))
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn3d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn3d_cuda_cache[key] = Upfirdn3dCuda
    return Upfirdn3dCuda

#----------------------------------------------------------------------------

def filter3d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 3d images using the given 3d FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn3d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample3d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    print(len(x))
    r"""Upsample a batch of 3d images using the given 3d FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy, upz = _parse_scaling(up)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)

    fw, fh, fd= _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
        padz0 + (fd + upz - 1) // 2,
        padz1 + (fd - upz) // 2,
    ]
    return upfirdn3d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy*upz, impl=impl)

#----------------------------------------------------------------------------

def downsample3d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 3d images using the given 3d FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels,out_depth, out_height, out_width]`.
    """
    downx, downy, downz = _parse_scaling(down)
    padx0, padx1, pady0, pady1, padz0, padz1 = _parse_padding(padding)
    fd, fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
        padz0 + (fd - downz + 1) // 2,
        padz1 + (fd - downz) // 2,

    ]
    return upfirdn3d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

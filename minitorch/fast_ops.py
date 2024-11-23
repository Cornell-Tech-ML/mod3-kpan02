from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """Apply a unary function to each element in the input tensor and store result in output tensor.

        Args:
        ----
            out: Output storage array
            out_shape: Shape of output tensor
            out_strides: Strides of output tensor
            in_storage: Input storage array
            in_shape: Shape of input tensor
            in_strides: Strides of input tensor

        Optimizations:
        - Uses parallel processing with prange
        - Avoids indexing when strides match
        - Uses numpy arrays for indices

        """
        strides_match = np.array_equal(in_strides, out_strides)
        size = int(np.prod(out_shape))

        for i in prange(size):
            if strides_match:
                out[i] = fn(in_storage[i])

            else:
                out_index = np.zeros(MAX_DIMS, np.int32)
                in_index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        """Apply a binary function elementwise to two input tensors and store result in output tensor.

        Args:
        ----
            out: Output storage array
            out_shape: Shape of output tensor
            out_strides: Strides of output tensor
            a_storage: First input storage array
            a_shape: Shape of first input tensor
            a_strides: Strides of first input tensor
            b_storage: Second input storage array
            b_shape: Shape of second input tensor
            b_strides: Strides of second input tensor

        Optimizations:
        - Uses parallel processing with prange
        - Avoids indexing when strides match and shapes match
        - Uses numpy arrays for indices when broadcasting needed

        """
        strides_match = np.array_equal(a_strides, out_strides) and np.array_equal(
            b_strides, out_strides
        )
        shape_match = np.array_equal(a_shape, b_shape)
        size = int(np.prod(out_shape))

        for i in prange(size):
            if strides_match and shape_match:
                out[i] = fn(a_storage[i], b_storage[i])
            else:
                out_index = np.zeros(MAX_DIMS, np.int32)
                a_pos = np.zeros(MAX_DIMS, np.int32)
                b_pos = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_pos)
                j = index_to_position(a_pos, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_pos)
                k = index_to_position(b_pos, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """NUMBA tensor reduce inner loop function.

        This function performs the reduction operation along a specified dimension.

        Args:
        ----
            out (Storage): Output storage buffer
            out_shape (Shape): Shape of output tensor
            out_strides (Strides): Strides of output tensor
            a_storage (Storage): Storage of input tensor
            a_shape (Shape): Shape of input tensor
            a_strides (Strides): Strides of input tensor
            reduce_dim (int): Dimension to reduce along

        Optimizations:
        * Parallel outer loop
        * Inner loop reduction
        * Index buffers using numpy arrays

        """
        size = int(np.prod(out_shape))
        reduce_size = a_shape[reduce_dim]

        for i in prange(size):
            out_index: Index = np.zeros(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for j in prange(reduce_size):
                out_index[reduce_dim] = j
                p = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[p])

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    for n in prange(out_shape[0]):
        for i in range(out_shape[1]):  # Rows of a
            for j in range(out_shape[2]):  # Columns of b
                sum = 0
                a_pos = n * a_batch_stride + i * a_strides[1]
                b_pos = n * b_batch_stride + j * b_strides[2]
                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]

                for k in range(a_shape[-1]):  # Columns of a and rows of b
                    sum += a_storage[a_pos] * b_storage[b_pos]
                    a_pos += a_strides[2]
                    b_pos += b_strides[1]

                out[out_pos] = sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
FiniteDifferenceUtils base
"""

import enum
import typing
import numpy as np
import paddle


def _product(t):
    """
    product
    """
    if isinstance(t, int):
        return t
    else:
        return np.product(t)


def _get_item(t, idx):
    """
    get_item
    """
    assert isinstance(t, paddle.fluid.framework.Variable), "The first argument t must be Tensor."
    assert isinstance(idx, int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    return flat_t.__getitem__(idx)


def _set_item(t, idx, value):
    """
    set_item
    """
    assert isinstance(t, paddle.fluid.framework.Variable), "The first argument t must be Tensor."
    assert isinstance(idx, int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    flat_t.__setitem__(idx, value)
    return paddle.reshape(flat_t, t.shape)


def _as_tensors(xs):
    """
    as_tensors
    """
    return (xs,) if isinstance(xs, paddle.fluid.framework.Variable) else xs


def _compute_numerical_jacobian(func, xs, delta, np_dtype):
    """
    compute_numerical_jacobian
    """
    xs = list(_as_tensors(xs))
    ys = list(_as_tensors(func(*xs)))
    fin_size = len(xs)
    fout_size = len(ys)
    jacobian = list([] for _ in range(fout_size))
    for i in range(fout_size):
        jac_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            jac_i[j] = np.zeros((_product(ys[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        jacobian[i] = jac_i

    for j in range(fin_size):
        for q in range(_product(xs[j].shape)):
            orig = _get_item(xs[j], q)
            x_pos = orig + delta
            xs[j] = _set_item(xs[j], q, x_pos)
            ys_pos = _as_tensors(func(*xs))

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = _as_tensors(func(*xs))

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.0
    return jacobian


def concat_to_matrix(xs, is_batched=False):
    """Concats a tuple of tuple of Jacobian/Hessian matrix into one matrix"""
    rows = []
    for i in range(len(xs)):
        rows.append(np.concatenate([x for x in xs[i]], -1))
    return np.concatenate(rows, 1) if is_batched else np.concatenate(rows, 0)


def _compute_numerical_batch_jacobian(func, xs, delta, np_dtype, merge_batch=True):
    """
    compute_numerical_batch_jacobian
    """
    no_batch_jacobian = _compute_numerical_jacobian(func, xs, delta, np_dtype)
    xs = list(_as_tensors(xs))
    ys = list(_as_tensors(func(*xs)))
    fin_size = len(xs)
    fout_size = len(ys)
    bs = xs[0].shape[0]
    bat_jac = []
    for i in range(fout_size):
        batch_jac_i = []
        for j in range(fin_size):
            jac = no_batch_jacobian[i][j]
            jac_shape = jac.shape
            out_size = jac_shape[0] // bs
            in_size = jac_shape[1] // bs
            jac = np.reshape(jac, (bs, out_size, bs, in_size))
            batch_jac_i_j = np.zeros(shape=(out_size, bs, in_size))
            for p in range(out_size):
                for b in range(bs):
                    for q in range(in_size):
                        batch_jac_i_j[p][b][q] = jac[b][p][b][q]
            if merge_batch:
                batch_jac_i_j = np.reshape(batch_jac_i_j, (out_size, -1))
            batch_jac_i.append(batch_jac_i_j)
        bat_jac.append(batch_jac_i)

    return bat_jac


def _compute_numerical_hessian(func, xs, delta, np_dtype):
    """
    compute_numerical_hessian
    """
    xs = list(_as_tensors(xs))
    # ys = list(_as_tensors(func(*xs)))
    fin_size = len(xs)
    hessian = list([] for _ in range(fin_size))
    for i in range(fin_size):
        hessian_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            hessian_i[j] = np.zeros((_product(xs[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        hessian[i] = hessian_i

    for i in range(fin_size):
        for p in range(_product(xs[i].shape)):
            for j in range(fin_size):
                for q in range(_product(xs[j].shape)):
                    orig = _get_item(xs[j], q)
                    x_pos = orig + delta
                    xs[j] = _set_item(xs[j], q, x_pos)
                    jacobian_pos = _compute_numerical_jacobian(func, xs, delta, np_dtype)
                    x_neg = orig - delta
                    xs[j] = _set_item(xs[j], q, x_neg)
                    jacobian_neg = _compute_numerical_jacobian(func, xs, delta, np_dtype)
                    xs[j] = _set_item(xs[j], q, orig)
                    hessian[i][j][p][q] = (jacobian_pos[0][i][0][p] - jacobian_neg[0][i][0][p]) / delta / 2.0
    return hessian


def _compute_numerical_batch_hessian(func, xs, delta, np_dtype):
    """
    compute_numerical_batch_hessian
    """
    xs = list(_as_tensors(xs))
    batch_size = xs[0].shape[0]
    fin_size = len(xs)
    hessian = []
    for b in range(batch_size):
        x_l = []
        for j in range(fin_size):
            x_l.append(paddle.reshape(xs[j][b], shape=[1, -1]))
        hes_b = _compute_numerical_hessian(func, x_l, delta, np_dtype)
        if fin_size == 1:
            hessian.append(hes_b[0][0])
        else:
            hessian.append(hes_b)

    hessian_res = []
    for index in range(fin_size):
        x_reshape = paddle.reshape(xs[index], shape=[batch_size, -1])
        for index_ in range(fin_size):
            for i in range(x_reshape.shape[1]):
                tmp = []
                for j in range(batch_size):
                    if fin_size == 1:
                        tmp.extend(hessian[j][i])
                    else:
                        tmp.extend(hessian[j][i][index_][index])
                hessian_res.append(tmp)
        if fin_size == 1:
            return hessian_res
    hessian_result = []
    mid = len(hessian_res) // 2
    for i in range(mid):
        hessian_result.append(np.stack((hessian_res[i], hessian_res[mid + i]), axis=0))
    return hessian_result


MatrixFormat = enum.Enum("MatrixFormat", ("NBM", "BNM", "NMB", "NM"))


def _np_transpose_matrix_format(src, src_format, des_format):
    """Transpose Jacobian/Hessian matrix format."""
    supported_format = (MatrixFormat.NBM, MatrixFormat.BNM, MatrixFormat.NMB)
    if src_format not in supported_format or des_format not in supported_format:
        raise ValueError(
            f"Supported Jacobian format is {supported_format}, but got src: {src_format}, des: {des_format}"
        )

    src_axis = {c: i for i, c in enumerate(src_format.name)}
    dst_axis = tuple(src_axis[c] for c in des_format.name)

    return np.transpose(src, dst_axis)


def _np_concat_matrix_sequence(src, src_format=MatrixFormat.NM):
    """Convert a sequence of sequence of Jacobian/Hessian matrix into one huge
    matrix."""

    def concat_col(xs):
        if src_format in (MatrixFormat.NBM, MatrixFormat.BNM, MatrixFormat.NM):
            return np.concatenate(xs, axis=-1)
        else:
            return np.concatenate(xs, axis=1)

    def concat_row(xs):
        if src_format in (MatrixFormat.NBM, MatrixFormat.NM, MatrixFormat.NMB):
            return np.concatenate(xs, axis=0)
        else:
            return np.concatenate(xs, axis=1)

    supported_format = (MatrixFormat.NBM, MatrixFormat.BNM, MatrixFormat.NMB, MatrixFormat.NM)
    if src_format not in supported_format:
        raise ValueError(f"Supported Jacobian format is {supported_format}, but got {src_format}")
    if not isinstance(src, typing.Sequence):
        return src
    if not isinstance(src[0], typing.Sequence):
        src = [src]
    return concat_row(tuple(concat_col(xs) for xs in src))


class FiniteDifferenceUtils(object):
    """
    jac and hes FiniteDifferenceUtils
    jvp and vjp calculation with jacobian
    """

    def __init__(self):
        # self.delta = 1e-4
        pass

    def numerical_jacobian(self, func, xs, delta=1e-4, np_dtype="float64", is_batch=False):
        """
        numerical_jacobian
        """
        if not is_batch:
            jac = _compute_numerical_jacobian(func, xs, delta, np_dtype)
            return _np_concat_matrix_sequence(jac, MatrixFormat.NM)
        else:
            jac = _compute_numerical_batch_jacobian(func, xs, delta, np_dtype, False)
            jac = _np_concat_matrix_sequence(jac, MatrixFormat.NBM)
            return _np_transpose_matrix_format(jac, MatrixFormat.NBM, MatrixFormat.BNM)

    def numerical_hessian(self, func, xs, delta=1e-5, np_dtype="float64", is_batched=False):
        """
        numerical hessian
        """
        if not is_batched:
            numerical_hessian = _compute_numerical_hessian(func, xs, delta, np_dtype)
            return _np_concat_matrix_sequence(numerical_hessian)
        else:
            numerical_hessian = _compute_numerical_batch_hessian(func, xs, delta, np_dtype)
            return numerical_hessian

    def jvp_with_jac(self, func, xs, v=None):
        """
        calculate jvp with jacobian
        """
        if v is None:
            v = [paddle.ones_like(x) for x in xs]
        jacocian = paddle.incubate.autograd.Jacobian(func, xs)
        jac = jacocian[:]
        v1 = np.array([v_el.numpy().reshape(-1) for v_el in v])
        v2 = paddle.reshape(paddle.to_tensor(np.concatenate(v1)), (-1, 1))
        jvp = paddle.matmul(jac, v2).reshape((-1,))
        return jvp

    def vjp_with_jac(self, func, xs, v=None):
        """
        calculate vjp with jacobian
        """
        if v is None:
            v = paddle.ones_like(xs[0]) if isinstance(xs, list) else paddle.ones_like(xs)
            r = func(*xs) if isinstance(xs, list) else func(xs)
            if isinstance(r, (list, tuple)):
                v = paddle.to_tensor([v for _ in range(len(r))])
        jacocian = paddle.incubate.autograd.Jacobian(func, xs)
        jac = jacocian[:]
        v1 = np.array([v_el.numpy().reshape(-1) for v_el in v])
        v2 = paddle.reshape(paddle.to_tensor(np.concatenate(v1)), (1, -1)).astype("float64")
        vjp = paddle.matmul(v2, jac).reshape((-1,))
        return vjp.numpy()

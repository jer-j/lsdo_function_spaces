"""Batched evaluation: coefficients (*batch, *coefficients_shape, d) against a per-function loop."""

import csdl_alpha as csdl
import numpy as np
import pytest

import lsdo_function_spaces as lfs


@pytest.fixture
def recorder():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()


def _one_by_one(space, coefficients, u, order):
    """Each function of a (m, n, d) batch evaluated on its own (unbatched path, NumPy)."""
    return np.stack([space._evaluate(c, u, order) for c in coefficients])


@pytest.mark.parametrize("order", [None, (1,), (2,), 1, 2])
def test_batched_curves_match_one_by_one(recorder, order):
    rng = np.random.default_rng(0)
    space = lfs.BSplineSpace(1, (3,), (7,))
    coefficients = rng.normal(size=(4, 7, 2))
    u = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    expected = _one_by_one(space, coefficients, u, (order,) if isinstance(order, int) else order)

    numpy_values = space._evaluate(coefficients, u, order)
    csdl_values = space._evaluate(csdl.Variable(value=coefficients), u, order)

    assert numpy_values.shape == (4, 11, 2)
    np.testing.assert_allclose(numpy_values, expected, atol=1e-12)
    np.testing.assert_allclose(csdl_values.value, expected, atol=1e-12)


def test_batched_curves_at_csdl_parameters(recorder):
    rng = np.random.default_rng(1)
    space = lfs.BSplineSpace(1, (3,), (6,))
    coefficients = rng.normal(size=(3, 6, 2))
    u = np.linspace(0.05, 0.95, 9)
    expected = _one_by_one(space, coefficients, u.reshape(-1, 1), (1,))

    values = space._evaluate(csdl.Variable(value=coefficients), csdl.Variable(value=u), 1)

    np.testing.assert_allclose(values.value, expected, atol=1e-10)


def test_batched_surfaces_and_extra_batch_axes(recorder):
    rng = np.random.default_rng(2)
    space = lfs.BSplineSpace(2, (3, 2), (6, 4))
    coefficients = rng.normal(size=(2, 3, 6, 4, 3))
    uv = rng.uniform(size=(5, 2))
    flat = coefficients.reshape((6, 6, 4, 3))
    expected = _one_by_one(space, flat, uv, None).reshape((2, 3, 5, 3))

    values = space._evaluate(csdl.Variable(value=coefficients), uv)

    assert values.shape == (2, 3, 5, 3)
    np.testing.assert_allclose(values.value, expected, atol=1e-12)


def test_unbatched_layouts_are_unchanged(recorder):
    rng = np.random.default_rng(3)
    space = lfs.BSplineSpace(1, (3,), (7,))
    coefficients = rng.normal(size=(7, 2))
    u = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

    assert space._batch_shape(coefficients) == ()
    assert space._batch_shape(coefficients.reshape(-1)) == ()
    assert space._evaluate(coefficients, u).shape == (5, 2)


def test_curves_classmethod_and_derivatives(recorder):
    rng = np.random.default_rng(4)
    points = csdl.Variable(value=rng.normal(size=(3, 5, 2)))
    curves = lfs.BSplineSpace.curves(points, degree=3)
    u = np.linspace(0.0, 1.0, 7)

    values = curves.evaluate(u)
    assert values.shape == (3, 7, 2)

    # The values are linear in the control points: d(sum values)/d(points) is the basis column sum.
    gradient = csdl.derivative(csdl.sum(values), points)
    basis = curves.space.compute_basis_matrix(u.reshape(-1, 1)).toarray()
    expected = np.broadcast_to(basis.sum(axis=0)[None, :, None], (3, 5, 2))
    np.testing.assert_allclose(gradient.value.reshape((3, 5, 2)), expected, atol=1e-12)


def test_int_derivative_order_needs_a_1d_space(recorder):
    space = lfs.BSplineSpace(2, (2, 2), (4, 4))
    with pytest.raises(ValueError):
        space._evaluate(np.zeros((4, 4, 3)), np.zeros((1, 2)), 1)

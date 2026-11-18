"""B-spline function space representation and operations (pure Python)."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import csdl_alpha as csdl
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from scipy.spatial import cKDTree

import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.function_space import LinearFunctionSpace
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_csdl_custom_ops_v2 import (
    BasisMatrixCustomOp,
    BSplineEvalCustomOp,
)
from lsdo_function_spaces.core.spaces.non_cython_bsplines.b_spline_patch_projection import (
    compute_point_to_bspline_projection,
)
from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy import (
    compute_basis_matrix_numpy,
)


class BSplineSpace(LinearFunctionSpace):
    """B-spline Function Space for curves, surfaces, and trivariate volumes.

    Inherits from :class:`LinearFunctionSpace`. Pure Python implementation
    accelerated with NumPy, SciPy, and JAX (no Cython required).

    Parameters
    ----------
    num_parametric_dimensions : int
        The number of parametric dimensions (1 for curve, 2 for surface, 3 for volume).
    degree : Union[int, Tuple[int, ...]]
        Polynomial degree of the B-spline basis in each parametric dimension.
    coefficients_shape : Tuple[int, ...]
        Shape of control points / coefficients in each parametric dimension.
    knots : Optional[Union[Tuple[np.ndarray, ...], np.ndarray]], optional
        Knot vectors for each parametric dimension. If None, open uniform knot
        vectors on [0, 1] are automatically generated.
    knot_indices : Optional[List[np.ndarray]], optional
        Indices of knots per dimension (maintained for backwards compatibility).
    """

    def __init__(
        self,
        num_parametric_dimensions: int,
        degree: Union[int, Tuple[int, ...]],
        coefficients_shape: Tuple[int, ...],
        knots: Optional[Union[Tuple[np.ndarray, ...], np.ndarray]] = None,
        knot_indices: Optional[Sequence[np.ndarray]] = None,
    ):
        self.degree = degree
        self.knots = knots
        self.knot_indices = list(knot_indices) if knot_indices is not None else None
        super().__init__(num_parametric_dimensions, coefficients_shape)

        if isinstance(self.degree, int):
            self.degree = (self.degree,) * self.num_parametric_dimensions

        for i in range(self.num_parametric_dimensions):
            if self.degree[i] < 0:
                raise ValueError(f"Degree in axis {i} must be non-negative.")
            if self.degree[i] >= self.coefficients_shape[i]:
                raise ValueError(
                    f"Degree in axis {i} must be less than the number of coefficients in each dimension."
                )

        # Handle 1D concatenated knot vectors for backward compatibility
        if self.knots is not None and isinstance(self.knots, np.ndarray) and self.knots.ndim == 1:
            split_knots = []
            idx = 0
            for i in range(self.num_parametric_dimensions):
                n_knots = self.coefficients_shape[i] + self.degree[i] + 1
                split_knots.append(self.knots[idx : idx + n_knots])
                idx += n_knots
            self.knots = tuple(split_knots)
        elif self.knots is not None and isinstance(self.knots, (list, tuple)):
            self.knots = tuple(np.asarray(k, dtype=float) for k in self.knots)

        if self.knots is None:
            # Create open uniform knot vectors on [0, 1] for each dimension
            self.knots = tuple(
                np.concatenate([
                    np.zeros(self.degree[i]),
                    np.linspace(0, 1, self.coefficients_shape[i] - self.degree[i] + 1),
                    np.ones(self.degree[i]),
                ])
                for i in range(self.num_parametric_dimensions)
            )

        if self.knot_indices is None:
            self.knot_indices = []
            knot_index = 0
            for i in range(self.num_parametric_dimensions):
                num_knots_i = self.coefficients_shape[i] + self.degree[i] + 1
                self.knot_indices.append(np.arange(knot_index, knot_index + num_knots_i))
                knot_index += num_knots_i

    @property
    def parametric_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """Returns the valid parametric bounds (u_min, u_max) for each parametric dimension."""
        bounds = []
        for i in range(self.num_parametric_dimensions):
            u_min = float(self.knots[i][self.degree[i]])
            u_max = float(self.knots[i][-self.degree[i] - 1])
            bounds.append((u_min, u_max))
        return tuple(bounds)

    def _evaluate(
        self,
        coefficients: Union[np.ndarray, csdl.Variable],
        parametric_coordinates: Union[np.ndarray, csdl.Variable],
        parametric_derivative_orders: Optional[Tuple[int, ...]] = None,
    ) -> Union[np.ndarray, csdl.Variable]:
        """Evaluate B-spline functions at the given parametric coordinates.

        Coefficients of shape ``(*batch, *coefficients_shape, d)``, with one or more
        leading batch axes, hold several functions of this space; they are evaluated
        together with one basis matrix and give values of shape ``(*batch, q, d)``. A
        derivative order may be given as an int for a one-dimensional space.
        """
        if isinstance(parametric_derivative_orders, (int, np.integer)):
            if self.num_parametric_dimensions != 1:
                raise ValueError("an int parametric_derivative_orders is only accepted for a 1D space; "
                                 "give one order per parametric dimension")
            parametric_derivative_orders = (int(parametric_derivative_orders),)

        if not isinstance(coefficients, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"coefficients must be a numpy array or a CSDL variable, "
                f"but got type {type(coefficients)}."
            )

        if not isinstance(parametric_coordinates, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"parametric_coordinates must be a numpy array or a CSDL variable, "
                f"but got type {type(parametric_coordinates)}."
            )

        try:
            parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
        except ValueError:
            raise ValueError(
                f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                f"but got shape {parametric_coordinates.shape}."
            )

        non_csdl = isinstance(coefficients, np.ndarray)

        batch_shape = self._batch_shape(coefficients)
        if batch_shape:
            return self._evaluate_batch(coefficients, parametric_coordinates, parametric_derivative_orders,
                                        batch_shape)

        if isinstance(parametric_coordinates, np.ndarray):
            basis_matrix = compute_basis_matrix_numpy(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=parametric_derivative_orders,
            )
            if coefficients.shape != (basis_matrix.shape[1], coefficients.size // basis_matrix.shape[1]):
                coefficients = coefficients.reshape(
                    (basis_matrix.shape[1], coefficients.size // basis_matrix.shape[1])
                )

            if non_csdl:
                values = basis_matrix @ coefficients
                if values.shape[0] == 1:
                    values = values.flatten()
            else:
                c_flat = coefficients.reshape((basis_matrix.shape[1], coefficients.shape[-1]))
                values = csdl.sparse.matmat(basis_matrix, c_flat)
                values = values.reshape((parametric_coordinates.shape[0], coefficients.shape[-1]))

            return values

        else:
            b_spline_eval_op = BSplineEvalCustomOp(
                knots=self.knots,
                degree=self.degree,
                coefficients_shape=self.coefficients_shape,
                der_orders=parametric_derivative_orders,
            )

            values = b_spline_eval_op.evaluate(
                parametric_coordinates=parametric_coordinates,
                coefficients=coefficients,
            )

            if non_csdl:
                values = values.value

            return values

    def _batch_shape(self, coefficients: Union[np.ndarray, csdl.Variable]) -> Tuple[int, ...]:
        """The leading batch axes of ``coefficients`` of shape ``(*batch, *coefficients_shape, d)``.

        Empty for the unbatched layouts: ``(*coefficients_shape, d)`` or any flattened form.
        """
        shape = tuple(coefficients.shape)
        k = len(self.coefficients_shape)
        if len(shape) > k + 1 and shape[-(k + 1):-1] == tuple(self.coefficients_shape):
            return shape[:-(k + 1)]
        return ()

    def _evaluate_batch(self, coefficients, parametric_coordinates, parametric_derivative_orders, batch_shape):
        """Evaluate a batch of functions of this space with one basis matrix.

        Parameters
        ----------
        coefficients : np.ndarray or csdl.Variable, shape (*batch, *coefficients_shape, d)
        parametric_coordinates : np.ndarray or csdl.Variable, shape (q, num_parametric_dimensions)
        parametric_derivative_orders : tuple of int or None
        batch_shape : tuple of int

        Returns
        -------
        np.ndarray or csdl.Variable, shape (*batch, q, d)
        """
        n = int(np.prod(self.coefficients_shape))
        m = int(np.prod(batch_shape))
        d = coefficients.shape[-1]
        q = parametric_coordinates.shape[0]
        basis = self.compute_basis_matrix(parametric_coordinates, parametric_derivative_orders)   # (q, n)
        if isinstance(coefficients, np.ndarray) and not isinstance(basis, csdl.Variable):
            # (n, m d): every function's coordinates are columns of one product
            columns = np.moveaxis(coefficients.reshape((m, n, d)), 1, 0).reshape((n, m * d))
            values = np.moveaxis(np.asarray(basis @ columns).reshape((q, m, d)), 1, 0)
            return values.reshape((*batch_shape, q, d))
        if isinstance(coefficients, np.ndarray):
            coefficients = csdl.Variable(value=coefficients)
        if not isinstance(basis, csdl.Variable):
            basis = basis.toarray()
        values = csdl.einsum(basis, coefficients.reshape((m, n, d)), action="qn,mnd->mqd")
        return values.reshape((*batch_shape, q, d))

    @classmethod
    def curves(cls, control_points, degree: int = 3, knots: Optional[np.ndarray] = None) -> lfs.Function:
        """B-spline curves of one space, as one Function: one curve, or a batch of them.

        Parameters
        ----------
        control_points : csdl.Variable or array_like, shape (n, d) or (*batch, n, d)
            The control points of each curve. Arrays become CSDL variables (which needs an
            active recorder).
        degree : int, optional
            Polynomial degree, less than n.
        knots : array_like, shape (n + degree + 1,), optional
            Knot vector; the default is clamped and uniform on [0, 1].

        Returns
        -------
        lfs.Function
            A function of a one-dimensional space with n coefficients; evaluated at q
            parameters it gives shape (q, d), or (*batch, q, d) for a batch.
        """
        if not isinstance(control_points, csdl.Variable):
            control_points = csdl.Variable(value=np.asarray(control_points, dtype=float))
        n = control_points.shape[-2]
        if knots is not None:
            knots = (np.asarray(knots, dtype=float),)
        space = cls(num_parametric_dimensions=1, degree=(degree,), coefficients_shape=(n,), knots=knots)
        return lfs.Function(space=space, coefficients=control_points)

    def _generate_parametric_grid(self, knot_vectors: Sequence[np.ndarray], N: int) -> np.ndarray:
        """Generate a tensor-grid of parametric sample points."""
        samples_1d = []
        for U in knot_vectors:
            knots = np.unique(U)
            pts = []
            for j in range(len(knots) - 1):
                a, b = knots[j], knots[j + 1]
                pts.append(np.linspace(a, b, N, endpoint=False))
            pts.append(np.array([knots[-1]]))
            samples_1d.append(np.concatenate(pts))

        mesh = np.meshgrid(*samples_1d, indexing="ij")
        coord_arrays = [m.flatten() for m in mesh]
        grid = np.stack(coord_arrays, axis=-1)
        return grid

    def _project(
        self,
        points_in_space: Union[np.ndarray, csdl.Variable],
        coefficients: Union[np.ndarray, csdl.Variable],
        plot: bool = False,
        grid_search_density: int = 100,
    ) -> np.ndarray:
        """Project points in physical space onto the B-spline entity."""
        if isinstance(coefficients, csdl.Variable):
            coefficients = coefficients.value

        if not isinstance(points_in_space, (np.ndarray, csdl.Variable)):
            raise TypeError(
                f"points_in_space must be a numpy array or a CSDL variable, "
                f"but got type {type(points_in_space)}."
            )
        if isinstance(points_in_space, csdl.Variable):
            raise NotImplementedError(
                "Projection of CSDL variables is not implemented yet. "
                "Please provide a numpy array of points in space."
            )
        fun = lfs.Function(
            space=self,
            coefficients=coefficients,
        )

        para_grid = self._generate_parametric_grid(
            knot_vectors=self.knots,
            N=grid_search_density,
        )

        basis_mat = compute_basis_matrix_numpy(
            us=para_grid,
            degrees=self.degree,
            knot_vectors=self.knots,
        )

        surface_grid = basis_mat @ coefficients.reshape(-1, coefficients.shape[-1])

        kd_tree = cKDTree(surface_grid)
        nearest_index = kd_tree.query(points_in_space, k=1)[1]
        nearest_para_points = para_grid[nearest_index]

        batched_projection = jax.jit(
            jax.vmap(
                lambda pt, u0, cps: compute_point_to_bspline_projection(
                    point=pt,
                    degrees=self.degree,
                    coefficients=cps,
                    para_coords=u0,
                    knots=tuple(tuple(float(x) for x in kv_i) for kv_i in self.knots),
                ),
                in_axes=(0, 0, None),
            )
        )

        para, res, converged, final_i, J, _, _ = batched_projection(
            points_in_space,
            nearest_para_points,
            coefficients,
        )
        para = np.array(para).reshape(-1, self.num_parametric_dimensions)

        if not converged.all():
            warnings.warn(
                f"{np.sum(~converged)} out of {len(converged)} projection points did not fully converge.",
                UserWarning,
                stacklevel=2,
            )

        if plot:
            point_cloud = lfs.plot_points(
                points=points_in_space,
                color="#00FF1A",
                opacity=0.5,
                size=8,
                show=False,
            )

            projected_points = fun.evaluate(
                parametric_coordinates=para,
            ).value
            project_point_cloud = lfs.plot_points(
                points=projected_points,
                color="#FF0000",
                size=4,
                show=False,
            )

            fun.plot(additional_plotting_elements=[point_cloud, project_point_cloud])

        return para

    def compute_basis_matrix(
        self,
        parametric_coordinates: Union[np.ndarray, csdl.Variable],
        parametric_derivative_orders: Optional[Tuple[int, ...]] = None,
        expansion_factor: Optional[int] = None,
    ) -> Union[sps.coo_matrix, csdl.Variable]:
        """Compute the B-spline basis matrix for given parametric coordinates."""
        if isinstance(parametric_coordinates, csdl.Variable):
            basis_mat_custom_op = BasisMatrixCustomOp(
                knots=self.knots,
                degree=self.degree,
                coefficients_shape=self.coefficients_shape,
                der_orders=parametric_derivative_orders,
            )
            try:
                parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
            except ValueError:
                raise ValueError(
                    f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                    f"but got shape {parametric_coordinates.shape}."
                )
            return basis_mat_custom_op.evaluate(parametric_coordinates)

        elif isinstance(parametric_coordinates, np.ndarray):
            try:
                parametric_coordinates = parametric_coordinates.reshape(-1, self.num_parametric_dimensions)
            except ValueError:
                raise ValueError(
                    f"parametric_coordinates must have shape (num_points, {self.num_parametric_dimensions}), "
                    f"but got shape {parametric_coordinates.shape}."
                )

            res = compute_basis_matrix_numpy(
                us=parametric_coordinates,
                degrees=self.degree,
                knot_vectors=self.knots,
                der_orders=parametric_derivative_orders,
            )
            if expansion_factor is not None and expansion_factor > 1:
                res = sps.kron(res, sps.eye(expansion_factor), format='csr')
            return res

        else:
            raise TypeError(
                f"parametric_coordinates must be a numpy array or a CSDL variable, "
                f"but got type {type(parametric_coordinates)}."
            )

    def _compute_distance_bounds(
        self, point: np.ndarray, function: lfs.Function, direction: Optional[np.ndarray] = None
    ) -> float:
        """Compute distance bounds for a given point relative to function bounding box."""
        if not hasattr(function, "bounding_box"):
            coefficients = function.coefficients.value.reshape((-1, function.num_physical_dimensions))
            function.bounding_box = np.zeros((2, coefficients.shape[-1]))
            if self.num_parametric_dimensions == 1:
                function.bounding_box[0, 0] = np.min(coefficients)
                function.bounding_box[1, 0] = np.max(coefficients)
            else:
                function.bounding_box[0, :] = np.min(coefficients, axis=0)
                function.bounding_box[1, :] = np.max(coefficients, axis=0)

        if direction is None:
            neg = function.bounding_box[0] - point
            pos = point - function.bounding_box[1]
            distance_vector = np.maximum(np.maximum(neg, pos), 0)
            return float(np.linalg.norm(distance_vector))
        else:
            closest_point = np.zeros((len(point),))
            for i in range(len(point)):
                if point[i] < function.bounding_box[0, i]:
                    closest_point[i] = function.bounding_box[0, i]
                elif point[i] > function.bounding_box[1, i]:
                    closest_point[i] = function.bounding_box[1, i]
                else:
                    closest_point[i] = point[i]
            t = np.dot(direction, (closest_point - point)) / np.dot(direction, direction)
            closest_point_on_line = point + t * direction
            return float(np.linalg.norm(closest_point_on_line - closest_point))

    def stitch(self, self_face: int, self_coeffs: np.ndarray, other: BSplineSpace, other_face: int, other_coeffs: np.ndarray):
        """Stitch two B-spline function spaces along adjacent faces."""
        ind_array = np.arange(np.prod(self.coefficients_shape)).reshape(self.coefficients_shape)

        if len(self_coeffs.shape) > 2:
            self_coeffs = self_coeffs.reshape((-1, self_coeffs.shape[-1]))
        if len(other_coeffs.shape) > 2:
            other_coeffs = other_coeffs.reshape((-1, other_coeffs.shape[-1]))

        if self_face == 1:
            self_inds = ind_array[:, 0]
        elif self_face == 2:
            self_inds = ind_array[-1, :]
        elif self_face == 3:
            self_inds = ind_array[:, -1]
        elif self_face == 4:
            self_inds = ind_array[0, :]
        self_inds = [int(ind) for ind in self_inds]

        if other_face == 1:
            other_inds = ind_array[:, 0]
        elif other_face == 2:
            other_inds = ind_array[-1, :]
        elif other_face == 3:
            other_inds = ind_array[:, -1]
        elif other_face == 4:
            other_inds = ind_array[0, :]
        other_inds = [int(ind) for ind in other_inds]

        return self_inds, other_inds


class BSplineSpaceNew(BSplineSpace):
    """Deprecated alias for :class:`BSplineSpace`.

    .. deprecated:: 1.0.0
        Use :class:`BSplineSpace` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BSplineSpaceNew is deprecated; use BSplineSpace instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

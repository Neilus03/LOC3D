"""Executed RGBN6 Mask3D geometry and covector normal transforms."""
import numpy as np
from delimit3d.data import point_augmentations as _aug

def _trilinear_displacement_jacobian(
    noise: np.ndarray,
    axes: list[np.ndarray],
    points: np.ndarray,
) -> np.ndarray:
    """Return analytic derivatives of the sampled trilinear vector field."""
    count = points.shape[0]
    indices: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    spacings: list[np.ndarray] = []
    valid = np.ones(count, dtype=bool)
    for axis_index, axis in enumerate(axes):
        if not np.all(np.diff(axis) > 0):
            raise ValueError("Elastic interpolation axes must be increasing")
        valid &= (points[:, axis_index] >= axis[0]) & (
            points[:, axis_index] <= axis[-1]
        )
        index = np.searchsorted(axis, points[:, axis_index], side="right") - 1
        index = np.clip(index, 0, len(axis) - 2)
        spacing = axis[index + 1] - axis[index]
        fraction = (points[:, axis_index] - axis[index]) / spacing
        indices.append(index)
        fractions.append(np.clip(fraction, 0.0, 1.0).astype(np.float32))
        spacings.append(spacing)

    gradient = np.zeros((count, 3, 3), dtype=np.float32)
    for input_axis in range(3):
        other_axes = [axis for axis in range(3) if axis != input_axis]
        for first_bit in (0, 1):
            for second_bit in (0, 1):
                bits_low = [0, 0, 0]
                bits_high = [0, 0, 0]
                bits_low[other_axes[0]] = bits_high[other_axes[0]] = first_bit
                bits_low[other_axes[1]] = bits_high[other_axes[1]] = second_bit
                bits_high[input_axis] = 1
                weight = np.ones(count, dtype=np.float32)
                for axis, bit in zip(other_axes, (first_bit, second_bit), strict=True):
                    fraction = fractions[axis]
                    weight *= fraction if bit else (1.0 - fraction)
                low = noise[
                    indices[0] + bits_low[0],
                    indices[1] + bits_low[1],
                    indices[2] + bits_low[2],
                ]
                high = noise[
                    indices[0] + bits_high[0],
                    indices[1] + bits_high[1],
                    indices[2] + bits_high[2],
                ]
                gradient[:, :, input_axis] += (
                    (high - low) / spacings[input_axis][:, None]
                ) * weight[:, None]
    gradient[~valid] = 0.0
    if not np.isfinite(gradient).all():
        raise ValueError("Elastic displacement Jacobian is non-finite")
    return gradient


def _elastic_distortion_with_aligned_normals(
    coord: np.ndarray,
    normals: np.ndarray,
    distortion_params: list[list[float]],
) -> None:
    """Apply Mask3D elastic warps and their local inverse-transpose to normals.

    Coordinate generation and RNG consumption intentionally mirror the archived
    RGB3 implementation. The only added computation evaluates the analytic
    Jacobian of the already-sampled trilinear displacement field.
    """
    if not _aug._HAS_SCIPY or not distortion_params:
        return
    if _aug.random.random() > 0.95:
        return

    blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3.0
    blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3.0
    blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3.0
    coords_min = coord.min(0)

    for granularity, magnitude in distortion_params:
        noise_dim = ((coord - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        for _ in range(2):
            noise = _aug._ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = _aug._ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = _aug._ndimage.convolve(noise, blurz, mode="constant", cval=0)
        axes = [
            np.linspace(d_min, d_max, count, dtype=np.float32)
            for d_min, d_max, count in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = _aug._interp.RegularGridInterpolator(
            axes, noise, bounds_error=False, fill_value=0
        )
        sample_points = coord.copy()
        displacement = interp(sample_points)
        jacobian = np.broadcast_to(
            np.eye(3, dtype=np.float32), (coord.shape[0], 3, 3)
        ).copy()
        jacobian += float(magnitude) * _trilinear_displacement_jacobian(
            noise, axes, sample_points
        )
        coord[:] = sample_points + displacement * float(magnitude)
        try:
            inverse = np.linalg.inv(jacobian)
        except np.linalg.LinAlgError:
            raise ValueError("Elastic deformation has a singular local Jacobian") from None
        condition = np.linalg.norm(jacobian, ord=np.inf, axis=(1, 2)) * np.linalg.norm(
            inverse, ord=np.inf, axis=(1, 2)
        )
        if not np.isfinite(condition).all() or np.any(condition > 1.0e4):
            raise ValueError("Elastic deformation has an ill-conditioned local Jacobian")
        normals[:] = np.einsum("ni,nij->nj", normals, inverse)
        _aug._normalize_normals_inplace(normals)


def augment_points_mask3d_scannet_rgbn6_aligned(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    use_colors: bool = True,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Use the RGB3 Mask3D transforms while keeping RGBN6 normals aligned."""
    coord = np.asarray(points, dtype=np.float32).copy()
    if colors is not None and use_colors:
        color = np.asarray(colors, dtype=np.float32).copy()
        if color.size and color.max() > 1.0:
            color /= 255.0
    else:
        color = None
    if normals is None:
        raise ValueError("The matched RGBN6 profile requires per-point normals")
    nrm = np.asarray(normals, dtype=np.float32).copy()
    if nrm.shape != coord.shape:
        raise ValueError(f"points/normals shape mismatch: {coord.shape} vs {nrm.shape}")
    _aug._normalize_normals_inplace(nrm)

    if coord.shape[0] > 0:
        coord -= coord.mean(axis=0, keepdims=True)
        coord += np.random.uniform(coord.min(axis=0), coord.max(axis=0)) / 2.0
        _aug._random_flip_mask3d(coord, p=0.5, normals=nrm)
        _elastic_distortion_with_aligned_normals(
            coord, nrm, [[0.2, 0.4], [0.8, 1.6]]
        )
        _aug._random_anisotropic_scale(coord, (0.9, 1.1), p=0.5, normals=nrm)
        _aug._random_rotate(
            coord, (-1.0, 1.0), "z", [0.0, 0.0, 0.0], p=0.5, normals=nrm
        )
        _aug._random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "y",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _aug._random_rotate(
            coord,
            (-1.0 / 24.0, 1.0 / 24.0),
            "x",
            [0.0, 0.0, 0.0],
            p=0.5,
            normals=nrm,
        )
        _aug._normalize_normals_inplace(nrm)
    if color is not None:
        _aug._mask3d_color_augment_float01(color)
    return coord, color, nrm

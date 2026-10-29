"""Unit-quaternion attitude utilities (scalar-first, Hamilton convention).

The attitude quaternion q rotates body-frame vectors into the inertial frame:
    v_inertial = R(q) @ v_body
Quaternions have no singularities, so the rocket can point anywhere (including
straight up, where Euler yaw/roll are undefined) without gimbal lock. Euler-like
angles are only ever derived for output, never integrated.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix body -> inertial."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < -1.0 + 1e-12:
        # 180 deg: pick any axis orthogonal to a
        ortho = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(a, [0.0, 1.0, 0.0])
        return from_axis_angle(ortho, np.pi)
    q = np.array([1.0 + d, *np.cross(a, b)])
    return normalize(q)


def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """dq/dt for body angular velocity omega_body (rad/s)."""
    return 0.5 * multiply(q, np.array([0.0, *omega_body]))

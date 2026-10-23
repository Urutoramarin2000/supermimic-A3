"""Differentiable rotation helpers (rotation matrices, wxyz quaternions)."""

from __future__ import annotations

import torch


def mm3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """3x3 matrix product as multiply + sum (fuses; batched bmm of tiny matrices is launch/latency bound)."""
  return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def mv3(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return (a * v.unsqueeze(-2)).sum(-1)


def tr(a: torch.Tensor) -> torch.Tensor:
  return a.transpose(-1, -2)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True)
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
  """Non-differentiable-safe conversion (used for writing simulator states)."""
  m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
  w = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-12)) / 2
  x = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-12)) / 2
  y = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-12)) / 2
  z = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-12)) / 2
  x = torch.copysign(x, m[..., 2, 1] - m[..., 1, 2])
  y = torch.copysign(y, m[..., 0, 2] - m[..., 2, 0])
  z = torch.copysign(z, m[..., 1, 0] - m[..., 0, 1])
  q = torch.stack([w, x, y, z], -1)
  return q / q.norm(dim=-1, keepdim=True)


def skew(v: torch.Tensor) -> torch.Tensor:
  x, y, z = v.unbind(-1)
  o = torch.zeros_like(x)
  return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(v.shape[:-1] + (3, 3))


def exp_map(r: torch.Tensor) -> torch.Tensor:
  """Rotation vector -> rotation matrix (Rodrigues), smooth at zero."""
  t2 = (r * r).sum(-1, keepdim=True)
  t = torch.sqrt(t2 + 1e-12)
  small = t2 < 1e-8
  a = torch.where(small, 1 - t2 / 6, torch.sin(t) / t)
  b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(t)) / (t2 + 1e-12))
  k = skew(r)
  eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(k.shape)
  return eye + a[..., None] * k + b[..., None] * mm3(k, k)


def log_map(m: torch.Tensor) -> torch.Tensor:
  """Rotation matrix -> rotation vector. Accurate away from pi; used for targets and errors."""
  v = torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
  s = 0.5 * torch.sqrt((v * v).sum(-1) + 1e-12)
  c = 0.5 * (m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1)
  theta = torch.atan2(s, c)
  ratio = torch.where(s < 1e-6, torch.ones_like(s), theta / s)
  return 0.5 * ratio[..., None] * v


def angle_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Squared geodesic angle between rotation matrices, smooth at zero (||log(a b^T)||^2)."""
  return (log_map(mm3(a, tr(b))) ** 2).sum(-1)


def gram_schmidt(m: torch.Tensor) -> torch.Tensor:
  a, b = m[..., :, 0], m[..., :, 1]
  x = a / a.norm(dim=-1, keepdim=True)
  b = b - (x * b).sum(-1, keepdim=True) * x
  y = b / b.norm(dim=-1, keepdim=True)
  z = torch.cross(x, y, dim=-1)
  return torch.stack([x, y, z], -1)


def yaw_of(m: torch.Tensor) -> torch.Tensor:
  return torch.atan2(m[..., 1, 0], m[..., 0, 0])


def yaw_mat(yaw: torch.Tensor) -> torch.Tensor:
  c, s = torch.cos(yaw), torch.sin(yaw)
  o, i = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack([c, -s, o, s, c, o, o, o, i], -1).reshape(yaw.shape + (3, 3))


def rot6d(m: torch.Tensor) -> torch.Tensor:
  return m[..., :, :2].reshape(m.shape[:-2] + (6,))

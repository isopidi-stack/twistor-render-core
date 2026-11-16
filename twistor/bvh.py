import torch
from twistor.renderer import TwistorRenderer

class BVHNode:
    """
    Node in a recursive Axis-Aligned Bounding Box (AABB) Bounding Volume Hierarchy.
    """
    __slots__ = ("bbox_min", "bbox_max", "point_indices", "left", "right")

    def __init__(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor, point_indices: torch.Tensor = None, left=None, right=None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.point_indices = point_indices
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.point_indices is not None


class TwistorBVH:
    """
    Genuine Recursive Axis-Aligned Bounding Box (AABB) Bounding Volume Hierarchy:
      - Recursively partitions 3D points along the longest AABB axis at the spatial median.
      - Expands each node AABB by the configurable Gaussian cutoff pad (pad = cutoff_mult * sigma, default 4.5 * sigma)
        so any ray within cutoff_mult * sigma of a point is guaranteed to intersect the padded AABB.
      - Uses the Kay-Kajiya slab method to cull subtrees and evaluates the Gaussian intensity model
        truncated at r_perp <= cutoff_mult * sigma (tail weight < 4e-5 at 4.5*sigma, < 3e-7 at 5.5*sigma).
    """
    def __init__(self, pts_3d: torch.Tensor, leaf_size: int = 16, sigma: float = 0.15, cutoff_mult: float = 4.5):
        self.pts_3d = pts_3d
        self.leaf_size = leaf_size
        self.sigma = sigma
        self.pad = sigma * cutoff_mult
        N = pts_3d.shape[0]
        all_indices = torch.arange(N, dtype=torch.long, device=pts_3d.device)
        self.root = self._build_recursive(all_indices)

    def _build_recursive(self, indices: torch.Tensor) -> BVHNode:
        sub_pts = self.pts_3d[indices]
        bbox_min = torch.min(sub_pts, dim=0).values - self.pad
        bbox_max = torch.max(sub_pts, dim=0).values + self.pad

        if indices.numel() <= self.leaf_size:
            return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, point_indices=indices)

        # Split along the axis of maximum spatial extent
        extents = bbox_max - bbox_min
        split_axis = int(torch.argmax(extents).item())
        coords = sub_pts[:, split_axis]
        sorted_order = torch.argsort(coords)
        mid = indices.numel() // 2

        left_indices = indices[sorted_order[:mid]]
        right_indices = indices[sorted_order[mid:]]

        left_node = self._build_recursive(left_indices)
        right_node = self._build_recursive(right_indices)
        return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, left=left_node, right=right_node)

    @staticmethod
    def rays_intersect_aabb(ray_origin: torch.Tensor, ray_dirs: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
        """
        Kay-Kajiya vectorized slab intersection test between M rays (ray_origin, ray_dirs)
        and a 3D Axis-Aligned Bounding Box [bbox_min, bbox_max].
        Returns a boolean mask of shape (M,) indicating which rays intersect the box for t >= 0.
        """
        # Avoid division by zero for axis-parallel rays
        safe_dirs = torch.where(torch.abs(ray_dirs) > 1e-12, ray_dirs, torch.full_like(ray_dirs, 1e-12))
        inv_d = 1.0 / safe_dirs  # (M, 3)

        t0 = (bbox_min.unsqueeze(0) - ray_origin.unsqueeze(0)) * inv_d  # (M, 3)
        t1 = (bbox_max.unsqueeze(0) - ray_origin.unsqueeze(0)) * inv_d  # (M, 3)

        t_smaller = torch.minimum(t0, t1)
        t_larger  = torch.maximum(t0, t1)

        t_enter = torch.max(t_smaller, dim=-1).values  # (M,)
        t_exit  = torch.min(t_larger, dim=-1).values   # (M,)

        return (t_exit >= torch.clamp(t_enter, min=0.0)) & (t_exit >= 0.0)

    def query_candidate_points_for_ray(self, ray_origin: torch.Tensor, ray_dir: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Traverses the AABB BVH tree for a single ray (or tight ray bundle), pruning subtrees
        whose bounding boxes do not intersect the ray. Returns (candidate_indices, nodes_visited).
        """
        ray_dir_2d = ray_dir.unsqueeze(0) if ray_dir.ndim == 1 else ray_dir
        collected = []
        nodes_visited = 0
        stack = [self.root]

        while stack:
            node = stack.pop()
            nodes_visited += 1
            hit_mask = self.rays_intersect_aabb(ray_origin, ray_dir_2d, node.bbox_min, node.bbox_max)
            if not torch.any(hit_mask):
                continue
            if node.is_leaf:
                collected.append(node.point_indices)
            else:
                stack.append(node.right)
                stack.append(node.left)

        if len(collected) == 0:
            return torch.empty(0, dtype=torch.long, device=self.pts_3d.device), nodes_visited
        return torch.cat(collected, dim=0), nodes_visited

    def render(self, camera_center: torch.Tensor, ray_dirs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Renders the scene using recursive BVH traversal. For each leaf AABB hit by a subset
        of rays, evaluates the exact same Gaussian intensity formula as TwistorRenderer.render
        only on the intersecting (ray_subset, leaf_points) pairs.
        """
        M = ray_dirs.shape[0]
        n_unit = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        intensities = torch.zeros(M, dtype=self.pts_3d.dtype, device=self.pts_3d.device)

        active_ray_indices = torch.arange(M, dtype=torch.long, device=self.pts_3d.device)
        stack = [(self.root, active_ray_indices)]
        nodes_visited = 0
        leaf_ray_point_evals = 0

        while stack:
            node, ray_idx = stack.pop()
            nodes_visited += 1
            sub_dirs = n_unit[ray_idx]
            hit_mask = self.rays_intersect_aabb(camera_center, sub_dirs, node.bbox_min, node.bbox_max)
            if not torch.any(hit_mask):
                continue

            hit_ray_idx = ray_idx[hit_mask]
            if node.is_leaf:
                leaf_pts = self.pts_3d[node.point_indices]  # (K_leaf, 3)
                hit_dirs = n_unit[hit_ray_idx]               # (M_hit, 3)
                leaf_ray_point_evals += hit_ray_idx.numel() * leaf_pts.shape[0]

                d_vec = leaf_pts - camera_center.unsqueeze(0)  # (K_leaf, 3)
                dist_sq = torch.sum(d_vec ** 2, dim=-1).unsqueeze(0)  # (1, K_leaf)
                dot_dn = torch.einsum("mc,nc->mn", hit_dirs, d_vec)   # (M_hit, K_leaf)

                forward_mask = dot_dn > 0.0
                r_perp_sq = torch.clamp(dist_sq - dot_dn ** 2, min=0.0)
                cutoff_mask = forward_mask & (r_perp_sq <= (self.pad ** 2))
                w = torch.exp(-r_perp_sq / (2.0 * (self.sigma ** 2))) / (1.0 + dist_sq)
                w = torch.where(cutoff_mask, w, torch.zeros_like(w))
                intensities[hit_ray_idx] += torch.sum(w, dim=-1)
            else:
                stack.append((node.right, hit_ray_idx))
                stack.append((node.left, hit_ray_idx))

        total_brute_force_pairs = M * self.pts_3d.shape[0]
        pruned_ratio = 1.0 - (float(leaf_ray_point_evals) / max(total_brute_force_pairs, 1))
        stats = {
            "nodes_visited": nodes_visited,
            "leaf_ray_point_evals": leaf_ray_point_evals,
            "brute_force_pairs": total_brute_force_pairs,
            "pruned_fraction": pruned_ratio,
        }
        return intensities, stats
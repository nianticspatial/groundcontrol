import math
from collections import defaultdict
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.decomposition import PCA


def _generate_single_overlay_image(attention_coeffs, base_image):
    """
    Helper to generate a single attention overlay image and return it as a NumPy array.

    Args:
        attention_coeffs (torch.Tensor): Attention coefficients for one item (H, W, head, z_v).
        base_image (torch.Tensor): The base image for the overlay (C, H, W).

    Returns:
        np.ndarray: The rendered overlay image as a NumPy array (H, W, 3) in BGR format.
    """
    # --- 1. Calculate Expected Z-Value Map ---
    attention_avg_heads = attention_coeffs.mean(dim=2)
    num_z = attention_avg_heads.shape[2]
    z_values = torch.linspace(-1, 1, num_z, device=attention_coeffs.device)
    z_values_reshaped = z_values.view(1, 1, -1)
    weighted_sum = (attention_avg_heads * z_values_reshaped).sum(dim=2)
    sum_of_weights = attention_avg_heads.sum(dim=2)
    expected_z_map_low_res = weighted_sum / (sum_of_weights + 1e-8)

    # --- 2. Upsample Attention Map ---
    img_h, img_w = base_image.shape[1], base_image.shape[2]
    expected_z_map_4d = expected_z_map_low_res.unsqueeze(0).unsqueeze(0)
    upsampled_attention_map = F.interpolate(
        expected_z_map_4d, size=(img_h, img_w), mode="bilinear", align_corners=False
    )
    upsampled_attention_np = upsampled_attention_map.squeeze().cpu().detach().numpy()

    # --- 3. Render Matplotlib figure to a NumPy array ---
    image_for_plot = base_image.permute(1, 2, 0).cpu().detach().numpy()
    image_for_plot = np.clip(image_for_plot, 0, 1)

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")

    ax.imshow(image_for_plot)
    ax.imshow(upsampled_attention_np, cmap="jet", alpha=0.5)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    overlay_rgba = np.asarray(buf)
    plt.close(fig)  # Important to free up memory

    overlay_bgr = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGR)

    return overlay_bgr


def _generate_single_overhead_map(
    normalized_attention_slice, overhead_image_slice, ray_points_slice, cmap
):
    """Helper function to generate a single overhead image with attention points."""
    img_np = np.copy(overhead_image_slice)
    W, num_z = normalized_attention_slice.shape
    for w_idx in range(W):
        for z_idx in range(num_z):
            attn_value = normalized_attention_slice[w_idx, z_idx]
            color_rgba = cmap(attn_value)
            color_bgr = (
                int(color_rgba[2] * 255),
                int(color_rgba[1] * 255),
                int(color_rgba[0] * 255),
            )
            point_xy = (ray_points_slice[w_idx, z_idx, 0], ray_points_slice[w_idx, z_idx, 1])
            cv2.circle(img_np, point_xy, radius=2, color=color_bgr, thickness=-1)
    return img_np


def visualize_attention_grid(
    attention_coefficients_batch,
    overhead_image_tensor,
    ray_points_batch,
    overlay_base_image_batch,
    new_column_image_batch,
    num_rows_to_visualize=3,
):
    """
    Generates a grid visualizing attention in multiple ways for a batch of data.

    Args:
        attention_coefficients_batch (torch.Tensor): 5D tensor (B, H, W, head, z_v).
        overhead_image_tensor (torch.Tensor): 3D tensor for the single overhead map (C, H_img, W_img).
        ray_points_batch (torch.Tensor): 4D tensor for ray points (B, W, z_v, 2).
        overlay_base_image_batch (torch.Tensor): 4D tensor for overlay base images (B, C, H_base, W_base).
        new_column_image_batch (torch.Tensor): 4D tensor for a new column of images (B, C, H_new, W_new).
        num_rows_to_visualize (int): Number of camera rows to show as overhead maps. Default is 3.

    Returns:
        np.ndarray: A single grid image composed of the visualizations.
    """
    cmap = plt.get_cmap("jet")
    all_grid_rows = []

    batch_size = attention_coefficients_batch.shape[0]

    # Prepare the SINGLE base overhead image once
    overhead_img_np = overhead_image_tensor.cpu().permute(1, 2, 0).numpy()
    overhead_img_np = (
        (overhead_img_np * 0.5 + 0.5) if overhead_img_np.min() < 0 else overhead_img_np
    )
    base_overhead_map = np.ascontiguousarray(overhead_img_np * 255).astype(np.uint8)
    target_h, _, _ = base_overhead_map.shape

    for b in range(batch_size):
        # --- Slice data for the current batch item ---
        attention_coeffs = attention_coefficients_batch[b]
        ray_points = ray_points_batch[b]
        overlay_base_image = overlay_base_image_batch[b]
        new_column_image = new_column_image_batch[b]

        # --- Helper function to prepare and resize an image for the grid ---
        def prepare_image_for_grid(img_tensor):
            img_np = img_tensor.float().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 0.5 + 0.5) if img_np.min() < 0 else np.clip(img_np, 0, 1)
            img_np_uint8 = np.ascontiguousarray(img_np * 255).astype(np.uint8)

            h, w, _ = img_np_uint8.shape
            if h != target_h:
                new_w = int(w * (target_h / h))
                return cv2.resize(img_np_uint8, (new_w, target_h), interpolation=cv2.INTER_AREA)
            return img_np_uint8

        # --- 1. Assemble the first three columns ---
        # Column 1: Base Image
        base_image_col = prepare_image_for_grid(overlay_base_image)

        # Column 2: Overlay Image
        overlay_image = _generate_single_overlay_image(attention_coeffs, overlay_base_image)
        h_overlay, w_overlay, _ = overlay_image.shape
        if h_overlay != target_h:
            new_w_overlay = int(w_overlay * (target_h / h_overlay))
            overlay_image = cv2.resize(
                overlay_image, (new_w_overlay, target_h), interpolation=cv2.INTER_AREA
            )

        # Column 3: New Custom Image
        new_image_col = prepare_image_for_grid(new_column_image)

        current_grid_row_images = [base_image_col, overlay_image, new_image_col]

        # --- 2. Generate Overhead Maps (Remaining Columns) ---
        mean_attention = attention_coeffs.mean(dim=2)
        mean_attention_np = mean_attention.float().cpu().detach().numpy()
        ray_points_np = ray_points.float().cpu().detach().numpy().astype(np.int32)

        min_attn, max_attn = mean_attention_np.min(), mean_attention_np.max()
        normalized_attention = (mean_attention_np - min_attn) / (max_attn - min_attn + 1e-8)

        H, W, _ = mean_attention.shape
        row_indices = np.linspace(0, H - 1, num_rows_to_visualize, dtype=int)

        for h_idx in row_indices:
            attention_slice = normalized_attention[h_idx]
            overhead_map = _generate_single_overhead_map(
                attention_slice, base_overhead_map, ray_points_np, cmap
            )
            current_grid_row_images.append(overhead_map)

        grid_row = np.hstack(current_grid_row_images)
        all_grid_rows.append(grid_row)

    final_grid = np.vstack(all_grid_rows)
    return final_grid


def sample_ray_points(batch, min_depth_px=2, max_depth_px=100, num_points=16):
    # Pick a row of the image, and sample points along a ray for each pixel.
    B, N, C, H, W = batch["images"].shape
    latents_W = W // 8
    u = torch.arange(latents_W, device=batch["images"].device)[None, None]
    intrinsics = batch["intrinsics"].clone()
    intrinsics = intrinsics.to(batch["images"].device)
    intrinsics_scale = 1.0 / 8
    intrinsics[..., :2, :3] *= intrinsics_scale
    x = (u - intrinsics[:, :, 0, 2, None]) / intrinsics[:, :, 0, 0, None]  # B, N, latents_W
    x = x.unsqueeze(-1).repeat(1, 1, 1, num_points)  # B, N, latents_W, num_points
    z_vals = torch.linspace(min_depth_px, max_depth_px, num_points, device=batch["images"].device)
    z_vals = z_vals[None, None, None, :].repeat(B, N, latents_W, 1)  # B, N, latents_W, num_points
    # need a -ve sign here, because in the satellite image, z axis is upwards
    ray_points = (
        torch.stack([x, -torch.ones_like(x)], dim=-1) * z_vals[..., None]
    )  # B, N, latents_W, num_points, 2
    return ray_points


def get_ray_features(batch, config):
    # Get the features for the ray points.
    ray_points = sample_ray_points(
        batch,
        min_depth_px=config.model.sample_points_min_depth,
        max_depth_px=config.model.sample_points_max_depth,
    )
    cam_poses = batch["camera_poses"].to(ray_points.device)
    ref_rot = batch["reference_rotation"].to(ray_points.device)
    latent_H = batch["images"].shape[-2] // 8

    view_ref_R_cam = ref_rot[:, None].transpose(-1, -2) @ cam_poses[..., :3, :3]  # B, N, 3, 3
    z_rot_clock = torch.atan2(view_ref_R_cam[:, :, 0, 2], view_ref_R_cam[:, :, 2, 2])
    # Compute the rotation from the pp to the image center
    img_w = batch["images"].shape[-1]
    x_angle = torch.atan2(
        (batch["intrinsics"][:, :, 0, 2] - img_w / 2), batch["intrinsics"][:, :, 0, 0]
    ).to(ray_points.device)

    img_center_rot_clock = z_rot_clock - x_angle
    # img_center_rot_clock = torch.zeros_like(img_center_rot_clock)
    c, s = torch.cos(img_center_rot_clock), torch.sin(img_center_rot_clock)
    rot_mat = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-1
    )  # B, N, 2, 2

    ray_points = (
        rot_mat[:, :, None, None] @ ray_points[..., None]
    )  # B, N, latents_W, num_points, 2, 1
    ray_points = ray_points.squeeze(-1)  # B, N, latents_W, num_points, 2
    sat_res = batch["satellite_images"].shape[-1]
    ray_points = ray_points + torch.tensor([sat_res / 2.0, sat_res / 2.0], device=ray_points.device)
    return ray_points


class ActivationExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = defaultdict(list)
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name].append(output.detach().float().cpu().numpy())

        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def get_activations(self):
        return self.activations

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()

    def clear_activations(self):
        self.activations = defaultdict(list)


def softmax(x, axis=-1):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def compute_entropy(attn_weight):
    probs = softmax(attn_weight, axis=-1)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
    return entropy


def compute_entropy_maps(query, key, latent_s, num_views):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1])
    attn_weight = query @ key.transpose(0, 2, 1) * scale_factor
    entropy = compute_entropy(attn_weight)
    entropy = rearrange(entropy, "(b h w) n -> b n h w", h=latent_s, w=latent_s, n=num_views)
    return entropy


def project_activations_to_pca_numpy(
    activations_list: List[np.ndarray],
    k: int = 3,
    share_basis: bool = False,
    scale_factor: float = 1.0,
) -> List[np.ndarray]:
    """
    Computes PCA, reshapes, upsamples (with PyTorch), and then projects data.

    This version uses NumPy and Scikit-learn, with a high-performance PyTorch
    backend for the upsampling step, which can leverage a CUDA GPU if available.

    Args:
        activations_list:
            A list of NumPy arrays. Each array is expected to have a shape of
            (B, N, C), where B is batch size, N is number of items (must be a
            perfect square), and C is the feature dimension.
        k:
            The number of top principal components to project onto. Defaults to 3.
        share_basis:
            If True, a single PCA basis is computed from all activations combined
            and then applied to each. If False (default), PCA is computed
            independently for each activation array.
        scale_factor:
            The factor by which to upscale the spatial dimensions (sqrt(N), sqrt(N))
            before the PCA transform. Defaults to 1.0 (no upsampling).

    Returns:
        A list of NumPy arrays. Each array has a shape of
        (B, sqrt(N)*scale_factor, sqrt(N)*scale_factor, k).
    """
    processed_activations_list = []

    # Determine device for PyTorch operations
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Processing {len(activations_list)} activation array(s) with share_basis={share_basis}..."
    )
    if scale_factor != 1.0:
        print(f"PyTorch will use device: '{device}' for upsampling.")

    pca_model = PCA(n_components=k)

    if share_basis:
        # --- 1a. Combine all data to fit a single PCA model ---
        print("Combining all activations to compute a shared PCA basis...")
        all_reshaped_activations = []
        for i, activation_arr in enumerate(activations_list):
            if activation_arr.ndim != 3:
                raise ValueError(
                    f"Array {i}: Expected 3D array (B, N, C), but got shape {activation_arr.shape}."
                )
            B, N, C = activation_arr.shape
            all_reshaped_activations.append(activation_arr.reshape(B * N, C))

        combined_data = np.concatenate(all_reshaped_activations, axis=0)

        if combined_data.shape[0] < k:
            raise ValueError(
                f"Combined samples ({combined_data.shape[0]}) are less than k ({k}). Cannot compute shared PCA."
            )

        print(f"Fitting shared PCA on combined data with shape {combined_data.shape}...")
        pca_model.fit(combined_data)

    # --- 2. Process each activation array ---
    for i, activation_arr in enumerate(activations_list):
        B, N, C = activation_arr.shape
        num_samples = B * N

        # --- 2a. Validate shape for reshaping ---
        side_length = math.isqrt(N)
        if side_length * side_length != N:
            raise ValueError(
                f"Array {i}: The 'N' dimension ({N}) must be a perfect square " "for reshaping."
            )

        if num_samples < k:
            print(
                f"Warning: Array {i} with shape {activation_arr.shape} has fewer samples ({num_samples}) than k ({k}). Skipping."
            )
            new_side = int(side_length * scale_factor)
            processed_activations_list.append(
                np.empty((B, new_side, new_side, 0), dtype=activation_arr.dtype)
            )
            continue

        # --- 2b. Fit the PCA model (if not using a shared basis) ---
        if not share_basis:
            X_fit = activation_arr.reshape(num_samples, C)
            pca_model.fit(X_fit)

        # --- 2c. Reshape original data to image-like format ---
        image_arr_c = activation_arr.reshape(B, side_length, side_length, C)

        # --- 2d. Upsample the high-dimensional reshaped array using PyTorch ---
        if scale_factor != 1.0:
            # Convert to tensor and move to device
            image_tensor_c = torch.from_numpy(image_arr_c).to(device)
            # Permute from (B, H, W, C) -> (B, C, H, W) for PyTorch interpolate
            image_tensor_c = image_tensor_c.permute(0, 3, 1, 2)

            # Upsample using bicubic interpolation
            upsampled_tensor_c = F.interpolate(
                image_tensor_c, scale_factor=scale_factor, mode="bicubic", align_corners=False
            )

            # Permute back from (B, C, H_new, W_new) -> (B, H_new, W_new, C)
            upsampled_tensor_c = upsampled_tensor_c.permute(0, 2, 3, 1)

            # Move back to CPU and convert to NumPy
            upsampled_arr_c = upsampled_tensor_c.cpu().numpy()
        else:
            upsampled_arr_c = image_arr_c

        # --- 2e. Reshape upsampled data for the transform step ---
        new_b, new_h, new_w, new_c = upsampled_arr_c.shape
        X_transform = upsampled_arr_c.reshape(new_b * new_h * new_w, new_c)

        # --- 2f. Transform the upsampled data using the fitted model ---
        X_projected = pca_model.transform(X_transform)

        # --- 2g. Reshape the final projected data into image format ---
        final_arr = X_projected.reshape(new_b, new_h, new_w, k)

        processed_activations_list.append(final_arr)

    return processed_activations_list


def visualize_ray_points(
    img,
    ray_points,
    is_normalized=False,
    attention_values: np.ndarray | None = None,
    cmap_name: str = "jet",
):
    img_np = img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_np = np.ascontiguousarray(img_np * 255).astype(np.uint8)

    if isinstance(ray_points, torch.Tensor):
        rp = ray_points.detach().cpu().numpy()
    else:
        rp = np.asarray(ray_points)

    if rp.ndim == 3:
        rp = rp.reshape(-1, rp.shape[-1])
    if is_normalized:
        img_half_res = img_np.shape[0] // 2
        rp = rp * img_half_res + img_half_res
    rp = rp.astype(np.int32)

    attn_flat = None
    if attention_values is not None:
        attn_flat = attention_values.reshape(-1)
        attn_norm = (attn_flat - attn_flat.min()) / (attn_flat.max() - attn_flat.min())
        cmap = plt.get_cmap(cmap_name)
        colors = (cmap(attn_norm) * 255).astype(np.uint8).tolist()
    else:
        colors = [(0, 255, 0)] * rp.shape[0]

    for idx in range(min(rp.shape[0], len(colors))):
        point = rp[idx]
        img_np = cv2.circle(img_np, (point[0], point[1]), 1, colors[idx], -1)
    return img_np


def get_fov_triangle_points(batch, device, fov_depth=50, camera_pos_depth=2.0):
    """
    Computes FOV triangle vertices for each camera view.

    The triangle is formed by:
    - Camera position (projected to satellite image)
    - Left edge of FOV at fov_depth
    - Right edge of FOV at fov_depth

    Args:
        batch: Batch dictionary with camera poses, intrinsics, etc.
        device: Device to perform computations on
        fov_depth: Depth at which to compute the FOV triangle edges (in pixels)
        camera_pos_depth: Depth for camera position vertex (in pixels)

    Returns:
        torch.Tensor: Triangle vertices of shape (B, N, 3, 2) where last dim is (x, y) in satellite image coords
    """
    B, N = batch["images"].shape[:2]
    intrinsics = batch["intrinsics"].clone().to(device)
    img_w = batch["images"].shape[-1]

    # Get camera center (principal point) and focal length
    cx = intrinsics[:, :, 0, 2]  # B, N
    fx = intrinsics[:, :, 0, 0]  # B, N

    # Use pixel centers (0.5 offset) to match ray sampling code
    # Left edge: x = 0.5 (pixel center of first pixel)
    # Right edge: x = img_w - 0.5 (pixel center of last pixel)
    # Center: use the principal point cx (which is the actual optical center)
    left_x = torch.full_like(cx, 0.5)
    right_x = torch.full_like(cx, img_w - 0.5)
    center_x = cx  # Use principal point as the center

    # Convert to normalized camera coordinates
    left_x_norm = (left_x - cx) / fx  # B, N
    right_x_norm = (right_x - cx) / fx  # B, N
    center_x_norm = (center_x - cx) / fx  # B, N (should be 0)

    # Create 3D points at fov_depth: (x_norm * depth, y_norm * depth, depth)
    # Using the same coordinate system as _sample_ray_points which uses (x, -1) format
    left_point = torch.stack(
        [left_x_norm * fov_depth, -torch.ones_like(left_x_norm) * fov_depth], dim=-1
    )  # B, N, 2
    right_point = torch.stack(
        [right_x_norm * fov_depth, -torch.ones_like(right_x_norm) * fov_depth], dim=-1
    )  # B, N, 2

    # Camera position (origin in camera frame, but we need to project it)
    # For the triangle, we'll use the center point at a closer depth as the "camera position"
    # Use center_x_norm (which is 0) to ensure it's aligned with the principal point
    camera_pos = torch.stack(
        [center_x_norm * camera_pos_depth, -torch.ones_like(cx) * camera_pos_depth], dim=-1
    )  # B, N, 2

    # Transform points using the same logic as _get_transformed_ray_points
    cam_poses = batch["camera_poses"].to(device)
    ref_rot = batch["reference_rotation"].to(device)

    view_ref_R_cam = ref_rot[:, None].transpose(-1, -2) @ cam_poses[..., :3, :3]  # B, N, 3, 3
    z_rot_clock = torch.atan2(view_ref_R_cam[:, :, 0, 2], view_ref_R_cam[:, :, 2, 2])

    # Match the ray sampling code exactly - no principal point offset adjustment
    # (The commented code in _get_transformed_ray_points shows x_angle = 0)
    img_center_rot_clock = z_rot_clock
    c, s = torch.cos(img_center_rot_clock), torch.sin(img_center_rot_clock)
    rot_mat = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-1
    )  # B, N, 2, 2

    # Transform all points
    def transform_point(point):
        # point: B, N, 2
        point_transformed = (rot_mat @ point.unsqueeze(-1)).squeeze(-1)  # B, N, 2
        sat_res = batch["satellite_images"].shape[-1]
        point_transformed = point_transformed + torch.tensor(
            [sat_res / 2.0, sat_res / 2.0], device=point_transformed.device
        )
        # Apply translation offset if available
        if "reference_translation" in batch:
            center_translation = batch["reference_translation"][:, None, :2].to(
                point_transformed.device
            )  # B, 1, 2
            translation_offset = cam_poses[..., :2, 3] - center_translation  # B, N, 2
            translation_offset = translation_offset * 4  # convert to pixels
            point_transformed = point_transformed + torch.stack(
                [translation_offset[..., 0], -translation_offset[..., 1]], dim=-1
            )
        return point_transformed

    left_point_transformed = transform_point(left_point)
    right_point_transformed = transform_point(right_point)
    camera_pos_transformed = transform_point(camera_pos)

    # Compute image center axis direction (ray through image center, not principal point)
    # Image center in pixel coordinates
    img_h = batch["images"].shape[-2]
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0

    # Convert image center to normalized camera coordinates
    img_center_x_norm = (img_center_x - cx) / fx  # B, N
    cy = intrinsics[:, :, 1, 2]  # B, N
    fy = intrinsics[:, :, 1, 1]  # B, N
    img_center_y_norm = (img_center_y - cy) / fy  # B, N

    # Create a point along the image center ray at arrow_length distance
    # In the 2D projection used here, y is fixed at -1, so we use img_center_x_norm
    arrow_length = fov_depth * 0.7  # Arrow length as fraction of fov_depth
    image_center_axis_point = torch.stack(
        [img_center_x_norm * arrow_length, -torch.ones_like(cx) * arrow_length], dim=-1
    )  # B, N, 2
    image_center_axis_point_transformed = transform_point(image_center_axis_point)

    # Compute arrow direction vector (from camera_pos to image_center_axis_point)
    image_center_axis_direction = (
        image_center_axis_point_transformed - camera_pos_transformed
    )  # B, N, 2

    # Stack to form triangles: (camera_pos, left_point, right_point)
    triangles = torch.stack(
        [camera_pos_transformed, left_point_transformed, right_point_transformed], dim=2
    )  # B, N, 3, 2

    return triangles, image_center_axis_direction


def visualize_fov_triangles(
    img,
    fov_triangles,
    image_center_axis_directions,
    input_mask,
    target_view_idx,
    input_color=(255, 0, 0),
    target_color=(0, 255, 0),
    alpha=0.3,
    arrow_length=20,
):
    """
    Visualizes camera FOV triangles on a satellite image with translucency.

    Args:
        img: Satellite image tensor (C, H, W) in range [-1, 1]
        fov_triangles: Array of triangle vertices of shape (N, 3, 2) with (x, y) coordinates
        image_center_axis_directions: Array of image center axis direction vectors of shape (N, 2)
        input_mask: Boolean array of shape (N,) indicating which views are input views
        target_view_idx: Index of the target view to highlight
        input_color: BGR color for input view triangles (default: red)
        target_color: BGR color for target view triangle (default: green)
        alpha: Transparency factor for filled triangles (0.0 = fully transparent, 1.0 = fully opaque)
        arrow_length: Length of the image center axis arrow in pixels

    Returns:
        np.ndarray: Image with triangles drawn (H, W, 3) in BGR format
    """
    img = img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.ascontiguousarray(img * 255).astype(np.uint8)

    # Create an overlay image for the triangles
    overlay = img.copy()

    # Draw input view triangles (excluding the target view if it's also an input)
    for i, triangle in enumerate(fov_triangles):
        if input_mask[i] and i != target_view_idx:
            # Draw filled triangle for input views
            triangle_int = triangle.astype(np.int32)
            cv2.fillPoly(overlay, [triangle_int], input_color)
            # Draw triangle outline (fully opaque)
            cv2.polylines(img, [triangle_int], isClosed=True, color=input_color, thickness=2)

            # Draw image center axis arrow
            camera_center = triangle_int[0]  # First vertex is camera position
            direction = image_center_axis_directions[i]
            # Normalize direction and scale to arrow_length
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction_normalized = direction / direction_norm
                arrow_end = camera_center + (direction_normalized * arrow_length).astype(np.int32)
                cv2.arrowedLine(
                    img,
                    tuple(camera_center),
                    tuple(arrow_end),
                    input_color,
                    thickness=2,
                    tipLength=0.3,
                )

    # Draw target view triangle (on top, so it's visible even if it's also an input)
    if 0 <= target_view_idx < len(fov_triangles):
        triangle_int = fov_triangles[target_view_idx].astype(np.int32)
        cv2.fillPoly(overlay, [triangle_int], target_color)
        # Draw triangle outline (fully opaque)
        cv2.polylines(img, [triangle_int], isClosed=True, color=target_color, thickness=2)

        # Draw image center axis arrow for target view
        camera_center = triangle_int[0]  # First vertex is camera position
        direction = image_center_axis_directions[target_view_idx]
        # Normalize direction and scale to arrow_length
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction_normalized = direction / direction_norm
            arrow_end = camera_center + (direction_normalized * arrow_length).astype(np.int32)
            cv2.arrowedLine(
                img,
                tuple(camera_center),
                tuple(arrow_end),
                target_color,
                thickness=2,
                tipLength=0.3,
            )

    # Blend the overlay with the original image using alpha transparency
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def create_fov_triangle_visualizations(
    batch,
    device,
    fov_depth=30,
    camera_pos_depth=0.0,
    input_color=(255, 0, 0),
    target_color=(0, 255, 0),
    alpha=0.3,
):
    """
    Creates FOV triangle visualizations for all target views.

    Args:
        batch: Batch dictionary with camera poses, intrinsics, input_mask, target_mask, satellite_images
        device: Device to perform computations on
        fov_depth: Depth at which to compute the FOV triangle edges (in pixels)
        camera_pos_depth: Depth for camera position vertex (in pixels)
        input_color: BGR color for input view triangles (default: red)
        target_color: BGR color for target view triangle (default: green)
        alpha: Transparency factor for filled triangles (0.0 = fully transparent, 1.0 = fully opaque)

    Returns:
        torch.Tensor: Images with FOV triangles drawn, shape (num_target_views, C, H, W)
    """
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        fov_triangles, image_center_axis_directions = get_fov_triangle_points(
            batch, device, fov_depth=fov_depth, camera_pos_depth=camera_pos_depth
        )

    B, N = batch["images"].shape[:2]
    input_mask = batch.get("input_mask", torch.zeros(B, N, dtype=torch.bool, device=device))
    target_mask = batch.get("target_mask", torch.zeros(B, N, dtype=torch.bool, device=device))

    # Get satellite image (assuming batch size 1 for visualization)
    sat_img = batch["satellite_images"][0, 0]  # C, H, W

    # Convert triangles and image center axis directions to numpy
    fov_triangles_np = fov_triangles[0].cpu().numpy()  # N, 3, 2
    image_center_axis_directions_np = image_center_axis_directions[0].cpu().numpy()  # N, 2
    input_mask_np = input_mask[0].cpu().numpy()  # N

    # Find all target views
    target_indices = torch.where(target_mask[0])[0].cpu().numpy()

    if len(target_indices) == 0:
        # If no target mask, create one visualization per view
        target_indices = np.arange(N)

    images = []
    for target_idx in target_indices:
        img = visualize_fov_triangles(
            sat_img,
            fov_triangles_np,
            image_center_axis_directions_np,
            input_mask_np,
            target_view_idx=int(target_idx),
            input_color=input_color,
            target_color=target_color,
            alpha=alpha,
        )
        images.append(img)

    if len(images) == 0:
        # Fallback: create one image with all views
        img = visualize_fov_triangles(
            sat_img,
            fov_triangles_np,
            image_center_axis_directions_np,
            input_mask_np,
            target_view_idx=0,
            input_color=input_color,
            target_color=target_color,
            alpha=alpha,
        )
        images.append(img)

    images = np.stack(images, axis=0)  # num_target_views, H, W, 3
    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device) / 255.0 * 2 - 1
    return images

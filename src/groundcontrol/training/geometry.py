import torch

import groundcontrol.seva.geometry as seva_geometry

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],
    is_extrinsics_c2w: bool = False,
    input_is_batched=False,
):
    """Forked from seva.geometry.get_plucker_coordinates, with some modifications."""
    if intrinsics is None:
        intrinsics = seva_geometry.get_default_intrinsics(fov_rad).to(extrinsics.device)
    else:
        # TODO: reduced to -1 as sometimes these can be cropped.
        # if not (torch.all(intrinsics[:, :2, -1] >= -1) and torch.all(intrinsics[:, :2, -1] <= 1)):
        #     intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
        # you should ensure the intrisics are expressed in
        # resolution-independent normalized image coordinates just performing a
        # very simple verification here checking if principal points are
        # between 0 and 1
        if not (torch.all(intrinsics[:, :2, -1] >= -1) and torch.all(intrinsics[:, :2, -1] <= 1)):
            invalid_intrinsics = ((intrinsics[:, :2, -1] < -1) | (intrinsics[:, :2, -1] > 1)).any(
                dim=1
            )
            raise ValueError(
                f"Intrinsics should be expressed in resolution-independent normalized "
                f"image coordinates. {invalid_intrinsics.cpu().numpy()}"
            )

    if not is_extrinsics_c2w:
        c2w_src = torch.linalg.inv(extrinsics_src.float()).to(extrinsics_src.dtype)
        w2c_all = extrinsics
    else:
        c2w_src = extrinsics_src
        w2c_all = torch.linalg.inv(extrinsics.float()).to(extrinsics.dtype)
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera

    if not input_is_batched:
        all_c2w_src = c2w_src[None].repeat(w2c_all.shape[0], 1, 1)
    else:
        all_c2w_src = c2w_src

    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", w2c_all, all_c2w_src
    )  # cam_T_src = cam_T_w @ w_T_src

    unnormd_intrinsics = intrinsics.clone()
    unnormd_intrinsics[:, :2] *= w2c_all.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)
    centers, rays, grid_cam = seva_geometry.get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=unnormd_intrinsics,
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    return plucker

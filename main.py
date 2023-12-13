import argparse as ap
import subprocess
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R_scipy
from skimage.color import rgb2gray
from tqdm import tqdm

from cp_hw6 import computeExtrinsic, computeIntrinsic, pixel2ray


def load_image(image_path: Path):
    image = imageio.imread(image_path)
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=-1)
    gray = rgb2gray(image)

    return gray


def load_image_paths(src_dir: Path, skip=1):
    image_paths = sorted(list(src_dir.glob("*.jpg")))[::skip]
    return image_paths


def pick_planar_regions(image_path: np.ndarray, out_name="bounds.npy"):
    image = imageio.imread(image_path)
    plt.imshow(image)
    pts = plt.ginput(n=4, timeout=-1)
    tl_v, br_v, tl_h, br_h = pts
    plt.show()
    plt.close()
    bounds = np.array(
        [[tl_v[1], br_v[1], tl_v[0], br_v[0]], [tl_h[1], br_h[1], tl_h[0], br_h[0]]]
    ).astype(int)
    np.save(image_path.parent.parent / "out" / out_name, bounds)


def estimate_shadow_edges(image_paths, save_images=False):
    """Returned shadow edges N x 3 x 2, vertical a,b,c = column1, horizontal a,b,c = column2"""
    max_intensities = np.zeros_like(load_image(image_paths[0]))
    min_intensities = np.ones_like(load_image(image_paths[0])) * np.inf
    H, W = max_intensities.shape
    for image_path in tqdm(image_paths):
        image = load_image(image_path)
        max_intensities = np.maximum(image, max_intensities)
        min_intensities = np.minimum(image, min_intensities)
    np.savez(
        image_paths[0].parent.parent / "out" / "intensities.npz",
        max_intensities=max_intensities,
        min_intensities=min_intensities,
    )
    bounds = np.load(image_paths[0].parent.parent / "out" / "bounds.npy")
    vertical_bounds = bounds[0]
    horizontal_bounds = bounds[1]
    vertical_coords = np.meshgrid(
        np.arange(vertical_bounds[0], vertical_bounds[1]),
        np.arange(vertical_bounds[2], vertical_bounds[3]),
        indexing="ij",
    )
    print(vertical_bounds)
    print(horizontal_bounds)
    horizontal_coords = np.meshgrid(
        np.arange(horizontal_bounds[0], horizontal_bounds[1]),
        np.arange(horizontal_bounds[2], horizontal_bounds[3]),
        indexing="ij",
    )

    shadow_threshold_image = (max_intensities + min_intensities) / 2
    shadow_times = np.zeros((H, W))
    prev_sign_image = np.zeros_like(shadow_times)
    unchecked = np.ones_like(shadow_times)
    shadow_edges = np.zeros((len(image_paths), 3, 2))
    for t, image_path in tqdm(enumerate(image_paths)):
        image = load_image(image_path)
        difference_image = image - shadow_threshold_image
        abs_difference_image = np.abs(difference_image)
        sign_image = abs_difference_image - difference_image
        sign_image[np.where(sign_image)] = 1
        if t != 0:
            shadow_sign_diff_image = np.abs(sign_image - prev_sign_image)
            shadow_times = shadow_times + (unchecked * shadow_sign_diff_image * t)
            unchecked[np.where(shadow_times)] = 0

        prev_sign_image = sign_image
        diff_image = np.abs(np.diff(sign_image, axis=1))
        vert_zero_crossings = np.argwhere(
            diff_image[vertical_coords[0], vertical_coords[1]]
        )
        if len(vert_zero_crossings) == 0:
            continue
        vert_zero_crossings += np.array(
            [vertical_coords[0].min(), vertical_coords[1].min()]
        )
        vert_zero_crossings += 1
        _, indices, counts = np.unique(
            vert_zero_crossings[:, 0], axis=0, return_index=True, return_counts=True
        )
        if len(np.where(counts == 2)[0]) != 0:
            indices = indices[np.where(counts == 2)] + 1
        elif t <= len(image_paths) // 2:
            indices += 0
        else:
            indices = np.array([]).astype(int)

        vert_zero_crossings = vert_zero_crossings[indices]
        if len(vert_zero_crossings) == 0:
            continue

        horiz_zero_crossings = np.argwhere(
            diff_image[horizontal_coords[0], horizontal_coords[1]]
        )
        if len(horiz_zero_crossings) == 0:
            continue

        horiz_zero_crossings += np.array(
            [horizontal_coords[0].min(), horizontal_coords[1].min()]
        )
        horiz_zero_crossings += 1
        _, indices, counts = np.unique(
            horiz_zero_crossings[:, 0], axis=0, return_index=True, return_counts=True
        )

        if len(np.where(counts == 2)[0]) != 0:
            indices = indices[np.where(counts == 2)] + 1
        elif t <= len(image_paths) // 2:
            indices += 0
        else:
            indices = np.array([]).astype(int)
        horiz_zero_crossings = horiz_zero_crossings[indices]
        if len(horiz_zero_crossings) == 0:
            continue

        im = cv2.imread(str(image_path))
        im2 = im.copy()
        im2[
            vertical_bounds[0] : vertical_bounds[1] + 1,
            vertical_bounds[2] : vertical_bounds[3] + 1,
        ] = [0, 255, 0]
        im2[
            horizontal_bounds[0] : horizontal_bounds[1] + 1,
            horizontal_bounds[2] : horizontal_bounds[3] + 1,
        ] = [255, 0, 0]
        im = cv2.addWeighted(im, 0.8, im2, 0.2, 0.0)
        if len(vert_zero_crossings) >= 3:
            A = np.hstack(
                [
                    np.flip(vert_zero_crossings, axis=1),
                    np.ones((len(vert_zero_crossings), 1)),
                ]
            )
            U, S, V_t = np.linalg.svd(A)
            a, b, c = V_t.T[:, -1]  # ax + by + c = 0, y = -(a/b)x - (c/b)
            v_line = np.array([a, b, c])
            get_y = lambda x: int(-(a / b) * x - (c / b))
            get_x = lambda y: int(-(b / a) * y - (c / a))
            start_y = vertical_coords[0].min()
            start_x = get_x(start_y)
            end_y = vertical_coords[0].max()
            end_x = get_x(end_y)
            cv2.line(im, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
            shadow_edges[t, :, 0] = v_line

        if len(horiz_zero_crossings) >= 3:
            A = np.hstack(
                [
                    np.flip(horiz_zero_crossings, axis=1),
                    np.ones((len(horiz_zero_crossings), 1)),
                ]
            )
            U, S, V_t = np.linalg.svd(A)
            a, b, c = V_t.T[:, -1]  # ax + by + c = 0, y = -(a/b)x - (c/b)
            h_line = np.array([a, b, c])
            get_y = lambda x: int(-(a / b) * x - (c / b))
            get_x = lambda y: int(-(b / a) * y - (c / a))
            start_y = horizontal_coords[0].min()
            start_x = get_x(start_y)
            end_y = horizontal_coords[0].max()
            end_x = get_x(end_y)
            cv2.line(im, (start_x, start_y), (end_x, end_y), (255, 0, 0), 4)
            shadow_edges[t, :, 1] = h_line
        if (
            len(vert_zero_crossings) >= 3
            or len(horiz_zero_crossings) >= 3
            and save_images
        ):
            out_dir = Path(image_path).parent.parent / "out" / "shadow_edges"
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=False)
            out = out_dir / f"{image_path.stem}_shadow_edge{image_path.suffix}"
            if save_images:
                cv2.imwrite(str(out), im)

    # shadow_times
    if save_images:
        canvas = np.zeros((H, W, 3))
        unique_vals = np.unique(shadow_times)
        step_size = len(unique_vals) // 32
        colors = interpolate_colors([255, 0, 0], [0, 255, 255], [0, 0, 255], 38)
        color_counter = 0
        for i in tqdm(range(0, len(unique_vals), step_size)):
            start_val = min(i, len(unique_vals) - 1)
            end_val = min(i + step_size, len(unique_vals) - 1)
            canvas[
                np.where(
                    (shadow_times >= unique_vals[start_val])
                    & (shadow_times < unique_vals[end_val])
                )
            ] = colors[color_counter]
            color_counter += 1
        out = (
            Path(image_paths[0]).parent.parent
            / "out"
            / f"shadow_times{image_paths[0].suffix}"
        )

        cv2.imwrite(str(out), canvas.astype(np.uint8))
    return shadow_edges, shadow_times


def interpolate_colors(start_color, middle_color, end_color, N):
    def interpolate(start, end, n):
        return np.linspace(start, end, n)

    start_r, start_g, start_b = start_color
    middle_r, middle_g, middle_b = middle_color
    end_r, end_g, end_b = end_color

    first_half_r = interpolate(start_r, middle_r, N // 2)
    first_half_g = interpolate(start_g, middle_g, N // 2)
    first_half_b = interpolate(start_b, middle_b, N // 2)

    second_half_r = interpolate(middle_r, end_r, N // 2)
    second_half_g = interpolate(middle_g, end_g, N // 2)
    second_half_b = interpolate(middle_b, end_b, N // 2)

    interpolated_colors = np.stack(
        (
            np.concatenate((first_half_r, second_half_r)),
            np.concatenate((first_half_g, second_half_g)),
            np.concatenate((first_half_b, second_half_b)),
        ),
        axis=1,
    ).astype(np.uint8)

    return interpolated_colors


def get_intrinsics(calib_dir=None, load_from=None, skip=1):
    if load_from is not None:
        intrinsics = np.load(load_from)
        K, dist = intrinsics["K"], intrinsics["dist"]
        print(f"Loaded {intrinsics.files} from {load_from}")
    else:
        assert calib_dir is not None
        K, dist = computeIntrinsic(
            [str(x) for x in sorted(list(Path(calib_dir).glob("*.jpg")))][::skip],
            [6, 8],
            [8, 8],
        )
        print(K, dist)
        out = Path(calib_dir).parent / "out"
        if not out.exists():
            out.mkdir(exist_ok=False, parents=True)

        np.savez(out / "intrinsics.npz", K=K, dist=dist)
        print(f"Saved K, dist to {out}")

    return K, dist


def get_extrinsics(
    img_path=None, K=None, dist=None, dX=558.8, dY=303.2125, load_from=None
):
    if load_from is not None:
        extrinsics = np.load(load_from)
        t_v, R_v = extrinsics["t_v"], extrinsics["R_v"]
        t_h, R_h = extrinsics["t_h"], extrinsics["R_h"]
        print(f"Loaded {extrinsics.files} from {load_from}")
    else:
        t_v, R_v = computeExtrinsic(img_path, K, dist, dX, dY)
        t_h, R_h = computeExtrinsic(img_path, K, dist, dX, dY)
        out = Path(img_path).parent.parent / "out"
        np.savez(out / "extrinsics.npz", t_v=t_v, R_v=R_v, t_h=t_h, R_h=R_h)

    return t_v, R_v, t_h, R_h


def calibrateShadowLinesFrame(points, K, dist, Rs, ts):
    """
    points: 4 x 2, first 2 rows horizontal, next 2 rows vertical
    Rs: 3 x 3 x 2, first channel horizontal, second channel vertical
    ts: 1 x 3 x 2, first channel horizontal, second channel vertical
    """
    rays = pixel2ray(points.astype(np.float32), K, dist)
    cam_center = np.array([0, 0, 0]).reshape(-1, 1)
    r1_h = rays[0].T
    r2_h = rays[1].T
    r1_v = rays[2].T
    r2_v = rays[3].T

    Rh = Rs[..., 0]
    Rv = Rs[..., 1]

    th = ts[..., 0]
    tv = ts[..., 1]

    h_start = Rh.T @ (cam_center - th)
    v_start = Rv.T @ (cam_center - tv)
    normalize = lambda x: x / np.linalg.norm(x)
    r1_h = normalize((Rh.T @ (r1_h - th)) - h_start)
    r2_h = normalize((Rh.T @ (r2_h - th)) - h_start)
    r1_v = normalize((Rv.T @ (r1_v - tv)) - v_start)
    r2_v = normalize((Rv.T @ (r2_v - tv)) - v_start)

    t1_h = -h_start[-1] / r1_h[-1]
    P1_h = h_start + t1_h * r1_h

    t2_h = -h_start[-1] / r2_h[-1]
    P2_h = h_start + t2_h * r2_h

    t1_v = -v_start[-1] / r1_v[-1]
    P1_v = v_start + t1_v * r1_v

    t2_v = -v_start[-1] / r2_v[-1]
    P2_v = v_start + t2_v * r2_v

    return (
        P1_h,
        P2_h,
        P1_v,
        P2_v,
    )  


def calibrateShadowLines(image_paths, shadow_edges, K, dist, Rs, ts):
    points_3d = np.zeros((len(image_paths), 4, 3))
    bounds = np.load(image_paths[0].parent.parent / "out" / "bounds.npy")
    vertical_bounds = bounds[0]
    horizontal_bounds = bounds[1]
    vertical_coords = np.meshgrid(
        np.arange(vertical_bounds[0], vertical_bounds[1]),
        np.arange(vertical_bounds[2], vertical_bounds[3]),
        indexing="ij",
    )
    horizontal_coords = np.meshgrid(
        np.arange(horizontal_bounds[0], horizontal_bounds[1]),
        np.arange(horizontal_bounds[2], horizontal_bounds[3]),
        indexing="ij",
    )

    for i in tqdm(range(len(image_paths))):
        points = np.zeros((4, 2))
        if (shadow_edges[i] == 0).any(axis=0).any():
            print(f"FRAME {i}: No estimated shadow edges for both")
            continue
        else:
            print(f"FRAME {i}: Estimated shadow edges for both")
            a, b, c = shadow_edges[i][:, 1]

            get_x = lambda y: int(-(b / a) * y - (c / a))
            start_y = horizontal_coords[0].min()

            start_x = get_x(start_y)
            end_y = horizontal_coords[0].max()

            end_x = get_x(end_y)
            points[0] = np.array([start_x, start_y])
            points[1] = np.array([end_x, end_y])

            a, b, c = shadow_edges[i][:, 0]
            get_x = lambda y: int(-(b / a) * y - (c / a))
            start_y = vertical_coords[0].min()

            start_x = get_x(start_y)

            end_y = vertical_coords[0].max()

            end_x = get_x(end_y)
            points[2] = np.array([start_x, start_y])
            points[3] = np.array([end_x, end_y])

            P1, P2, P3, P4 = calibrateShadowLinesFrame(points, K, dist, Rs, ts)
            P1 = P1.reshape(1, -1)
            P2 = P2.reshape(1, -1)
            P3 = P3.reshape(1, -1)
            P4 = P4.reshape(1, -1)

            points_3d[i] = np.vstack([P1, P2, P3, P4])

    out = image_paths[0].parent.parent / "out"
    np.savez(out / "calibrated_shadow_line_points.npz", points=points_3d)
    return points_3d


def calibrateShadowPlane(points_3d, Rs, ts, out_dir):
    normals = np.zeros((len(points_3d), 3))
    P1s = np.zeros((len(points_3d), 3))

    Rh = Rs[..., 0]
    Rv = Rs[..., 1]

    th = ts[..., 0]
    tv = ts[..., 1]

    for i, points in tqdm(enumerate(points_3d)):
        if (points == 0).all(axis=1).all():
            print(f"Skipping {i}...")
        else:
            normalize = lambda x: x / np.linalg.norm(x)
            P1, P2, P3, P4 = points

            P1 = (Rh @ P1.reshape(-1, 1)) + th
            P1 = P1.reshape(1, -1)
            P2 = (Rh @ P2.reshape(-1, 1)) + th
            P2 = P2.reshape(1, -1)
            P3 = (Rv @ P3.reshape(-1, 1)) + tv
            P3 = P3.reshape(1, -1)
            P4 = (Rv @ P4.reshape(-1, 1)) + tv
            P4 = P4.reshape(1, -1)

            normal = normalize(np.cross((P2 - P1), (P4 - P3)))
            normals[i] = normal
            P1s[i] = P1

    out = out_dir / "calibrated_shadow_planes.npz"
    np.savez(out, normals=normals, points=P1s)


def eliminate_outliers(points, colors, threshold=0.05):
    mean = points.mean(axis=0)
    std_dev = points.std(axis=0)
    z_scores = np.abs((points - mean) / std_dev)
    outlier_indices = np.any(z_scores > 0.1, axis=1)
    return points[~outlier_indices], colors[~outlier_indices]


def reconstruct(image_paths, shadow_times, shadow_planes, K, dist):
    out = image_paths[0].parent.parent / "out"
    bounds = np.load(out / "bounds_reconstruct.npy")
    vertical_bounds = bounds[0]
    print(vertical_bounds)
    coords = np.meshgrid(
        np.arange(vertical_bounds[0], vertical_bounds[1]),
        np.arange(vertical_bounds[2], vertical_bounds[3]),
        indexing="ij",
    )

    xy_coords = np.hstack([coords[1].reshape(-1, 1), coords[0].reshape(-1, 1)]).astype(
        np.float32
    )
    reconstructed_points = []
    colors = []
    rays = pixel2ray(xy_coords, K, dist)
    intensities = np.load(out / "intensities.npz")
    max_intensities = intensities["max_intensities"]
    min_intensities = intensities["min_intensities"]
    images = [cv2.imread(str(im_path)) for im_path in image_paths]

    for i, (ray, coord) in tqdm(enumerate(zip(rays, xy_coords))):
        # if (
        #     max_intensities[int(coord[1]), int(coord[0])]
        #     - min_intensities[int(coord[1]), int(coord[0])]
        # ) <= 0.2 or (
        #     max_intensities[int(coord[1]), int(coord[0])]
        #     - min_intensities[int(coord[1]), int(coord[0])]
        # ) >= 0.5:
        #     continue
        # print(
        #     max_intensities[int(coord[1]), int(coord[0])]
        #     - min_intensities[int(coord[1]), int(coord[0])]
        # )
        ray = ray[0]
        frame_num = shadow_times[int(coord[1]), int(coord[0])]
        a, b, c, d = shadow_planes[int(frame_num)].reshape(-1, 1)
        if a == 0 and b == 0 and c == 0 and d == 0:
            continue
        t = d / (a * ray[0] + b * ray[1] + c * ray[2])
        if t > 500 or t < -500:
            continue
        reconstructed_points.append((ray * t).reshape(1, -1))
        colors.append(images[int(frame_num)][int(coord[1]), int(coord[0])][::-1])

    reconstructed_points = np.vstack(reconstructed_points)
    colors = np.vstack(colors)
    # reconstructed_points, colors = eliminate_outliers(reconstructed_points, colors)
    np.savez(
        out / "reconstructed_points.npz", points=reconstructed_points, colors=colors
    )


def plot_3d_points(reconstructed_points, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=96.0, azim=170)
    ax.scatter(
        reconstructed_points[..., 0],
        reconstructed_points[..., 1],
        reconstructed_points[..., 2],
        c=colors / 255.0,
        s=1,
    )

    plt.show()


def direct_indirect(exposure_stack, image_dir):
    exposure_paths = sorted(
        list(exposure_stack.glob("*.jpg")),
        key=lambda x: int(str(x).split("exposure")[-1].split(".")[0]),
    )

    exposure_stack = [
        cv2.imread(str(exposure_path)) for exposure_path in exposure_paths
    ]
    exposures = np.array(
        [(1 / 2048) * (2 ** (k)) for k in range(len(exposure_stack))]
    ).astype(np.float32)
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(exposure_stack, exposures)
    response = np.squeeze(response)

    image_paths = sorted(list(image_dir.glob("*.jpg")))
    maxes = np.zeros_like(cv2.imread(str(image_paths[0])))
    max_r, max_g, max_b = maxes[..., 0], maxes[..., 1], maxes[..., 2]

    mins = np.ones_like(cv2.imread(str(image_paths[0]))) * np.inf
    min_r, min_g, min_b = mins[..., 0], mins[..., 1], mins[..., 2]

    for im_path in tqdm(image_paths):
        im = cv2.imread(str(im_path))
        im_r, im_g, im_b = im[..., 0], im[..., 1], im[..., 2]
        linearized_im_r = response[:, 0][im_r.reshape(-1, 1)].reshape(im_r.shape)
        linearized_im_g = response[:, 1][im_g.reshape(-1, 1)].reshape(im_g.shape)
        linearized_im_b = response[:, 2][im_b.reshape(-1, 1)].reshape(im_b.shape)

        max_r = np.maximum(max_r, linearized_im_r)
        max_g = np.maximum(max_g, linearized_im_g)
        max_b = np.maximum(max_b, linearized_im_b)

        min_r = np.minimum(min_r, linearized_im_r)
        min_g = np.minimum(min_g, linearized_im_g)
        min_b = np.minimum(min_b, linearized_im_b)

    direct = np.dstack([max_r, max_g, max_b]) - np.dstack([min_r, min_g, min_b]).astype(
        np.float32
    )
    indirect = 2 * np.dstack([min_r, min_g, min_b]).astype(np.float32)
    tonemap = cv2.createTonemap(2.2)
    direct_tonemapped = tonemap.process(direct)
    indirect_tonemapped = tonemap.process(indirect)

    cv2.imwrite(
        "./mydata/direct-indirect2/direct.jpg",
        (np.clip(direct, 0.0, 1.0) * 255).astype(np.uint8),
    )
    cv2.imwrite(
        "./mydata/direct-indirect2/indirect.jpg",
        (np.clip(indirect, 0.0, 1.0) * 255).astype(np.uint8),
    )

    H, W, C = exposure_stack[0].shape
    collage_pre_linearization = np.zeros((4 * H, 4 * W, 3))
    collage_post_linearization = np.zeros((4 * H, 4 * W, 3)).astype(np.uint8)
    tonemap = cv2.createTonemap(2.2)
    for i, im in tqdm(enumerate(exposure_stack)):
        row = i // 4
        col = i % 4
        collage_pre_linearization[
            (row) * H : (row + 1) * H,
            (col) * W : (col + 1) * W,
        ] = im
        im_r, im_g, im_b = im[..., 0], im[..., 1], im[..., 2]
        linearized_im_r = response[:, 0][im_r.reshape(-1, 1)].reshape(im_r.shape)
        linearized_im_g = response[:, 1][im_g.reshape(-1, 1)].reshape(im_g.shape)
        linearized_im_b = response[:, 2][im_b.reshape(-1, 1)].reshape(im_b.shape)

        linearized_im = np.dstack([linearized_im_r, linearized_im_g, linearized_im_b])
        linearized_im_tonemapped = tonemap.process(linearized_im)
        linearized_im_tonemapped = (
            (np.clip(linearized_im_tonemapped, 0.0, 1.0) * 255.0)
        ).astype(np.uint8)
        collage_post_linearization[
            (row) * H : (row + 1) * H,
            (col) * W : (col + 1) * W,
        ] = linearized_im_tonemapped

    cv2.imwrite("./mydata/direct-indirect2/collage_pre.jpg", collage_pre_linearization)
    cv2.imwrite(
        "./mydata/direct-indirect2/collage_post.jpg",
        collage_post_linearization.astype(np.uint8),
    )
    plt.plot(np.arange(256), response[:, 0], c="r", label="Red")
    plt.plot(np.arange(256), response[:, 1], c="g", label="Green")
    plt.plot(np.arange(256), response[:, 2], c="b", label="Blue")
    plt.title("Inverse Camera Response Function")
    plt.legend(loc="upper left")
    plt.savefig("./mydata/direct-indirect2/CRF.jpg")


def capture_exposure_stack(num_exposures):
    for i in range(1, num_exposures + 1):
        print(f"Capturing Image {i}...")
        cmd = [
            "sudo",
            "gphoto2",
            "--set-config-value",
            f"/main/capturesettings/shutterspeed={2**(i - 1)}/2048",
        ]
        subprocess.run(cmd)
        cmd = [
            "sudo",
            "gphoto2",
            "--capture-image-and-download",
            "--filename",
            f"./mydata/direct-indirect/exposure_stack/exposure{i}.%C",
        ]
        subprocess.run(cmd)


def pipeline(src_dir):
    src_dir = Path(src_dir)
    image_paths = load_image_paths(src_dir, skip=5)
    out = Path(image_paths[0]).parent.parent / "out"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
    pick_planar_regions(image_paths[0], out_name="bounds.npy")
    shadow_edges, shadow_times = estimate_shadow_edges(image_paths, save_images=True)
    K, dist = get_intrinsics(
        calib_dir=Path(image_paths[0]).parent.parent / "calib", skip=1
    )
    K, dist = get_intrinsics(load_from=out / "intrinsics.npz")
    t_v, R_v, t_h, R_h = get_extrinsics(
        str(image_paths[0]), K, dist, dX=11 * 2, dY=8.5 * 2
    )
    t_v, R_v, t_h, R_h = get_extrinsics(load_from=out / "extrinsics.npz")

    Rs = np.dstack([R_h, R_v])
    ts = np.dstack([t_h, t_v])
    points_3d = calibrateShadowLines(image_paths, shadow_edges, K, dist, Rs, ts)
    points_3d = np.load(out / "calibrated_shadow_line_points.npz")["points"]
    calibrateShadowPlane(points_3d, Rs, ts, out)
    shadow_planes = np.load(out / "calibrated_shadow_planes.npz")
    normals = shadow_planes["normals"]
    P1s = shadow_planes["points"]
    plane_constants = (-P1s.reshape(-1, 3) * normals.reshape(-1, 3)).sum(axis=1)
    planes = np.hstack([normals, plane_constants.reshape(-1, 1)])
    pick_planar_regions(image_paths[0], out_name="bounds_reconstruct.npy")
    reconstruct(image_paths, shadow_times, planes, K, dist)
    reconstructed_points = np.load(out / "reconstructed_points.npz")
    points = reconstructed_points["points"]
    colors = reconstructed_points["colors"]
    plot_3d_points(points, colors)


def main(args):
    pipeline(args.src_dir)
    # capture_exposure_stack(16)
    # direct_indirect(
    #     Path("./mydata/direct-indirect2/exposure_stack"),
    #     Path("./mydata/direct-indirect2/frames"),
    # )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--save_images", action=ap.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)

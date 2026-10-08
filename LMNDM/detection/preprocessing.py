"""SimpleITK-only preprocessing for detector and classifier crops."""

from __future__ import annotations

from itertools import product
import math
import warnings

import numpy as np
import SimpleITK as sitk


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def read_image(image, pixel_type=sitk.sitkFloat32):
    if isinstance(image, sitk.Image):
        return sitk.Cast(image, pixel_type)
    return sitk.ReadImage(str(image), outputPixelType=pixel_type)


def image_world_bounds(image: sitk.Image):
    """Return the physical bounds of an entire image."""
    size = np.asarray(image.GetSize(), dtype=np.float64)
    corners = [
        image.TransformContinuousIndexToPhysicalPoint(index)
        for index in product(
            (0.0, float(size[0])),
            (0.0, float(size[1])),
            (0.0, float(size[2])),
        )
    ]
    corners = np.asarray(corners, dtype=np.float64)
    return corners.min(axis=0), corners.max(axis=0)


def automatic_lung_mask(
    image: sitk.Image,
    *,
    threshold_hu: float = -320.0,
    target_spacing_mm: float = 2.0,
    minimum_volume_mm3: float = 10_000.0,
) -> sitk.Image | None:
    """Estimate a coarse lung mask from CT without an external segmenter.

    The method removes air connected to the volume corners and retains the
    largest internal low-density components. It is intended to constrain the
    detector search region, not as a clinical lung segmentation.
    """
    image = read_image(image)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    shrink = [
        max(int(round(float(target_spacing_mm) / float(axis_spacing))), 1)
        for axis_spacing in spacing
    ]
    coarse = sitk.Shrink(image, shrink)
    air = sitk.Cast(coarse < float(threshold_hu), sitk.sitkUInt8)
    component_filter = sitk.ConnectedComponentImageFilter()
    component_filter.SetFullyConnected(True)
    components = component_filter.Execute(air)
    labels = sitk.GetArrayViewFromImage(components)
    corner_labels = {
        int(labels[z, y, x])
        for z in (0, labels.shape[0] - 1)
        for y in (0, labels.shape[1] - 1)
        for x in (0, labels.shape[2] - 1)
    }

    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(components)
    candidates = [
        label
        for label in statistics.GetLabels()
        if label not in corner_labels
        and statistics.GetPhysicalSize(label) >= float(minimum_volume_mm3)
    ]
    candidates.sort(key=statistics.GetPhysicalSize, reverse=True)
    candidates = candidates[:3]
    if not candidates:
        warnings.warn(
            "automatic lung-region estimation failed; falling back to the "
            "full CT volume",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    mask = sitk.Image(components.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(components)
    for label in candidates:
        mask = mask | sitk.Cast(components == int(label), sitk.sitkUInt8)
    radius = [
        max(int(round(3.0 / float(axis_spacing))), 1)
        for axis_spacing in coarse.GetSpacing()
    ]
    return sitk.BinaryMorphologicalClosing(mask, radius, sitk.sitkBall)


def mask_world_bounds(mask: sitk.Image):
    array = sitk.GetArrayViewFromImage(mask)
    coordinates = np.argwhere(array > 0)
    if coordinates.size == 0:
        raise ValueError("lung mask is empty")

    min_zyx = coordinates.min(axis=0)
    max_zyx = coordinates.max(axis=0)
    min_xyz = min_zyx[::-1].astype(np.float64)
    # Use the outer edge of the final mask voxel, matching the historical
    # ``world_box_full`` behavior.
    max_xyz = max_zyx[::-1].astype(np.float64) + 1.0

    corners = []
    for index in product(
        (float(min_xyz[0]), float(max_xyz[0])),
        (float(min_xyz[1]), float(max_xyz[1])),
        (float(min_xyz[2]), float(max_xyz[2])),
    ):
        corners.append(mask.TransformContinuousIndexToPhysicalPoint(index))
    corners = np.asarray(corners, dtype=np.float64)
    return corners.min(axis=0), corners.max(axis=0)


def plan_patches(
    world_min,
    world_max,
    spacing,
    margin_mm,
    patch_size,
):
    world_min = np.asarray(world_min, dtype=np.float64)
    world_max = np.asarray(world_max, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    margin = np.asarray(margin_mm, dtype=np.float64)
    patch_size = np.asarray(patch_size, dtype=np.int32)

    patch_world_size = patch_size * spacing
    step_size = patch_world_size - 2.0 * margin
    if np.any(step_size <= 0):
        raise ValueError("patch size is too small for the configured overlap margin")

    extent = np.maximum(world_max - world_min, 0.0)
    patch_counts = np.maximum(np.ceil(extent / step_size).astype(np.int32), 1)
    allowed_max = world_max + margin

    total = int(np.prod(patch_counts))
    patch_index = 0
    for z_index in range(int(patch_counts[2])):
        for y_index in range(int(patch_counts[1])):
            for x_index in range(int(patch_counts[0])):
                grid_index = np.array((x_index, y_index, z_index), dtype=np.float64)
                origin = world_min + step_size * grid_index - margin
                end = origin + patch_world_size
                shift = np.maximum(end - allowed_max, 0.0)
                origin -= shift
                patch_index += 1
                yield origin, patch_size.copy(), patch_index, total


def resample_patch(
    image: sitk.Image,
    origin,
    spacing,
    size,
    *,
    linear: bool = True,
    default_value: float = -600.0,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(value) for value in size])
    resampler.SetOutputOrigin(tuple(float(value) for value in origin))
    resampler.SetOutputSpacing(tuple(float(value) for value in spacing))
    resampler.SetOutputDirection(IDENTITY_DIRECTION)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear if linear else sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)


def centered_patch(
    image: sitk.Image,
    center,
    spacing,
    size,
    *,
    default_value: float = -600.0,
) -> sitk.Image:
    center = np.asarray(center, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    size = np.asarray(size, dtype=np.int32)
    origin = center - spacing * size.astype(np.float64) / 2.0
    return resample_patch(
        image,
        origin,
        spacing,
        size,
        linear=True,
        default_value=default_value,
    )


def normalized_array(
    image: sitk.Image,
    mean: float,
    stddev: float,
    clip: bool = True,
) -> np.ndarray:
    if stddev <= 0:
        raise ValueError("normalizer standard deviation must be positive")
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    array = (array - float(mean)) / float(stddev)
    if clip:
        np.clip(array, -1.0, 1.0, out=array)
    return np.ascontiguousarray(array)


def normalized_array_from_config(
    image: sitk.Image,
    normalizer,
) -> np.ndarray:
    normalizer_type = int(normalizer["type"])
    if normalizer_type == 0:
        return normalized_array(
            image,
            normalizer["mean"],
            normalizer["stddev"],
            bool(normalizer.get("clip", True)),
        )

    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if normalizer_type == 1:
        low, high = np.quantile(
            array,
            (float(normalizer["min_p"]), float(normalizer["max_p"])),
        )
        stddev = float(high - low) / 2.0
        if stddev <= 0:
            raise ValueError("adaptive normalizer has a zero intensity range")
        array = (array - float(low + high) / 2.0) / stddev
        if bool(normalizer.get("clip", True)):
            np.clip(array, -1.0, 1.0, out=array)
    elif normalizer_type == 2:
        np.clip(
            array,
            float(normalizer["p_005"]),
            float(normalizer["p_995"]),
            out=array,
        )
        array = (
            array - float(normalizer["mean"])
        ) / float(normalizer["stddev"])
    else:
        raise ValueError(f"unknown normalizer type: {normalizer_type}")
    return np.ascontiguousarray(array)


def dilate_mask(mask: sitk.Image, radius_mm: float) -> sitk.Image:
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    radius = [
        max(int(math.ceil(float(radius_mm) / float(axis_spacing))), 0)
        for axis_spacing in spacing
    ]
    return sitk.BinaryDilate(binary, radius, sitk.sitkBall)


def points_inside_mask(points, mask: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayViewFromImage(mask)
    size = np.asarray(mask.GetSize(), dtype=np.int64)
    result = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        try:
            voxel = np.asarray(
                mask.TransformPhysicalPointToIndex(tuple(float(value) for value in point)),
                dtype=np.int64,
            )
        except RuntimeError:
            continue
        if np.any(voxel < 0) or np.any(voxel >= size):
            continue
        result[index] = array[int(voxel[2]), int(voxel[1]), int(voxel[0])] > 0
    return result

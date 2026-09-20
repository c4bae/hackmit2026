"""Synchronous GPU reconstruction service used by the FastAPI job worker."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
import pickle
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from scene_postprocessing import (
    _world_to_gltf,
    build_room_shell,
    colorize_object_meshes,
)
from shaper_runtime import ShapeRRuntime
from video_pipeline import (
    _as_numpy,
    _to_homogeneous,
    build_object_pkls,
    extract_video_frames,
    infer_da3_geometry,
    refine_tracks_by_world_geometry,
    segment_video_sam3,
)
from video_to_3d import _combine_meshes


class CancelledError(RuntimeError):
    pass


class JobReporter(Protocol):
    id: str
    root: Path
    cancelled: bool

    def emit(
        self,
        event_type: str,
        *,
        stage: str,
        progress: float,
        title: str,
        detail: str = "",
        artifact: dict | None = None,
        payload: dict | None = None,
    ) -> None: ...


_RUNTIME: ShapeRRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def _get_runtime() -> ShapeRRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = ShapeRRuntime()
    return _RUNTIME


def _asset_url(job: JobReporter, path: Path) -> str:
    relative = path.relative_to(job.root).as_posix()
    return f"/api/jobs/{job.id}/files/{relative}"


def _check_cancelled(job: JobReporter) -> None:
    if job.cancelled:
        raise CancelledError("Job cancelled")


def create_lightweight_mesh_preview(
    source_path: Path,
    output_path: Path,
    *,
    max_faces: int = 20_000,
) -> Path:
    """Create a colored low-poly GLB for responsive browser previews."""
    try:
        mesh = trimesh.load(source_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) <= max_faces:
            return source_path

        source_vertices = np.asarray(mesh.vertices)
        source_colors = np.asarray(mesh.visual.vertex_colors)
        preview = mesh.simplify_quadric_decimation(face_count=max_faces)

        if (
            source_colors.ndim == 2
            and len(source_colors) == len(source_vertices)
            and len(preview.vertices)
        ):
            nearest = cKDTree(source_vertices).query(
                np.asarray(preview.vertices),
                k=1,
            )[1]
            preview.visual.vertex_colors = source_colors[nearest]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview.export(output_path)
        return output_path
    except Exception:
        return source_path


def _contact_sheet(frames: list[np.ndarray], output_path: Path, columns: int = 4) -> None:
    selected = frames[:12]
    thumb_width, thumb_height = 360, 202
    rows = int(np.ceil(len(selected) / columns))
    canvas = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (8, 12, 18))
    for index, frame in enumerate(selected):
        image = Image.fromarray(frame).resize((thumb_width, thumb_height))
        canvas.paste(image, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def _segmentation_sheet(
    frames: list[np.ndarray],
    tracks,
    output_path: Path,
    columns: int = 4,
) -> None:
    palette = np.array(
        [
            [61, 214, 198],
            [255, 111, 97],
            [255, 190, 92],
            [128, 145, 255],
            [214, 112, 255],
            [112, 220, 112],
            [255, 130, 190],
            [104, 185, 255],
        ],
        dtype=np.uint8,
    )
    selected_indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
    thumb_width, thumb_height = 400, 225
    rows = int(np.ceil(len(selected_indices) / columns))
    canvas = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (8, 12, 18))
    draw = ImageDraw.Draw(canvas)
    for cell, frame_index in enumerate(selected_indices):
        frame = frames[int(frame_index)].copy()
        for track_index, track in enumerate(tracks):
            mask = track.masks[int(frame_index)]
            if mask is None:
                continue
            color = palette[track_index % len(palette)]
            frame[mask] = (frame[mask] * 0.48 + color * 0.52).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color.tolist(), 2)
        thumb = np.asarray(Image.fromarray(frame).resize((thumb_width, thumb_height)))
        x = (cell % columns) * thumb_width
        y = (cell // columns) * thumb_height
        canvas.paste(Image.fromarray(thumb), (x, y))
        labels = ", ".join(sorted({track.label for track in tracks if track.masks[int(frame_index)] is not None}))
        draw.rounded_rectangle((x + 10, y + 10, x + min(360, 18 + len(labels) * 7), y + 38), 8, fill=(5, 9, 15))
        draw.text((x + 18, y + 17), labels or "no persistent object", fill=(235, 245, 245))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-7:
        return None
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=6)
    rotation = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction / length)
    transform = np.asarray(rotation if rotation is not None else np.eye(4))
    transform[:3, 3] = (start + end) * 0.5
    mesh.apply_transform(transform)
    mesh.visual.face_colors = np.asarray(color, dtype=np.uint8)
    return mesh


def _geometry_preview_glb(
    frames: list[np.ndarray],
    prediction,
    output_path: Path,
) -> None:
    depths = _as_numpy(prediction.depth)
    intrinsics = _as_numpy(prediction.intrinsics)
    extrinsics = _to_homogeneous(prediction.extrinsics)
    confidences = (
        _as_numpy(prediction.conf)
        if getattr(prediction, "conf", None) is not None
        else None
    )
    height, width = depths.shape[-2:]
    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    stride = 4
    for index, depth in enumerate(depths):
        frame = cv2.resize(frames[index], (width, height), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:height:stride, 0:width:stride]
        sampled_depth = depth[::stride, ::stride]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
        if confidences is not None and np.any(valid):
            sampled_confidence = confidences[index][::stride, ::stride]
            valid &= sampled_confidence >= np.percentile(sampled_confidence[valid], 25)
        x = xx[valid]
        y = yy[valid]
        z = sampled_depth[valid]
        pixels = np.stack([x, y, np.ones_like(x)], axis=0)
        camera_points = (np.linalg.inv(intrinsics[index]) @ pixels) * z[None]
        camera_h = np.vstack([camera_points, np.ones((1, len(z)))])
        world = (np.linalg.inv(extrinsics[index]) @ camera_h)[:3].T
        points_all.append(world)
        colors_all.append(frame[::stride, ::stride][valid])

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if len(points) > 180_000:
        keep = np.random.default_rng(23).choice(len(points), 180_000, replace=False)
        points, colors = points[keep], colors[keep]
    world_to_gltf = _world_to_gltf(extrinsics)
    points_h = np.column_stack([points, np.ones(len(points))])
    gltf_points = (world_to_gltf @ points_h.T).T[:, :3]
    rgba = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])

    preview = trimesh.Scene()
    preview.add_geometry(
        trimesh.points.PointCloud(gltf_points, colors=rgba),
        node_name="colored_point_cloud",
        geom_name="colored_point_cloud",
    )
    center = np.median(gltf_points, axis=0)
    extent = np.percentile(np.linalg.norm(gltf_points - center, axis=1), 85)
    plane_depth = float(np.clip(extent * 0.055, 0.07, 0.22))
    line_radius = max(plane_depth * 0.012, 0.0012)
    frustum_color = (62, 146, 255, 255)

    for index, (frame, intrinsic, extrinsic) in enumerate(
        zip(frames, intrinsics, extrinsics)
    ):
        fx = max(float(intrinsic[0, 0]), 1e-6)
        fy = max(float(intrinsic[1, 1]), 1e-6)
        half_width = plane_depth * width / fx * 0.5
        half_height = plane_depth * height / fy * 0.5
        camera_vertices = np.array(
            [
                [-half_width, -half_height, plane_depth],
                [half_width, -half_height, plane_depth],
                [half_width, half_height, plane_depth],
                [-half_width, half_height, plane_depth],
            ],
            dtype=np.float64,
        )
        camera_to_gltf = world_to_gltf @ np.linalg.inv(extrinsic)
        plane_h = np.column_stack([camera_vertices, np.ones(4)])
        plane_vertices = (camera_to_gltf @ plane_h.T).T[:, :3]
        texture = Image.fromarray(frame).resize((320, 180))
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture,
            roughnessFactor=1.0,
            metallicFactor=0.0,
            doubleSided=True,
        )
        plane = trimesh.Trimesh(
            vertices=plane_vertices,
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
            visual=trimesh.visual.texture.TextureVisuals(
                uv=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float),
                material=material,
            ),
        )
        preview.add_geometry(plane, node_name=f"frame_{index:02d}")
        origin = camera_to_gltf[:3, 3]
        segments = [(origin, corner) for corner in plane_vertices]
        segments.extend(
            (plane_vertices[edge], plane_vertices[(edge + 1) % 4])
            for edge in range(4)
        )
        frustum_parts = [
            part
            for start, end in segments
            if (part := _cylinder_between(start, end, line_radius, frustum_color))
            is not None
        ]
        if frustum_parts:
            preview.add_geometry(
                trimesh.util.concatenate(frustum_parts),
                node_name=f"camera_{index:02d}",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.export(output_path)


def run_reconstruction(
    job: JobReporter,
    *,
    video_path: Path,
    prompts: list[str],
    max_frames: int,
    max_objects: int,
    preset: str,
) -> dict:
    work_dir = job.root / "work"
    output_dir = job.root / "output"
    preview_dir = job.root / "previews"
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    job.emit(
        "stage",
        stage="frames",
        progress=0.04,
        title="Sampling the capture",
        detail="Selecting stable, evenly spaced views from the uploaded video.",
    )
    frames, frame_paths, metadata = extract_video_frames(
        video_path, work_dir / "frames", max_frames=max_frames
    )
    _check_cancelled(job)
    frames_preview = preview_dir / "frames.jpg"
    _contact_sheet(frames, frames_preview)
    job.emit(
        "artifact",
        stage="frames",
        progress=0.10,
        title=f"{len(frames)} views selected",
        detail="These views will anchor segmentation, camera recovery and color projection.",
        artifact={"kind": "image", "url": _asset_url(job, frames_preview)},
    )

    job.emit(
        "stage",
        stage="segmentation",
        progress=0.13,
        title="Finding persistent objects",
        detail="SAM3 is detecting and tracking common indoor objects across the clip.",
    )
    tracks = segment_video_sam3(
        frames,
        prompts,
        max_objects=max_objects,
    )
    if not tracks:
        raise RuntimeError("SAM3 did not find persistent objects. Try a slower camera pass with clearer views.")
    _check_cancelled(job)
    segmentation_preview = preview_dir / "segmentation.jpg"
    _segmentation_sheet(frames, tracks, segmentation_preview)
    labels = [track.label for track in tracks]
    job.emit(
        "artifact",
        stage="segmentation",
        progress=0.27,
        title=f"{len(tracks)} object tracks found",
        detail=", ".join(labels),
        artifact={"kind": "image", "url": _asset_url(job, segmentation_preview)},
        payload={"labels": labels},
    )

    job.emit(
        "stage",
        stage="geometry",
        progress=0.31,
        title="Recovering the room",
        detail="Depth Anything 3 is estimating depth, intrinsics and camera motion.",
    )
    prediction = infer_da3_geometry(frame_paths)
    sam_track_count = len(tracks)
    tracks = refine_tracks_by_world_geometry(
        tracks, prediction, max_objects=max_objects
    )
    if not tracks:
        raise RuntimeError("No spatially consistent object tracks survived 3D refinement.")
    _check_cancelled(job)
    confidence = getattr(prediction, "conf", None)
    geometry_path = work_dir / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        depths=_as_numpy(prediction.depth).astype(np.float32),
        confidences=(
            _as_numpy(confidence).astype(np.float32)
            if confidence is not None
            else np.empty(0, dtype=np.float32)
        ),
        intrinsics=_as_numpy(prediction.intrinsics).astype(np.float32),
        extrinsics=_as_numpy(prediction.extrinsics).astype(np.float32),
    )
    geometry_preview = preview_dir / "geometry_preview.glb"
    _geometry_preview_glb(frames, prediction, geometry_preview)
    job.emit(
        "artifact",
        stage="geometry",
        progress=0.39,
        title="3D geometry and camera poses ready",
        detail=(
            f"Colored point cloud with {len(frames)} posed video frames. "
            f"3D consistency produced {len(tracks)} separate object instances "
            f"from {sam_track_count} SAM tracks."
        ),
        artifact={"kind": "model", "url": _asset_url(job, geometry_preview)},
        payload={
            "samTrackCount": sam_track_count,
            "refinedInstanceCount": len(tracks),
        },
    )

    sample_paths, object_metadata = build_object_pkls(
        frames, tracks, prediction, work_dir / "objects"
    )
    del prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not sample_paths:
        raise RuntimeError("No tracked object produced enough consistent geometry.")
    job.emit(
        "stage",
        stage="shaper",
        progress=0.43,
        title="Loading ShapeR",
        detail=f"Preparing generative reconstruction for {len(sample_paths)} objects.",
    )
    runtime = _get_runtime()
    _check_cancelled(job)

    mesh_paths = []
    color_statistics = []
    label_totals = Counter(item["label"] for item in object_metadata)
    label_seen: defaultdict[str, int] = defaultdict(int)
    for item in object_metadata:
        label = item["label"]
        label_seen[label] += 1
        item["display_label"] = (
            f"{label}{label_seen[label]:02d}" if label_totals[label] > 1 else label
        )
    metadata_by_name = {item["name"]: item for item in object_metadata}
    total = len(sample_paths)
    for index, sample_path in enumerate(sample_paths):
        _check_cancelled(job)
        mesh_path = runtime.reconstruct_many(
            [sample_path], object_dir, preset=preset
        )[0]
        statistics = colorize_object_meshes(
            [sample_path], [mesh_path], work_dir
        )[0]
        mesh_paths.append(mesh_path)
        color_statistics.append(statistics)
        name = mesh_path.stem
        preview_path = create_lightweight_mesh_preview(
            mesh_path,
            preview_dir / "objects" / mesh_path.name,
        )
        item = metadata_by_name[name]
        progress = 0.46 + 0.37 * ((index + 1) / total)
        job.emit(
            "object",
            stage="shaper",
            progress=progress,
            title=f"{item['display_label']} reconstructed",
            detail=f"Object {index + 1} of {total} · {item['visible_views']} supporting views",
            artifact={
                "kind": "model",
                "url": _asset_url(job, preview_path),
                "name": name,
                "label": item["display_label"],
            },
            payload={
                "name": name,
                "label": item["display_label"],
                "visibleViews": item["visible_views"],
                "directColorFraction": statistics["direct_color_fraction"],
            },
        )

    job.emit(
        "stage",
        stage="room",
        progress=0.86,
        title="Fusing walls and floor",
        detail="Background RGB-D is being fused into a colored room shell.",
    )
    room_path, room_nodes = build_room_shell(
        sample_paths, work_dir, output_dir / "room_shell.glb"
    )
    _check_cancelled(job)
    job.emit(
        "artifact",
        stage="room",
        progress=0.92,
        title="Room shell complete",
        detail="Floor, walls and remaining background are separate editable nodes.",
        artifact={"kind": "model", "url": _asset_url(job, room_path)},
        payload={"nodes": room_nodes},
    )

    scene_path = output_dir / "scene.glb"
    _combine_meshes([*mesh_paths, room_path], scene_path)
    manifest = {
        **metadata,
        "prompts": prompts,
        "preset": preset,
        "sam_track_count": sam_track_count,
        "refined_instance_count": len(tracks),
        "objects": object_metadata,
        "object_colors": color_statistics,
        "room_nodes": room_nodes,
        "coordinate_system": "glTF Y-up",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    objects = []
    for item, mesh_path, statistics in zip(
        object_metadata, mesh_paths, color_statistics
    ):
        objects.append(
            {
                "name": item["name"],
                "label": item["display_label"],
                "visibleViews": item["visible_views"],
                "boundsMeters": item["bounds_m"],
                "directColorFraction": statistics["direct_color_fraction"],
                "url": _asset_url(job, mesh_path),
            }
        )
    result = {
        "jobId": job.id,
        "sceneUrl": _asset_url(job, scene_path),
        "roomUrl": _asset_url(job, room_path),
        "manifestUrl": _asset_url(job, manifest_path),
        "objects": objects,
        "roomNodes": room_nodes,
        "coordinateSystem": "glTF Y-up",
    }
    job.emit(
        "complete",
        stage="complete",
        progress=1.0,
        title="Interactive scene ready",
        detail=f"{len(objects)} objects plus the reconstructed room are ready to explore.",
        artifact={"kind": "model", "url": result["sceneUrl"]},
        payload=result,
    )
    return result

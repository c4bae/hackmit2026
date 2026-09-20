"""Ephemeral FastAPI wrapper for the ShapeR video reconstruction pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import shutil
import ssl
import time
import uuid
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
import certifi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.pipeline_service import CancelledError, run_reconstruction


JOB_ROOT = Path(os.environ.get("SHAPER_JOB_ROOT", "/tmp/shaper-video-jobs"))
JOB_STATE_FILENAME = "job_state.json"
JOB_TTL_SECONDS = int(os.environ.get("SHAPER_JOB_TTL_SECONDS", "604800"))
MAX_UPLOAD_BYTES = int(os.environ.get("SHAPER_MAX_UPLOAD_BYTES", str(1024**3)))
PAIRING_TTL_SECONDS = int(os.environ.get("HONKPACK_PAIRING_TTL_SECONDS", "1200"))
DEFAULT_SCENE_CONCEPTS = (
    "bed", "bench", "book", "box", "cabinet", "chair", "computer",
    "desk", "dresser", "keyboard", "lamp", "monitor", "nightstand",
    "pillow", "plant", "refrigerator", "rug", "shelf", "sink", "sofa",
    "stool", "table", "television", "toilet", "trash can", "wardrobe",
)
GPU_GATE = asyncio.Lock()
TASKS: set[asyncio.Task] = set()


@dataclass
class ReconstructionJob:
    id: str
    root: Path
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    title: str = "Waiting for the GPU"
    detail: str = ""
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    cancelled: bool = False
    delete_when_done: bool = False
    input_sha256: str | None = None
    cache_key: str | None = None
    parameters: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def _persist_unlocked(self) -> None:
        state = {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "title": self.title,
            "detail": self.detail,
            "events": self.events,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
            "input_sha256": self.input_sha256,
            "cache_key": self.cache_key,
            "parameters": self.parameters,
        }
        temporary = self.root / f".{JOB_STATE_FILENAME}.tmp"
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(self.root / JOB_STATE_FILENAME)

    def set_result(self, result: dict) -> None:
        with self.lock:
            self.result = result
            self.updated_at = time.time()
            self._persist_unlocked()

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
    ) -> None:
        with self.lock:
            self.updated_at = time.time()
            self.stage = stage
            self.progress = float(progress)
            self.title = title
            self.detail = detail
            if event_type == "complete":
                self.status = "complete"
            elif event_type == "error":
                self.status = "error"
            elif event_type == "cancelled":
                self.status = "cancelled"
            elif self.status == "queued":
                self.status = "processing"
            event = {
                "id": len(self.events) + 1,
                "type": event_type,
                "stage": stage,
                "progress": self.progress,
                "title": title,
                "detail": detail,
                "timestamp": self.updated_at,
            }
            if artifact is not None:
                event["artifact"] = artifact
            if payload is not None:
                event["payload"] = payload
            self.events.append(event)
            self._persist_unlocked()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "jobId": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "title": self.title,
                "detail": self.detail,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "error": self.error,
                "result": self.result,
            }


@dataclass
class MobilePairing:
    id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + PAIRING_TTL_SECONDS)
    status: str = "waiting"
    events: list[dict] = field(default_factory=list)
    job_id: str | None = None
    lock: Lock = field(default_factory=Lock)

    def emit(
        self,
        event_type: str,
        *,
        status: str,
        title: str,
        detail: str = "",
        payload: dict | None = None,
        expected: set[str] | None = None,
    ) -> bool:
        with self.lock:
            if expected is not None and self.status not in expected:
                return False
            self.updated_at = time.time()
            self.status = status
            if payload and payload.get("jobId"):
                self.job_id = str(payload["jobId"])
            event = {
                "id": len(self.events) + 1,
                "type": event_type,
                "status": status,
                "title": title,
                "detail": detail,
                "timestamp": self.updated_at,
            }
            if payload is not None:
                event["payload"] = payload
            self.events.append(event)
            return True

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "pairId": self.id,
                "status": self.status,
                "jobId": self.job_id,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "expiresAt": self.expires_at,
                "expiresInSeconds": max(0, int(self.expires_at - time.time())),
            }


JOBS: dict[str, ReconstructionJob] = {}
PAIRINGS: dict[str, MobilePairing] = {}
CACHE_VERSION = 1


def _make_cache_key(input_sha256: str, parameters: dict) -> str:
    payload = json.dumps(
        {"version": CACHE_VERSION, "input_sha256": input_sha256, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _find_reusable_job(cache_key: str) -> ReconstructionJob | None:
    matches = [
        job for job in JOBS.values()
        if job.cache_key == cache_key
        and job.status in {"queued", "processing", "complete"}
        and not job.cancelled
        and not job.delete_when_done
    ]
    if not matches:
        return None
    return max(matches, key=lambda job: (job.status == "complete", job.updated_at))


def _job_submission(job: ReconstructionJob, *, cache_hit: bool) -> dict:
    return {
        "jobId": job.id,
        "status": job.status,
        "eventsUrl": f"/api/jobs/{job.id}/events",
        "statusUrl": f"/api/jobs/{job.id}",
        "expiresInSeconds": max(0, int(JOB_TTL_SECONDS - (time.time() - job.updated_at))),
        "cacheHit": cache_hit,
    }


def _restore_jobs() -> None:
    for root in JOB_ROOT.iterdir():
        state_path = root / JOB_STATE_FILENAME
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            job = ReconstructionJob(
                id=str(state.get("id") or root.name),
                root=root,
                created_at=float(state.get("created_at", root.stat().st_mtime)),
                updated_at=float(state.get("updated_at", root.stat().st_mtime)),
                status=str(state.get("status", "error")),
                stage=str(state.get("stage", "error")),
                progress=float(state.get("progress", 0.0)),
                title=str(state.get("title", "Recovered reconstruction")),
                detail=str(state.get("detail", "")),
                events=list(state.get("events") or []),
                result=state.get("result"),
                error=state.get("error"),
                cancelled=bool(state.get("cancelled", False)),
                input_sha256=state.get("input_sha256"),
                cache_key=state.get("cache_key"),
                parameters=dict(state.get("parameters") or {}),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if job.status in {"queued", "processing"}:
            job.status = "error"
            job.stage = "error"
            job.error = "Backend restarted before reconstruction completed"
            job.title = "Reconstruction interrupted"
            job.detail = job.error
            job.emit(
                "error",
                stage="error",
                progress=job.progress,
                title=job.title,
                detail=job.detail,
            )
        JOBS[job.id] = job


def _get_job(job_id: str) -> ReconstructionJob:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or already expired")
    return job


def _get_pairing(pair_id: str) -> MobilePairing:
    pairing = PAIRINGS.get(pair_id)
    if pairing is None:
        raise HTTPException(status_code=404, detail="Phone pairing not found")
    if pairing.status not in {"submitted", "expired"} and time.time() >= pairing.expires_at:
        pairing.emit(
            "expired",
            status="expired",
            title="QR code expired",
            detail="Create a new phone connection from the desktop.",
        )
    if pairing.status == "expired":
        raise HTTPException(status_code=410, detail="Phone pairing expired")
    return pairing


def _remove_job(job: ReconstructionJob) -> None:
    JOBS.pop(job.id, None)
    shutil.rmtree(job.root, ignore_errors=True)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - JOB_TTL_SECONDS
        for job in list(JOBS.values()):
            if job.status in {"complete", "error", "cancelled"} and job.updated_at < cutoff:
                _remove_job(job)
        now = time.time()
        for pair_id, pairing in list(PAIRINGS.items()):
            if pairing.status not in {"submitted", "expired"} and now >= pairing.expires_at:
                pairing.emit(
                    "expired",
                    status="expired",
                    title="QR code expired",
                    detail="Create a new phone connection from the desktop.",
                )
            if now - pairing.updated_at > PAIRING_TTL_SECONDS:
                PAIRINGS.pop(pair_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    _restore_jobs()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(
    title="ShapeR Scene Reconstruction API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        value.strip()
        for value in os.environ.get(
            "SHAPER_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,https://honkpack.vercel.app",
        ).split(",")
        if value.strip()
    ],
    allow_origin_regex=os.environ.get(
        "SHAPER_CORS_ORIGIN_REGEX",
        r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _execute_job(
    job: ReconstructionJob,
    video_path: Path,
    prompts: list[str],
    max_frames: int,
    max_objects: int,
    preset: str,
) -> None:
    try:
        job.emit(
            "queued",
            stage="queued",
            progress=0.01,
            title="Queued for reconstruction",
            detail="This demo runs one GPU reconstruction at a time.",
        )
        async with GPU_GATE:
            if job.cancelled:
                raise CancelledError("Job cancelled before processing")
            result = await asyncio.to_thread(
                run_reconstruction,
                job,
                video_path=video_path,
                prompts=prompts,
                max_frames=max_frames,
                max_objects=max_objects,
                preset=preset,
            )
            job.set_result(result)
    except CancelledError:
        job.emit(
            "cancelled",
            stage="cancelled",
            progress=job.progress,
            title="Reconstruction cancelled",
            detail="Temporary files will be removed.",
        )
    except Exception as error:
        job.error = str(error)
        job.emit(
            "error",
            stage="error",
            progress=job.progress,
            title="Reconstruction stopped",
            detail=str(error),
        )
    finally:
        if job.delete_when_done:
            _remove_job(job)


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "gpuQueueBusy": GPU_GATE.locked(),
        "activeJobs": sum(
            job.status in {"queued", "processing"} for job in JOBS.values()
        ),
        "persistence": "disk",
        "contentCache": "sha256+parameters",
        "reusableJobs": sum(job.status == "complete" and bool(job.cache_key) for job in JOBS.values()),
        "waitingPhonePairings": sum(
            pairing.status in {"waiting", "uploading"} for pairing in PAIRINGS.values()
        ),
    }


@app.post("/api/pairings", status_code=201)
async def create_pairing() -> dict:
    pair_id = uuid.uuid4().hex[:24]
    pairing = MobilePairing(id=pair_id)
    PAIRINGS[pair_id] = pairing
    pairing.emit(
        "waiting",
        status="waiting",
        title="Waiting for your phone",
        detail="Scan the QR code and choose or record a video.",
    )
    return {
        **pairing.snapshot(),
        "eventsUrl": f"/api/pairings/{pair_id}/events",
        "uploadUrl": f"/api/pairings/{pair_id}/jobs",
    }


@app.get("/api/pairings/{pair_id}")
async def get_pairing(pair_id: str) -> dict:
    return _get_pairing(pair_id).snapshot()


@app.post("/api/pairings/{pair_id}/jobs", status_code=202)
async def create_paired_job(
    pair_id: str,
    video: UploadFile = File(...),
    prompts: str = Form(""),
    max_frames: int = Form(16),
    max_objects: int = Form(8),
    preset: str = Form("balance"),
) -> dict:
    pairing = _get_pairing(pair_id)
    started = pairing.emit(
        "uploading",
        status="uploading",
        title="Phone upload started",
        detail="Keep this page open until the upload finishes.",
        expected={"waiting"},
    )
    if not started:
        raise HTTPException(status_code=409, detail="This phone pairing is already in use")
    try:
        job = await create_job(
            video=video,
            prompts=prompts,
            max_frames=max_frames,
            max_objects=max_objects,
            preset=preset,
        )
    except Exception:
        pairing.emit(
            "waiting",
            status="waiting",
            title="Upload did not finish",
            detail="Choose the video again and retry from this phone.",
            expected={"uploading"},
        )
        raise
    pairing.emit(
        "submitted",
        status="submitted",
        title="Video received",
        detail="Reconstruction is continuing on the desktop.",
        payload=job,
        expected={"uploading"},
    )
    return {**job, "pairId": pair_id}


@app.get("/api/pairings/{pair_id}/events")
async def stream_pairing_events(pair_id: str, request: Request):
    pairing = PAIRINGS.get(pair_id)
    if pairing is None:
        raise HTTPException(status_code=404, detail="Phone pairing not found")
    last_event_id = int(request.headers.get("last-event-id", "0") or "0")

    async def event_stream():
        cursor = last_event_id
        while True:
            if await request.is_disconnected():
                return
            if pairing.status not in {"submitted", "expired"} and time.time() >= pairing.expires_at:
                pairing.emit(
                    "expired",
                    status="expired",
                    title="QR code expired",
                    detail="Create a new phone connection from the desktop.",
                )
            with pairing.lock:
                pending = [event for event in pairing.events if event["id"] > cursor]
                terminal = pairing.status in {"submitted", "expired"}
            for event in pending:
                cursor = event["id"]
                yield (
                    f"id: {cursor}\n"
                    f'event: {event["type"]}\n'
                    f"data: {json.dumps(event)}\n\n"
                )
            if terminal and not pending:
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/api/pairings/{pair_id}/events/ws")
async def stream_pairing_events_ws(websocket: WebSocket, pair_id: str) -> None:
    pairing = PAIRINGS.get(pair_id)
    if pairing is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    cursor = 0
    try:
        while True:
            if pairing.status not in {"submitted", "expired"} and time.time() >= pairing.expires_at:
                pairing.emit(
                    "expired",
                    status="expired",
                    title="QR code expired",
                    detail="Create a new phone connection from the desktop.",
                )
            with pairing.lock:
                pending = [event for event in pairing.events if event["id"] > cursor]
                terminal = pairing.status in {"submitted", "expired"}
            for event in pending:
                cursor = event["id"]
                await websocket.send_json(event)
            if terminal and not pending:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    prompts: str = Form(""),
    max_frames: int = Form(16),
    max_objects: int = Form(8),
    preset: str = Form("balance"),
) -> dict:
    if preset not in {"speed", "balance", "quality"}:
        raise HTTPException(status_code=422, detail="Invalid ShapeR preset")
    if not 2 <= max_frames <= 32:
        raise HTTPException(status_code=422, detail="max_frames must be between 2 and 32")
    if not 1 <= max_objects <= 16:
        raise HTTPException(status_code=422, detail="max_objects must be between 1 and 16")
    suffix = Path(video.filename or "capture.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, M4V or WebM video")

    job_id = uuid.uuid4().hex[:12]
    root = JOB_ROOT / job_id
    root.mkdir(parents=True, exist_ok=False)
    input_path = root / f"capture{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        with input_path.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds upload limit")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        await video.close()

    concepts = [item.strip() for item in prompts.split(",") if item.strip()]
    if not concepts:
        concepts = list(DEFAULT_SCENE_CONCEPTS)
    normalized_concepts = sorted(set(concepts))
    parameters = {
        "concepts": normalized_concepts,
        "max_frames": max_frames,
        "max_objects": max_objects,
        "preset": preset,
    }
    input_sha256 = digest.hexdigest()
    cache_key = _make_cache_key(input_sha256, parameters)
    reusable = _find_reusable_job(cache_key)
    if reusable is not None:
        shutil.rmtree(root, ignore_errors=True)
        reusable.updated_at = time.time()
        with reusable.lock:
            reusable._persist_unlocked()
        return _job_submission(reusable, cache_hit=True)

    job = ReconstructionJob(
        id=job_id,
        root=root,
        input_sha256=input_sha256,
        cache_key=cache_key,
        parameters=parameters,
    )
    JOBS[job_id] = job
    with job.lock:
        job._persist_unlocked()
    task = asyncio.create_task(
        _execute_job(job, input_path, normalized_concepts, max_frames, max_objects, preset)
    )
    TASKS.add(task)
    task.add_done_callback(TASKS.discard)
    return _job_submission(job, cache_hit=False)


@app.get("/api/maps/autocomplete")
async def autocomplete_map_place(q: str = ""):
    query = q.strip()
    if len(query) < 2 or len(query) > 200:
        return {"suggestions": []}
    api_key = (
        os.environ.get("GOOGLE_MAPS_PLACES_API_KEY", "").strip()
        or os.environ.get("GOOGLE_MAPS_BROWSER_API_KEY", "").strip()
        or os.environ.get("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY", "").strip()
    )
    if not api_key:
        raise HTTPException(status_code=503, detail="Google Places API key is not configured")

    def call_google_places() -> dict:
        payload = json.dumps({"input": query, "includeQueryPredictions": False}).encode("utf-8")
        google_request = urllib.request.Request(
            "https://places.googleapis.com/v1/places:autocomplete",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(google_request, timeout=10, context=ssl.create_default_context(cafile=certifi.where())) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(detail[:500]) from error

    try:
        result = await asyncio.to_thread(call_google_places)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Google place search failed: {error}") from error
    suggestions = []
    for suggestion in result.get("suggestions") or []:
        prediction = suggestion.get("placePrediction") or {}
        text = ((prediction.get("text") or {}).get("text") or "").strip()
        if text:
            suggestions.append({"text": text, "placeId": prediction.get("placeId")})
        if len(suggestions) >= 6:
            break
    return {"suggestions": suggestions}


@app.post("/api/maps/route")
async def compute_map_route(request: Request):
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Google Maps API key is not configured")
    body = await request.json()
    origin = str(body.get("origin", "")).strip()
    destination = str(body.get("destination", "")).strip()
    if not origin or not destination or len(origin) > 300 or len(destination) > 300:
        raise HTTPException(status_code=422, detail="Valid origin and destination are required")

    def call_google() -> dict:
        payload = json.dumps({
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "DRIVE",
            "units": "METRIC",
        }).encode("utf-8")
        google_request = urllib.request.Request(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline",
            },
        )
        try:
            with urllib.request.urlopen(google_request, timeout=15, context=ssl.create_default_context(cafile=certifi.where())) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(detail[:500]) from error

    try:
        result = await asyncio.to_thread(call_google)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Google route request failed: {error}") from error
    routes = result.get("routes") or []
    if not routes:
        raise HTTPException(status_code=404, detail="No driving route found")
    route = routes[0]
    duration = str(route.get("duration") or "0s")
    try:
        duration_seconds = float(duration.removesuffix("s"))
    except ValueError:
        duration_seconds = 0.0
    return {
        "distanceKm": round(float(route.get("distanceMeters", 0)) / 1000, 2),
        "duration": duration,
        "durationSeconds": round(duration_seconds),
        "encodedPolyline": (route.get("polyline") or {}).get("encodedPolyline"),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return _get_job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = _get_job(job_id)
    if job.status == "error":
        return JSONResponse(status_code=500, content=job.snapshot())
    if job.status != "complete" or job.result is None:
        return JSONResponse(status_code=202, content=job.snapshot())
    return job.result


@app.get("/api/jobs/{job_id}/events/history")
async def get_event_history(job_id: str, after: int = 0) -> dict:
    job = _get_job(job_id)
    with job.lock:
        events = [event for event in job.events if event["id"] > after]
    return {"events": events}


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str, request: Request):
    job = _get_job(job_id)
    last_event_id = int(request.headers.get("last-event-id", "0") or "0")

    async def event_stream():
        cursor = last_event_id
        while True:
            if await request.is_disconnected():
                return
            with job.lock:
                pending = [event for event in job.events if event["id"] > cursor]
                terminal = job.status in {"complete", "error", "cancelled"}
            for event in pending:
                cursor = event["id"]
                yield (
                    f"id: {cursor}\n"
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event)}\n\n"
                )
            if terminal and not pending:
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/api/jobs/{job_id}/events/ws")
async def stream_job_events_ws(websocket: WebSocket, job_id: str) -> None:
    job = JOBS.get(job_id)
    if job is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    cursor = 0
    try:
        while True:
            with job.lock:
                pending = [event for event in job.events if event["id"] > cursor]
                terminal = job.status in {"complete", "error", "cancelled"}
            for event in pending:
                cursor = event["id"]
                await websocket.send_json(event)
            if terminal and not pending:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


@app.get("/api/jobs/{job_id}/files/{asset_path:path}")
async def get_file(job_id: str, asset_path: str):
    job = _get_job(job_id)
    root = job.root.resolve()
    path = (root / asset_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    job = _get_job(job_id)
    job.cancelled = True
    if job.status in {"complete", "error", "cancelled"}:
        _remove_job(job)
        return {"deleted": True}
    job.delete_when_done = True
    return {"deleted": False, "status": "cancelling"}

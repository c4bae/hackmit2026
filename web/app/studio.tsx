"use client";

import {
  Box,
  Check,
  ChevronRight,
  CloudUpload,
  Cuboid,
  FileVideo,
  Grip,
  Layers3,
  LoaderCircle,
  Move3d,
  Play,
  Rotate3d,
  ScanSearch,
  Sparkles,
  PanelRightClose,
  RotateCcw,
  Truck as TruckIcon,
  Video,
  X,
} from "lucide-react";
import {
  Bounds,
  Environment,
  Html,
  Lightformer,
  OrbitControls,
  TransformControls,
  useGLTF,
} from "@react-three/drei";
import { Canvas, type ThreeEvent, useFrame } from "@react-three/fiber";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { MobileCapturePage, PhonePairingButton } from "./mobile-flow";
import { estimatePackingItemWeight, PackingItemThumbnail, TruckFitting, type PackingItem } from "./truck-fitting";

const API_BASE =
  process.env.NEXT_PUBLIC_SHAPER_API_URL?.replace(/\/$/, "") ||
  "/api/shaper";

function websocketUrl(path: string) {
  const base = new URL(API_BASE, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}${path}`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

type Artifact = {
  kind: "image" | "model";
  url: string;
  name?: string;
  label?: string;
};

type PipelineEvent = {
  id: number;
  type: string;
  stage: string;
  progress: number;
  title: string;
  detail: string;
  artifact?: Artifact;
  payload?: Record<string, unknown>;
};

type ObjectResult = {
  name: string;
  label: string;
  visibleViews: number;
  boundsMeters: number[];
  directColorFraction: number;
  url: string;
};

type JobResult = {
  jobId: string;
  sceneUrl: string;
  roomUrl: string;
  manifestUrl: string;
  objects: ObjectResult[];
  roomNodes: string[];
  coordinateSystem: string;
};

type TransformMode = "translate" | "rotate";
type TransformScope = "object" | "scene";
type ResultView = "scene" | "truck";

const ITEM_COLORS = ["#f39a43", "#79d9b5", "#6d9f91", "#f1bd63", "#8eb6a9", "#d98461", "#87a8d0", "#b79ed8"];

const STAGES = [
  { id: "frames", label: "Frame sampling", icon: Video },
  { id: "segmentation", label: "SAM3 segmentation", icon: ScanSearch },
  { id: "geometry", label: "Depth and point cloud", icon: Grip },
  { id: "shaper", label: "Object reconstruction", icon: Cuboid },
  { id: "room", label: "Walls and floor", icon: Layers3 },
  { id: "complete", label: "Scene ready", icon: Sparkles },
];

function absoluteUrl(path?: string) {
  if (!path) return "";
  return /^https?:\/\//.test(path) ? path : `${API_BASE}${path}`;
}

function stageIndex(stage: string) {
  const index = STAGES.findIndex((item) => item.id === stage);
  return index < 0 ? 0 : index;
}

function prettyName(name: string) {
  if (name.startsWith("room_")) {
    return name
      .replace("room_floor", "Floor")
      .replace("room_walls", "Walls")
      .replace("room_other", "Room shell");
  }
  return name
    .replace(/^object_\d+_/, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (value) => value.toUpperCase());
}

function materialsFor(root: THREE.Object3D) {
  const materials: THREE.Material[] = [];
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const source = Array.isArray(child.material)
      ? child.material
      : [child.material];
    source.forEach((material) => materials.push(material));
  });
  return materials;
}

function setHighlight(root: THREE.Object3D | null, active: boolean) {
  if (!root) return;
  materialsFor(root).forEach((material) => {
    const standard = material as THREE.MeshStandardMaterial;
    if (!standard.userData.baseEmissive && "emissive" in standard) {
      standard.userData.baseEmissive = standard.emissive.clone();
      standard.userData.baseEmissiveIntensity = standard.emissiveIntensity;
    }
    if ("emissive" in standard) {
      if (active) {
        standard.emissive.set("#ff9f6e");
        standard.emissiveIntensity = 0.62;
      } else if (standard.userData.baseEmissive) {
        standard.emissive.copy(standard.userData.baseEmissive);
        standard.emissiveIntensity =
          standard.userData.baseEmissiveIntensity ?? 1;
      }
      material.needsUpdate = true;
    }
  });
}

function ModelLoader() {
  return (
    <Html center>
      <div className="model-loader">
        <LoaderCircle size={18} />
        <span>Loading scene</span>
      </div>
    </Html>
  );
}

function InteractiveModel({
  url,
  mode,
  scope,
  requestedSelection,
  onSelect,
  onHover,
  onTransforming,
  resetSignal,
}: {
  url: string;
  mode: TransformMode;
  scope: TransformScope;
  requestedSelection: string | null;
  onSelect: (name: string | null) => void;
  onHover: (name: string | null) => void;
  onTransforming: (active: boolean) => void;
  resetSignal: number;
}) {
  const { scene } = useGLTF(url);
  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => material.clone())
        : child.material.clone();
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      materials.forEach((material) => {
        const standard = material as THREE.MeshStandardMaterial;
        if (!standard.isMeshStandardMaterial) return;
        standard.metalness = 0;
        standard.roughness = 0.86;
        standard.envMapIntensity = 0.72;
        standard.needsUpdate = true;
      });
    });
    clone.updateMatrixWorld(true);
    [...clone.children].forEach((root) => {
      const bounds = new THREE.Box3().setFromObject(root);
      if (bounds.isEmpty()) return;
      const center = bounds.getCenter(new THREE.Vector3());
      const pivot = new THREE.Group();
      pivot.name = root.name;
      pivot.userData.isObjectPivot = true;
      clone.add(pivot);
      pivot.position.copy(center);
      pivot.updateMatrixWorld(true);
      pivot.attach(root);
      root.name = `${pivot.name}__geometry`;
    });
    clone.updateMatrixWorld(true);
    return clone;
  }, [scene]);
  const [selected, setSelected] = useState<THREE.Object3D | null>(null);
  const [hovered, setHovered] = useState<THREE.Object3D | null>(null);
  const initial = useRef(
    new Map<
      string,
      {
        position: THREE.Vector3;
        quaternion: THREE.Quaternion;
        scale: THREE.Vector3;
      }
    >(),
  );

  const resolveRoot = useCallback(
    (source: THREE.Object3D) => {
      let current = source;
      while (current.parent && current.parent !== model) {
        current = current.parent;
      }
      return current;
    },
    [model],
  );

  useEffect(() => {
    initial.current.clear();
    model.position.set(0, 0, 0);
    model.quaternion.identity();
    model.scale.set(1, 1, 1);
    model.children.forEach((child) => {
      initial.current.set(child.name, {
        position: child.position.clone(),
        quaternion: child.quaternion.clone(),
        scale: child.scale.clone(),
      });
    });
    setSelected(null);
    onSelect(null);
  }, [model, onSelect]);

  useEffect(() => {
    setHighlight(hovered, true);
    return () => setHighlight(hovered, false);
  }, [hovered]);

  useEffect(() => {
    if (!requestedSelection) return;
    const target = model.children.find(
      (child) => child.name === requestedSelection,
    );
    if (target) {
      setSelected(target);
      onSelect(target.name);
    }
  }, [requestedSelection, model, onSelect]);

  useEffect(() => {
    model.position.set(0, 0, 0);
    model.quaternion.identity();
    model.scale.set(1, 1, 1);
    model.children.forEach((child) => {
      const transform = initial.current.get(child.name);
      if (!transform) return;
      child.position.copy(transform.position);
      child.quaternion.copy(transform.quaternion);
      child.scale.copy(transform.scale);
      child.updateMatrix();
    });
  }, [resetSignal, model]);

  const handleOver = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const root = resolveRoot(event.object);
    setHovered(root);
    onHover(root.name);
    document.body.style.cursor = "pointer";
  };

  const handleOut = () => {
    setHovered(null);
    onHover(null);
    document.body.style.cursor = "default";
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    const root = resolveRoot(event.object);
    setSelected(root);
    onSelect(root.name);
  };

  const transformTarget = scope === "scene" ? model : selected;

  return (
    <>
      <Bounds fit clip margin={1.28}>
        <primitive
          object={model}
          onPointerOver={handleOver}
          onPointerOut={handleOut}
          onClick={handleClick}
        />
      </Bounds>
      {transformTarget && (
        <TransformControls
          object={transformTarget}
          mode={mode}
          size={0.72}
          onMouseDown={() => onTransforming(true)}
          onMouseUp={() => onTransforming(false)}
        />
      )}
    </>
  );
}

function brightPreviewClone(source: THREE.Object3D) {
  const clone = source.clone(true);
  clone.traverse((child) => {
    if (child instanceof THREE.Points) {
      const material = (child.material as THREE.PointsMaterial).clone();
      material.size = 0.012;
      material.sizeAttenuation = true;
      material.vertexColors = true;
      child.material = material;
      return;
    }
    if (!(child instanceof THREE.Mesh)) return;
    const vertexColors = Boolean(child.geometry.getAttribute("color"));
    const sourceMaterials = Array.isArray(child.material) ? child.material : [child.material];
    const converted = sourceMaterials.map((material) => {
      const sourceMaterial = material as THREE.MeshStandardMaterial;
      return new THREE.MeshBasicMaterial({
        color: sourceMaterial.color?.clone() ?? new THREE.Color("#ffffff"),
        map: sourceMaterial.map ?? null,
        vertexColors,
        transparent: sourceMaterial.transparent,
        opacity: sourceMaterial.opacity,
        side: THREE.DoubleSide,
        toneMapped: false,
      });
    });
    child.material = Array.isArray(child.material) ? converted : converted[0];
  });
  return clone;
}

function PreviewModel({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const model = useMemo(() => brightPreviewClone(scene), [scene]);
  return (
    <Bounds fit clip observe margin={1.35}>
      <primitive object={model} />
    </Bounds>
  );
}

function ProcessingModelPreview({ url }: { url: string }) {
  return (
    <div className="scene-viewport processing-model-preview">
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [3.8, 2.8, 4.8], fov: 42, near: 0.01, far: 100 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.45;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <color attach="background" args={["#f7faf9"]} />
        <ambientLight intensity={2.1} />
        <hemisphereLight args={["#ffffff", "#dce8e3", 1.5]} />
        <directionalLight position={[5, 8, 3]} intensity={2.1} color="#ffffff" />
        <Suspense fallback={<ModelLoader />}>
          <PreviewModel url={url} />
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.075} minDistance={0.5} maxDistance={12} />
      </Canvas>
    </div>
  );
}

function RotatingObjectModel({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const group = useRef<THREE.Group>(null);
  const model = useMemo(() => {
    const clone = brightPreviewClone(scene);
    clone.updateMatrixWorld(true);
    const center = new THREE.Box3().setFromObject(clone).getCenter(new THREE.Vector3());
    clone.position.sub(center);
    return clone;
  }, [scene]);
  useFrame((_, delta) => {
    if (group.current) group.current.rotation.y += delta * 0.9;
  });
  return (
    <Bounds fit clip observe margin={1.3}>
      <group ref={group}><primitive object={model} /></group>
    </Bounds>
  );
}

function ObjectMeshGrid({ items }: { items: PackingItem[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  return (
    <div className="object-mesh-grid">
      {items.map((item) => (
        <article key={item.id} className="object-mesh-card" onMouseEnter={() => setActiveId(item.id)} onMouseLeave={() => setActiveId((current) => current === item.id ? null : current)}>
          <div className="object-mesh-preview">
            {activeId === item.id ? (
              <Canvas camera={{ position: [2.4, 1.7, 2.8], fov: 38 }} dpr={[1, 1.4]}>
                <color attach="background" args={["#f2f7f5"]} />
                <ambientLight intensity={2.2} />
                <hemisphereLight args={["#ffffff", "#e0e9e5", 1.4]} />
                <Suspense fallback={null}><RotatingObjectModel url={item.url} /></Suspense>
              </Canvas>
            ) : (
              <span className="object-mesh-static"><PackingItemThumbnail item={item} /></span>
            )}
          </div>
          <strong>{item.name}</strong>
        </article>
      ))}
    </div>
  );
}

function SceneViewport({
  url,
  selectedName,
  onSelect,
  onHover,
}: {
  url: string;
  selectedName: string | null;
  onSelect: (name: string | null) => void;
  onHover: (name: string | null) => void;
}) {
  const [mode, setMode] = useState<TransformMode>("translate");
  const [scope, setScope] = useState<TransformScope>("object");
  const [transforming, setTransforming] = useState(false);
  const [resetSignal, setResetSignal] = useState(0);

  return (
    <div className="scene-viewport">
      <div className="viewport-toolbar" role="toolbar" aria-label="Scene tools">
        <div className="tool-group">
          <button
            className={mode === "translate" ? "active" : ""}
            onClick={() => setMode("translate")}
            title="Move"
          >
            <Move3d size={16} />
            Move
          </button>
          <button
            className={mode === "rotate" ? "active" : ""}
            onClick={() => setMode("rotate")}
            title="Rotate"
          >
            <Rotate3d size={16} />
            Rotate
          </button>
        </div>
        <div className="tool-group scope-switch">
          <button
            className={scope === "object" ? "active" : ""}
            onClick={() => setScope("object")}
          >
            Object
          </button>
          <button
            className={scope === "scene" ? "active" : ""}
            onClick={() => setScope("scene")}
          >
            Scene
          </button>
        </div>
        <button
          className="reset-button"
          onClick={() => setResetSignal((value) => value + 1)}
          title={`Reset ${scope} transforms`}
        >
          <RotateCcw size={15} />
          Reset
        </button>
      </div>

      <Canvas
        dpr={[1, 1.75]}
        camera={{ position: [4.6, 3.2, 5.8], fov: 42, near: 0.01, far: 100 }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.22;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
        onPointerMissed={() => onSelect(null)}
      >
        <color attach="background" args={["#f7faf9"]} />
        <fog attach="fog" args={["#f7faf9", 12, 30]} />
        <ambientLight intensity={0.38} />
        <hemisphereLight args={["#e8f3ff", "#6c7483", 0.86]} />
        <directionalLight
          position={[5, 8, 3]}
          intensity={2.55}
          color="#fff2dc"
        />
        <directionalLight position={[-4, 3, -5]} intensity={0.92} color="#8fc5ff" />
        <Environment resolution={192}>
          <Lightformer form="rect" intensity={3.2} color="#ffffff" position={[0, 6, -5]} scale={[8, 4, 1]} />
          <Lightformer form="rect" intensity={2} color="#9bcaff" position={[-6, 2, 1]} rotation={[0, Math.PI / 2, 0]} scale={[5, 5, 1]} />
          <Lightformer form="ring" intensity={1.2} color="#dbeafe" position={[5, 1, 4]} scale={2.5} />
        </Environment>
        <gridHelper
          args={[20, 40, "#cbdcd6", "#e8f0ed"]}
          position={[0, -1.72, 0]}
        />
        <Suspense fallback={<ModelLoader />}>
          <InteractiveModel
            url={url}
            mode={mode}
            scope={scope}
            requestedSelection={selectedName}
            onSelect={onSelect}
            onHover={onHover}
            onTransforming={setTransforming}
            resetSignal={resetSignal}
          />
        </Suspense>
        <OrbitControls
          makeDefault
          enabled={!transforming}
          enableDamping
          dampingFactor={0.075}
          minDistance={1}
          maxDistance={18}
        />
      </Canvas>

    </div>
  );
}

function UploadPanel({
  onStarted,
}: {
  onStarted: (jobId: string, eventsUrl: string, cacheHit?: boolean) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState("balance");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptFile = (candidate?: File) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("video/")) {
      setError("Choose an MP4, MOV, M4V, or WebM video.");
      return;
    }
    setFile(candidate);
    setError("");
  };

  const submit = () => {
    if (!file || uploading) return;
    setUploading(true);
    setError("");
    const data = new FormData();
    data.append("video", file);
    data.append("max_frames", "16");
    data.append("max_objects", "12");
    data.append("preset", preset);
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/api/jobs`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setUploadProgress(event.loaded / event.total);
      }
    };
    request.onerror = () => {
      setError("Could not reach the reconstruction service. Try again in a moment.");
      setUploading(false);
    };
    request.onload = () => {
      setUploading(false);
      if (request.status < 200 || request.status >= 300) {
        try {
          setError(JSON.parse(request.responseText).detail || "Upload failed.");
        } catch {
          setError("Upload failed.");
        }
        return;
      }
      const response = JSON.parse(request.responseText);
      onStarted(response.jobId, response.eventsUrl, Boolean(response.cacheHit));
    };
    request.send(data);
  };

  return (
    <section className="landing-composition">
      <div className="landing-copy">
        <h1>Scan. Pack. Honk your way out.</h1>
        <p>
          Take a quick 360° video of your packed items at home and let our AI handle the rest.
        </p>
        <p>
          We rebuild your objects for virtual truck fitting.
        </p>
      </div>

      <div className="upload-card">
        <img
          className="hero-mascot"
          src="/brand/honkpack-hero.png"
          alt="HonkPack goose filming packed items"
        />
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            acceptFile(event.dataTransfer.files[0]);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,.m4v"
            onChange={(event) => acceptFile(event.target.files?.[0])}
            hidden
          />

          <img
            className="upload-phone"
            src="/brand/upload-phone.jpeg"
            alt=""
            aria-hidden="true"
          />
          {file ? (
            <div className="selected-file">
              <FileVideo size={22} />
              <div>
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(1)} MB · ready to reconstruct</span>
              </div>
            </div>
          ) : (
            <div className="drop-copy">
              <strong>Drag and drop your moving item video here.</strong>
              <span>MP4, MOV or WebM · a 10–30 second walk-around works best</span>
            </div>
          )}

          <div className="source-actions">
            <button className="choose-file" type="button" onClick={() => inputRef.current?.click()}>
              <CloudUpload size={18} />
              {file ? "Choose again" : "Choose files"}
            </button>
            <PhonePairingButton onStarted={onStarted} />
          </div>

          <div className="preset-row">
            <span>Reconstruction quality</span>
            <div className="segmented">
              {["speed", "balance", "quality"].map((value) => (
                <button
                  key={value}
                  type="button"
                  className={preset === value ? "active" : ""}
                  onClick={() => setPreset(value)}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>

          {error && <div className="form-error">{error}</div>}

          <button className="primary-action" disabled={!file || uploading} onClick={submit}>
            {uploading ? (
              <>
                <LoaderCircle className="spin" size={18} />
                Uploading {Math.round(uploadProgress * 100)}%
              </>
            ) : (
              <>
                <Play size={17} fill="currentColor" />
                Reconstruct my items
              </>
            )}
          </button>
          {uploading && (
            <div className="upload-progress">
              <span style={{ width: `${uploadProgress * 100}%` }} />
            </div>
          )}
        </div>
      </div>

      <img
        className="city-line"
        src="/brand/waterloo-line.jpeg"
        alt=""
        aria-hidden="true"
      />

      <footer className="landing-footer">
        <span className="footer-brand-name"><b>HonkPack</b> @ HackMIT 2026</span>
      </footer>
    </section>
  );
}

function PipelineTimeline({
  events,
  currentStage,
  selectedStage,
  interactive,
  onSelect,
}: {
  events: PipelineEvent[];
  currentStage: string;
  selectedStage: string;
  interactive: boolean;
  onSelect: (stage: string, event?: PipelineEvent) => void;
}) {
  const activeIndex = stageIndex(currentStage);
  return (
    <div className="pipeline-list">
      {STAGES.map((stage, index) => {
        const Icon = stage.icon;
        const complete = index < activeIndex || currentStage === "complete";
        const active = index === activeIndex && currentStage !== "complete";
        const statusEvent = [...events].reverse().find((item) => item.stage === stage.id);
        const previewEvent = [...events].reverse().find((item) => item.stage === stage.id && item.artifact);
        const dedicatedView = stage.id === "complete" || stage.id === "shaper" || stage.id === "room";
        const canView = interactive && (dedicatedView || Boolean(previewEvent));
        return (
          <button
            key={stage.id}
            type="button"
            disabled={!canView}
            className={`pipeline-step ${active ? "active" : ""} ${complete ? "done" : ""} ${interactive && selectedStage === stage.id ? "selected" : ""}`}
            onClick={() => canView && onSelect(stage.id, previewEvent)}
          >
            <span className="step-icon">
              {complete ? <Check size={15} /> : active ? <LoaderCircle className="spin" size={15} /> : <Icon size={15} />}
            </span>
            <span>
              <strong>{stage.label}</strong>
              <small>{statusEvent?.title || (active ? "Processing…" : "Waiting")}</small>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ArtifactPreview({ event }: { event?: PipelineEvent }) {
  if (!event?.artifact) {
    return (
      <div className="empty-preview">
        <ScanSearch size={38} />
        <strong>Pipeline output will appear here</strong>
        <span>Frames · masks · point cloud · object mesh</span>
      </div>
    );
  }
  if (event.artifact.kind === "image") {
    return (
      <div className="image-preview">
        <img src={absoluteUrl(event.artifact.url)} alt={event.title} />
        <div className="artifact-caption">
          <span>{event.stage}</span>
          <strong>{event.title}</strong>

        </div>
      </div>
    );
  }
  return <ProcessingModelPreview url={absoluteUrl(event.artifact.url)} />;
}

function Inspector({
  name,
  result,
  item,
  onClose,
}: {
  name: string;
  result: JobResult;
  item?: PackingItem;
  onClose: () => void;
}) {
  const object = result.objects.find((candidate) => candidate.name === name);
  const room = name.startsWith("room_");
  const weight = item ? estimatePackingItemWeight(item) : null;
  return (
    <aside className="inspector">
      <div className="panel-heading">
        <span>Item details</span>
        <button className="panel-close" onClick={onClose} aria-label="Close item details">
          <PanelRightClose size={16} />
        </button>
      </div>
      <div className="object-orb">
        {item ? <PackingItemThumbnail item={item} /> : <Layers3 size={28} />}
      </div>
      <h2>{object?.label || prettyName(name)}</h2>
      <div className="metric-grid item-metric-grid">
        {item ? (
          <>
            <div>
              <span>Size</span>
              <strong>{item.width.toFixed(2)} × {item.height.toFixed(2)} × {item.depth.toFixed(2)} m</strong>
            </div>
            <div>
              <span>Estimated weight</span>
              <strong>~{weight?.kg.toFixed(1)} kg</strong>
            </div>
          </>
        ) : (
          <div>
            <span>Surface</span>
            <strong>{room ? "Room geometry" : "Reconstructed item"}</strong>
          </div>
        )}
      </div>
    </aside>
  );
}

export function Studio() {
  const [phase, setPhase] = useState<"idle" | "processing" | "complete" | "error">("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [result, setResult] = useState<JobResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("queued");
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [hoveredName, setHoveredName] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const eventSocket = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const reconnectEnabled = useRef(false);
  const replayRun = useRef(0);
  const [mobilePairId, setMobilePairId] = useState<string | null | undefined>(undefined);
  const [resultView, setResultView] = useState<ResultView>("scene");
  const [pipelinePreview, setPipelinePreview] = useState<PipelineEvent | null>(null);
  const [selectedPipelineStage, setSelectedPipelineStage] = useState("complete");
  const [truckVisited, setTruckVisited] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const pairId = params.get("pair");
    setMobilePairId(params.get("mobile") === "1" && pairId ? pairId : null);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (!cancelled) setBackendOnline(response.ok);
      } catch {
        if (!cancelled) setBackendOnline(false);
      }
    };
    check();
    const timer = window.setInterval(check, 8000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => () => {
    replayRun.current += 1;
    reconnectEnabled.current = false;
    eventSocket.current?.close();
    if (reconnectTimer.current !== null) window.clearTimeout(reconnectTimer.current);
  }, []);

  const connectEvents = useCallback((id: string, _eventsUrl: string, replayCached = false) => {
    const run = ++replayRun.current;
    reconnectEnabled.current = false;
    eventSocket.current?.close();
    if (reconnectTimer.current !== null) window.clearTimeout(reconnectTimer.current);
    reconnectEnabled.current = true;

    const stop = () => {
      reconnectEnabled.current = false;
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      eventSocket.current?.close();
    };

    const handle = (event: PipelineEvent) => {
      setEvents((current) =>
        current.some((item) => item.id === event.id)
          ? current
          : [...current, event],
      );
      setProgress(event.progress);
      setCurrentStage(event.stage);
      if (event.type === "complete") {
        setResult(event.payload as unknown as JobResult);
        setPhase("complete");
        setSelectedPipelineStage("complete");
        setPipelinePreview(null);
        stop();
      } else if (event.type === "error" || event.type === "cancelled") {
        setPhase("error");
        stop();
      }
    };

    const connect = () => {
      const socket = new WebSocket(
        websocketUrl(`/api/jobs/${encodeURIComponent(id)}/events/ws`),
      );
      eventSocket.current = socket;
      socket.onmessage = (message) => {
        try {
          handle(JSON.parse(String(message.data)) as PipelineEvent);
        } catch {
          // Ignore malformed transport frames; valid events continue streaming.
        }
      };
      socket.onclose = () => {
        if (!reconnectEnabled.current) return;
        reconnectTimer.current = window.setTimeout(connect, 1000);
      };
      socket.onerror = () => socket.close();
    };

    if (!replayCached) {
      connect();
      return;
    }

    const replay = async () => {
      try {
        const response = await fetch(
          `${API_BASE}/api/jobs/${encodeURIComponent(id)}/events/history`,
        );
        if (!response.ok) throw new Error("Could not load cached reconstruction history");
        const payload = await response.json() as { events?: PipelineEvent[] };
        const history = payload.events ?? [];
        const lastForStage = (stage: string) => (
          [...history].reverse().find((event) => event.stage === stage && event.artifact) ||
          [...history].reverse().find((event) => event.stage === stage)
        );
        const firstForStage = (stage: string) => history.find((event) => event.stage === stage);
        const sequence = [
          lastForStage("frames"),
          lastForStage("segmentation"),
          lastForStage("geometry"),
          firstForStage("shaper"),
          ...history.filter((event) => event.stage === "shaper" && event.type === "object"),
          lastForStage("room"),
          [...history].reverse().find((event) => event.type === "complete"),
        ].filter((event): event is PipelineEvent => Boolean(event));

        for (const event of sequence) {
          if (run !== replayRun.current) return;
          handle(event);
          if (event.type === "complete") return;
          const delay = event.type === "object" ? 1000 : 2000;
          await new Promise((resolve) => window.setTimeout(resolve, delay));
        }
      } catch {
        if (run === replayRun.current) connect();
      }
    };

    void replay();
  }, []);

  useEffect(() => {
    const resumeId = new URLSearchParams(window.location.search).get("job");
    if (!resumeId || jobId) return;
    setJobId(resumeId);
    setPhase("processing");
    connectEvents(resumeId, `/api/jobs/${resumeId}/events`);
  }, [connectEvents, jobId]);

  const startJob = (id: string, eventsUrl: string, cacheHit = false) => {
    setJobId(id);
    setEvents([]);
    setResult(null);
    setProgress(0);
    setCurrentStage("queued");
    setPhase("processing");
    setResultView("scene");
    setPipelinePreview(null);
    setSelectedPipelineStage("complete");
    setTruckVisited(false);
    window.history.replaceState(null, "", `?job=${id}`);
    connectEvents(id, eventsUrl, cacheHit);
  };

  const packingItems = useMemo<PackingItem[]>(() => (
    result?.objects.map((object, index) => {
      const [width = 0.1, height = 0.1, depth = 0.1] = object.boundsMeters;
      return {
        id: object.name,
        name: object.label || prettyName(object.name),
        width: Math.max(0.05, Number(width) || 0.1),
        height: Math.max(0.05, Number(height) || 0.1),
        depth: Math.max(0.05, Number(depth) || 0.1),
        color: ITEM_COLORS[index % ITEM_COLORS.length],
        url: absoluteUrl(object.url),
      };
    }) ?? []
  ), [result]);

  const reset = async () => {
    replayRun.current += 1;
    reconnectEnabled.current = false;
    eventSocket.current?.close();
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    setPhase("idle");
    setJobId(null);
    setEvents([]);
    setResult(null);
    setSelectedName(null);
    setHoveredName(null);
    setResultView("scene");
    setPipelinePreview(null);
    setSelectedPipelineStage("complete");
    setTruckVisited(false);
    window.history.replaceState(null, "", window.location.pathname);
  };

  if (mobilePairId === undefined) {
    return <main className="mobile-capture-shell mobile-capture-loading" aria-busy="true" />;
  }
  if (mobilePairId) {
    return <MobileCapturePage pairId={mobilePairId} />;
  }

  const latestArtifact = [...events].reverse().find((event) => event.artifact);
  const displayName = hoveredName || selectedName;
  const statusEvent = events[events.length - 1];

  return (
    <main className={`app-shell ${phase === "idle" ? "landing-shell" : "workspace-shell"}`}>
      {phase === "idle" ? (
        <div className="landing-layout">
          <UploadPanel onStarted={startJob} />
        </div>
      ) : (
        <div className={"workspace " + (phase === "complete" && result && resultView === "truck" ? "truck-workspace" : "")} >
          {!(phase === "complete" && result && resultView === "truck") && (
          <aside className={`pipeline-panel ${phase === "processing" ? "has-loader" : ""}`}>
            <div className="panel-heading">
              <span>Reconstruction</span>
              <button className="panel-new" onClick={reset}>New capture</button>
            </div>
            <div className="progress-block">
              <div className="progress-number">{Math.round(progress * 100)}<small>%</small></div>
              <div>
                <strong>{statusEvent?.title || "Starting"}</strong>
              </div>
            </div>
            <div className="progress-track"><span style={{ width: `${progress * 100}%` }} /></div>
            <PipelineTimeline events={events} currentStage={currentStage} selectedStage={selectedPipelineStage} interactive={phase === "complete" && Boolean(result)} onSelect={(stage, event) => { setSelectedPipelineStage(stage); setPipelinePreview(event ?? null); setResultView("scene"); }} />

            {phase === "processing" && (
              <div className="reconstruction-goose-loader" role="img" aria-label="Reconstruction in progress">
                <video autoPlay loop muted playsInline preload="auto" aria-hidden="true">
                  <source src="/brand/reconstruction-loader-transparent.webm" type="video/webm" />
                </video>
              </div>
            )}

            {result && (
              <div className="object-list detected-items-list">
                <div className="list-title">
                  <span>Detected items</span>
                  <b>{result.objects.length}</b>
                </div>
                {result.objects.map((object) => {
                  const item = packingItems.find((candidate) => candidate.id === object.name);
                  const weight = item ? estimatePackingItemWeight(item) : null;
                  return (
                    <button
                      key={object.name}
                      className={selectedName === object.name ? "selected" : ""}
                      onClick={() => setSelectedName(object.name)}
                    >
                      <span className="item-thumb detected-item-thumb">{item ? <PackingItemThumbnail item={item} /> : <Cuboid size={15} />}</span>
                      <span>
                        <strong>{object.label}</strong>
                        {item && <small>{item.width.toFixed(2)} × {item.height.toFixed(2)} × {item.depth.toFixed(2)} m · ~{weight?.kg.toFixed(1)} kg</small>}
                      </span>
                      <ChevronRight size={14} />
                    </button>
                  );
                })}
              </div>
            )}
          </aside>
          )}

          <section className={phase === "complete" && result ? "main-stage complete-stage" : "main-stage"}>
            {!(phase === "complete" && result) && (
              <div className="stage-header">
                <h2>{statusEvent?.title || "Reconstructing"}</h2>
              </div>
            )}
            <div className="stage-canvas">
              {phase === "complete" && result ? (
                <>
                  {resultView === "scene" && selectedPipelineStage === "complete" && (
                    <div className="result-switch scene-view-switch">
                      <button className="fit-truck-cta" onClick={() => { setTruckVisited(true); setResultView("truck"); }}>Fit in truck</button>
                    </div>
                  )}
                  {resultView === "scene" && selectedPipelineStage === "complete" && (
                    <img className="scene-fit-goose" src="/brand/pointing-goose.png" alt="" aria-hidden="true" />
                  )}
                  <div className={"scene-result-surface " + (resultView === "scene" ? "active" : "hidden")}>
                    {selectedPipelineStage === "shaper" ? (
                      <ObjectMeshGrid items={packingItems} />
                    ) : selectedPipelineStage === "room" ? (
                      <ProcessingModelPreview url={absoluteUrl(result.roomUrl)} />
                    ) : pipelinePreview && selectedPipelineStage !== "complete" ? (
                      <ArtifactPreview event={pipelinePreview} />
                    ) : (
                      <SceneViewport
                        url={absoluteUrl(result.sceneUrl)}
                        selectedName={selectedName}
                        onSelect={setSelectedName}
                        onHover={setHoveredName}
                      />
                    )}
                  </div>
                  {truckVisited && (
                    <div className={"truck-result-surface " + (resultView === "truck" ? "active" : "hidden")}>
                      <TruckFitting items={packingItems} onViewScene={() => { setSelectedPipelineStage("complete"); setPipelinePreview(null); setResultView("scene"); }} />
                    </div>
                  )}
                </>
              ) : (
                <ArtifactPreview event={latestArtifact} />
              )}
            </div>

          </section>

          {phase === "complete" && result && resultView === "scene" && selectedPipelineStage === "complete" && displayName && (
            <Inspector
              name={displayName}
              result={result}
              item={packingItems.find((candidate) => candidate.id === displayName)}
              onClose={() => {
                setSelectedName(null);
                setHoveredName(null);
              }}
            />
          )}
        </div>
      )}
    </main>
  );
}

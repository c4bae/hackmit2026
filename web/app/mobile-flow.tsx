"use client";

import { Camera, CheckCircle2, CloudUpload, LoaderCircle, QrCode, Smartphone, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { QRCodeSVG } from "qrcode.react";

const API_BASE =
  process.env.NEXT_PUBLIC_SHAPER_API_URL?.replace(/\/$/, "") ||
  "/api/shaper";

const PUBLIC_BASE = process.env.NEXT_PUBLIC_HONKPACK_PUBLIC_URL;

type PairingEvent = {
  id: number;
  type: string;
  status: string;
  title: string;
  detail: string;
  payload?: {
    jobId?: string;
    eventsUrl?: string;
    cacheHit?: boolean;
  };
};

function websocketUrl(path: string) {
  const base = new URL(API_BASE, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}${path}`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

function phoneUrl(pairId: string) {
  const url = new URL("/", PUBLIC_BASE || window.location.origin);
  url.searchParams.set("mobile", "1");
  url.searchParams.set("pair", pairId);
  return url.toString();
}

export function PhonePairingButton({
  onStarted,
}: {
  onStarted: (jobId: string, eventsUrl: string, cacheHit?: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pairId, setPairId] = useState<string | null>(null);
  const [mobileUrl, setMobileUrl] = useState("");
  const [status, setStatus] = useState("Creating a secure phone link…");
  const [error, setError] = useState("");
  const startedRef = useRef(onStarted);

  useEffect(() => {
    startedRef.current = onStarted;
  }, [onStarted]);

  useEffect(() => {
    if (!pairId) return;
    let active = true;
    let socket: WebSocket | null = null;
    let retryTimer: number | null = null;
    let pollTimer: number | null = null;

    const continueOnDesktop = (jobId: string, eventsUrl?: string, cacheHit = false) => {
      if (!active) return;
      active = false;
      socket?.close();
      startedRef.current(
        jobId,
        eventsUrl || `/api/jobs/${jobId}/events`,
        cacheHit,
      );
    };

    const connect = () => {
      if (!active) return;
      socket = new WebSocket(websocketUrl(`/api/pairings/${encodeURIComponent(pairId)}/events/ws`));
      socket.onmessage = (message) => {
        try {
          const event = JSON.parse(String(message.data)) as PairingEvent;
          setStatus(event.title);
          if (event.type === "submitted" && event.payload?.jobId) {
            continueOnDesktop(
              event.payload.jobId,
              event.payload.eventsUrl || `/api/jobs/${event.payload.jobId}/events`,
              Boolean(event.payload.cacheHit),
            );
          } else if (event.type === "expired") {
            active = false;
            setError(event.detail || "This QR code expired. Create a new one.");
            socket?.close();
          }
        } catch {
          // Ignore malformed frames; the pairing socket remains authoritative.
        }
      };
      socket.onclose = () => {
        if (!active) return;
        retryTimer = window.setTimeout(connect, 1000);
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    const poll = async () => {
      if (!active) return;
      try {
        const response = await fetch(`${API_BASE}/api/pairings/${encodeURIComponent(pairId)}`);
        const payload = await response.json() as { status?: string; jobId?: string };
        if (response.ok && payload.status === "submitted" && payload.jobId) {
          continueOnDesktop(payload.jobId);
          return;
        }
      } catch {
        // The WebSocket remains primary; polling retries transient HTTP failures.
      }
      if (active) pollTimer = window.setTimeout(poll, 1000);
    };
    pollTimer = window.setTimeout(poll, 1000);
    return () => {
      active = false;
      socket?.close();
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      if (pollTimer !== null) window.clearTimeout(pollTimer);
    };
  }, [pairId]);

  const createPairing = async () => {
    setOpen(true);
    setPairId(null);
    setMobileUrl("");
    setError("");
    setStatus("Creating a secure phone link…");
    try {
      const response = await fetch(`${API_BASE}/api/pairings`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not create a phone link.");
      setPairId(payload.pairId);
      setMobileUrl(phoneUrl(payload.pairId));
      setStatus("Scan with your phone");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create a phone link.");
    }
  };

  const close = () => {
    setOpen(false);
    setPairId(null);
  };

  return (
    <>
      <button className="phone-pair-button" type="button" onClick={createPairing}>
        <QrCode size={18} />
        Use phone camera
      </button>

      {open && typeof document !== "undefined" && createPortal(
        (
        <div className="pairing-overlay" role="dialog" aria-modal="true" aria-labelledby="pairing-title">
          <div className="pairing-dialog">
            <button className="pairing-close" type="button" onClick={close} aria-label="Close phone pairing">
              <X size={19} />
            </button>
            <div className="pairing-heading">
              <span><Smartphone size={22} /></span>
              <div>
                <h2 id="pairing-title">Continue on your phone</h2>
                <p>Scan once, record or choose a video, and this desktop will continue automatically.</p>
              </div>
            </div>

            <div className="pairing-qr-stage">
              {mobileUrl ? (
                <div className="pairing-qr">
                  <QRCodeSVG value={mobileUrl} size={210} level="M" marginSize={2} />
                </div>
              ) : (
                <div className="pairing-qr pairing-qr-loading"><LoaderCircle className="spin" size={30} /></div>
              )}
              <img
                className="pairing-qr-mascot"
                src="/brand/qr-goose-transparent.png"
                alt=""
                aria-hidden="true"
              />
            </div>

            <strong className="pairing-status">{status}</strong>
            <p className="pairing-help">Keep this window open. The QR link expires after 20 minutes and can be used once.</p>
            {mobileUrl && (
              <div className="pairing-link">
                <span>Phone link</span>
                <div className="pairing-link-value" title={mobileUrl}>{mobileUrl}</div>
              </div>
            )}
            {error && <div className="pairing-error">{error}</div>}
          </div>
        </div>
        ),
        document.body,
      )}
    </>
  );
}

export function MobileCapturePage({ pairId }: { pairId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<"checking" | "ready" | "uploading" | "done" | "error">("checking");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("Checking the secure link…");
  const libraryRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/pairings/${encodeURIComponent(pairId)}`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "This phone link is no longer available.");
        if (!active) return;
        if (payload.status === "submitted") {
          setState("done");
          setMessage("This video was already sent to the desktop.");
        } else {
          setState("ready");
          setMessage("Walk once around your packed items for the best result.");
        }
      })
      .catch((caught) => {
        if (!active) return;
        setState("error");
        setMessage(caught instanceof Error ? caught.message : "This phone link is no longer available.");
      });
    return () => { active = false; };
  }, [pairId]);

  const acceptFile = (candidate?: File) => {
    if (!candidate) return;
    const isVideo = candidate.type.startsWith("video/") || /\.(mp4|mov|m4v|webm)$/i.test(candidate.name);
    if (!isVideo) {
      setState("error");
      setMessage("Choose an MP4, MOV, M4V, or WebM video.");
      return;
    }
    setFile(candidate);
    setState("ready");
    setMessage("Video ready. Send it to the desktop when you are happy with it.");
  };

  const upload = () => {
    if (!file || state === "uploading") return;
    const data = new FormData();
    data.append("video", file);
    data.append("max_frames", "16");
    data.append("max_objects", "12");
    data.append("preset", "balance");
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/api/pairings/${encodeURIComponent(pairId)}/jobs`);
    setState("uploading");
    setProgress(0);
    setMessage("Uploading securely to the desktop…");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setProgress(event.loaded / event.total);
    };
    request.onerror = () => {
      setState("error");
      setMessage("The upload was interrupted. Check the connection and try again.");
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        setProgress(1);
        setState("done");
        setMessage("Video received. Reconstruction is now continuing on the desktop.");
        return;
      }
      try {
        setMessage(JSON.parse(request.responseText).detail || "Upload failed.");
      } catch {
        setMessage("Upload failed.");
      }
      setState("error");
    };
    request.send(data);
  };

  return (
    <main className="mobile-capture-shell">
      <header className="mobile-capture-brand">
        <img src="/brand/honkpack-mark.png" alt="" />
        <span>HonkPack</span>
      </header>
      <section className="mobile-capture-card">
        {state === "done" ? (
          <div className="mobile-success">
            <CheckCircle2 size={54} />
            <h1>Sent to your desktop</h1>
            <p>{message}</p>
            <span>You can close this page.</span>
          </div>
        ) : (
          <>
            <div className="mobile-capture-intro">
              <span><Camera size={25} /></span>
              <h1>Capture your packed items</h1>
              <p>{message}</p>
            </div>
            <input ref={cameraRef} type="file" accept="video/*" capture="environment" hidden onChange={(event) => acceptFile(event.target.files?.[0])} />
            <input ref={libraryRef} type="file" accept="video/mp4,video/quicktime,video/webm,.m4v" hidden onChange={(event) => acceptFile(event.target.files?.[0])} />
            <div className="mobile-source-actions">
              <button type="button" className="mobile-record-button" onClick={() => cameraRef.current?.click()} disabled={state === "checking" || state === "uploading"}>
                <Camera size={20} /> Record video
              </button>
              <button type="button" className="mobile-library-button" onClick={() => libraryRef.current?.click()} disabled={state === "checking" || state === "uploading"}>
                <CloudUpload size={20} /> Choose video
              </button>
            </div>
            {file && (
              <div className="mobile-file">
                <FileVideoIcon />
                <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(1)} MB</span></div>
              </div>
            )}
            {state === "uploading" && (
              <div className="mobile-upload-progress"><span style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            )}
            <button type="button" className="mobile-send-button" onClick={upload} disabled={!file || state === "checking" || state === "uploading" || state === "error"}>
              {state === "uploading" ? <><LoaderCircle className="spin" size={20} /> Uploading {Math.round(progress * 100)}%</> : <>Send video to desktop</>}
            </button>
            {state === "error" && <div className="mobile-capture-error">{message}</div>}
          </>
        )}
      </section>
      <footer>HonkPack @ HackMIT 2026</footer>
    </main>
  );
}

function FileVideoIcon() {
  return <span className="mobile-file-icon"><CloudUpload size={20} /></span>;
}

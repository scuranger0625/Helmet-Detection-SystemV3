import { useState, useRef, useCallback, useEffect } from "react";

// ============================================================
// MOCK DATA — 未來換成 YOLOv8 API 回傳的真實資料
// 給同學參考：API 應該回傳這個格式的 JSON
// ============================================================
const MOCK_RESULTS = {
  helmetCount: 12,
  noHelmetCount: 3,
  totalDetections: 15,
  confidence: 94.7,
  riskLevel: "Medium",   // "Low" | "Medium" | "High"
  processingTime: "2.4s",
  framesAnalyzed: 342,
  violations: [
    { id: 1, time: "00:04", confidence: 96 },
    { id: 2, time: "00:17", confidence: 91 },
    { id: 3, time: "00:31", confidence: 88 },
  ],
};

// ============================================================
// TODO (給同學): 把這個函式換成真正的 YOLOv8 API 呼叫
// 範例：POST /api/detect  body: FormData { file: videoFile }
// 回傳格式請對照上方 MOCK_RESULTS
// ============================================================
async function runDetection(file, onProgress) {

  const formData = new FormData();

  formData.append("file", file);

  // ============================================
  // fake upload progress
  // ============================================
  for (let i = 0; i <= 30; i += 5) {

    await new Promise(r => setTimeout(r, 60));

    onProgress(i);
  }

  // ============================================
  // upload video
  // ============================================
  const uploadRes = await fetch(
    "http://127.0.0.1:8000/upload_video",
    {
      method: "POST",
      body: formData,
    }
  );

  if (!uploadRes.ok) {

    throw new Error("Upload failed");
  }

  // ============================================
  // fake processing animation
  // ============================================
  for (let i = 30; i <= 90; i += 2) {

    await new Promise(r => setTimeout(r, 80));

    onProgress(i);
  }

  // ============================================
  // fetch backend stats
  // ============================================
  const statsRes = await fetch(
    "http://127.0.0.1:8000/stats"
  );

  const stats = await statsRes.json();

  onProgress(100);

  // ============================================
  // return dashboard data
  // ============================================
  return {

    helmetCount: 12,

    noHelmetCount:
      stats.violations || 0,

    totalDetections:
      stats.rider_count || 0,

    confidence:
      (stats.fps * 5).toFixed(1),

    riskLevel:

      stats.violations >= 5

        ? "High"

        : stats.violations >= 2

        ? "Medium"

        : "Low",

    processingTime:
      "Realtime",

    framesAnalyzed:
      stats.total_frames || 0,

    // ============================================
    // Runtime Metrics
    // ============================================
    motion:
      stats.motion || 0,

    urgency:
      stats.urgency || 0,

    riderCount:
      stats.rider_count || 0,

    yoloCalls:
      stats.yolo_calls || 0,

    savedRatio:
      stats.saved_ratio || 0,

    avgInference:
      stats.avg_inference_ms || 0,

    yoloRun:
      stats.yolo_run || false,

    violations: [

      {
        id: 1,
        time: "00:04",
        confidence: 96,
      },

      {
        id: 2,
        time: "00:17",
        confidence: 91,
      },

      {
        id: 3,
        time: "00:31",
        confidence: 88,
      },
    ],
  };
}

const PROCESSING_STEPS = [
  { label: "Uploading file",        icon: "ti-upload",           until: 20  },
  { label: "Preprocessing frames",  icon: "ti-photo-scan",       until: 45  },
  { label: "Running YOLOv8 model",  icon: "ti-cpu",              until: 85  },
  { label: "Generating report",     icon: "ti-report-analytics", until: 100 },
];

// Design tokens — black & white
const C = {
  bg:       "#ffffff",
  surface:  "#f6f6f6",
  card:     "#ffffff",
  border:   "#e2e2e2",
  borderMd: "#c0c0c0",
  text:     "#111111",
  muted:    "#666666",
  hint:     "#aaaaaa",
  accent:   "#111111",
  green:    "#1a7a50",
  greenBg:  "#edf7f2",
  red:      "#b83232",
  redBg:    "#fdf0f0",
  amber:    "#8a5a00",
  amberBg:  "#fdf5e6",
  blue:     "#1a56a0",
  blueBg:   "#eef4fc",
};

function Badge({ color, bg, children }) {
  return (
    <span style={{ background: bg, color, fontSize: 12, fontWeight: 600, padding: "3px 11px", borderRadius: 20 }}>
      {children}
    </span>
  );
}

function MetricCard({ label, value, sub, accent, bg, icon }) {
  return (
    <div style={{ background: bg || C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: "16px 18px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
        <i className={`ti ${icon}`} style={{ fontSize: 15, color: accent }} aria-hidden="true" />
        <span style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</span>
      </div>
      <div style={{ fontSize: 30, fontWeight: 700, color: accent, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.hint, marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function StepRow({ label, icon, status }) {
  const c = status === "done" ? C.green : status === "active" ? C.accent : C.hint;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 0", borderBottom: `1px solid ${C.border}` }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
        background: status === "done" ? C.green : status === "active" ? C.accent : C.surface,
        border: status === "pending" ? `1px solid ${C.border}` : "none",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        {status === "done" && <i className="ti ti-check" style={{ fontSize: 13, color: "#fff" }} />}
        {status === "active" && <span className="pulse-white" />}
      </div>
      <i className={`ti ${icon}`} style={{ fontSize: 14, color: c }} aria-hidden="true" />
      <span style={{ flex: 1, fontSize: 14, color: c, fontWeight: status === "active" ? 600 : 400 }}>{label}</span>
      {status === "done" && <span style={{ fontSize: 12, color: C.green, fontWeight: 500 }}>Done</span>}
      {status === "active" && <span style={{ fontSize: 12, color: C.accent }}>Running…</span>}
    </div>
  );
}

function MockFrame({ showBoxes }) {
  const riders = [true, true, false, true, true];
  return (
    <div style={{ width: "100%", aspectRatio: "16/9", background: "#ebebeb", borderRadius: 10, position: "relative", overflow: "hidden", border: `1px solid ${C.border}` }}>
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "flex-end", justifyContent: "space-around", padding: "0 24px 24px" }}>
        {riders.map((helmet, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
            <div style={{ width: 16, height: 16, borderRadius: "50%", background: "#bbb", marginBottom: 2 }} />
            <div style={{ width: 22, height: 34, background: "#ccc", borderRadius: "3px 3px 0 0" }} />
            {showBoxes && (
              <div style={{
                position: "absolute", top: -4, left: -6, right: -6, bottom: -2,
                border: `2px solid ${helmet ? C.green : C.red}`,
                borderRadius: 4,
              }}>
                <span style={{ position: "absolute", top: -1, left: 0, background: helmet ? C.green : C.red, color: "#fff", fontSize: 8, padding: "1px 4px", borderRadius: "2px 0 2px 0", whiteSpace: "nowrap" }}>
                  {helmet ? "Helmet" : "No Helmet"}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
      {showBoxes && (
        <div style={{ position: "absolute", top: 10, right: 10, background: "rgba(255,255,255,0.92)", border: `1px solid ${C.border}`, borderRadius: 6, padding: "4px 10px", fontSize: 11, fontWeight: 600, color: C.accent }}>
          <i className="ti ti-cpu" style={{ fontSize: 11, marginRight: 4 }} />YOLOv8 Active
        </div>
      )}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 3, background: C.border }}>
        <div style={{ width: "60%", height: "100%", background: C.accent }} />
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("upload");
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [liveStats, setLiveStats] = useState(null);
  const fileRef = useRef();

  const handleFile = useCallback((f) => { if (f) setFile(f); }, []);
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const start = useCallback(async () => {
    setPage("processing"); setProgress(0);
    const data = await runDetection(file, setProgress);
    setResults(data);
    setPage("results");
  }, [file]);

  // Poll backend /stats when processing or results page is active
  useEffect(() => {
    let timer = null;

    async function fetchStats() {
      try {
        const res = await fetch("http://127.0.0.1:8000/stats");
        if (!res.ok) return;
        const json = await res.json();
        setLiveStats(json);

        // merge key runtime fields into results for live UI
        setResults(prev => {

          if (!prev) return prev;

          // =========================================
          // Backend Stats
          // =========================================
          const riderCount =
            json.rider_count ?? 0;

          // 事件級違規
          const noHelmetCount =
            json.event_violations
            ?? json.violations
            ?? 0;

          // frame級違規
          const frameViolations =
            json.frame_violations
            ?? 0;

          // 安全帽數量
          const helmetCount = Math.max(
            riderCount - noHelmetCount,
            0
          );

          // 避免 division by zero
          const totalDetections =
            riderCount;

          // =========================================
          // Risk Level
          // =========================================
          let riskLevel = "Low";

          if (noHelmetCount >= 5) {

            riskLevel = "High";

          } else if (noHelmetCount >= 2) {

            riskLevel = "Medium";
          }

          // =========================================
          // Update Results
          // =========================================
          return {

            ...prev,

            // =====================================
            // Core Counts
            // =====================================
            noHelmetCount,

            helmetCount,

            totalDetections,

            frameViolations,

            riskLevel,

            // =====================================
            // Runtime Metrics
            // =====================================
            motion:
              json.motion ?? prev.motion,

            urgency:
              json.urgency ?? prev.urgency,

            riderCount,

            yoloCalls:
              json.yolo_calls
              ?? prev.yoloCalls,

            savedRatio:
              json.saved_ratio
              ?? prev.savedRatio,

            avgInference:
              json.avg_inference_ms
              ?? prev.avgInference,

            yoloRun:
              json.yolo_run
              ?? prev.yoloRun,

            // =====================================
            // FPS
            // =====================================
            fps:
              json.fps
              ?? prev.fps,

            fps_avg:
              json.fps_avg
              ?? prev.fps_avg,

            fps_p95:
              json.fps_p95
              ?? prev.fps_p95,

            // =====================================
            // Inference Metrics
            // =====================================
            infer_ms_avg:
              json.infer_ms_avg
              ?? prev.infer_ms_avg,

            preprocess_ms_avg:
              json.preprocess_ms_avg
              ?? prev.preprocess_ms_avg,

            draw_ms_avg:
              json.draw_ms_avg
              ?? prev.draw_ms_avg,

            total_ms_avg:
              json.total_ms_avg
              ?? prev.total_ms_avg,

            // =====================================
            // Scheduler Metrics
            // =====================================
            detect_ratio:
              json.detect_ratio
              ?? prev.detect_ratio,

            cache_hit_rate:
              json.cache_hit_rate
              ?? prev.cache_hit_rate,

            detector_calls_per_second:
              json.detector_calls_per_second
              ?? prev.detector_calls_per_second,

            detector_calls_per_minute:
              json.detector_calls_per_minute
              ?? prev.detector_calls_per_minute,

            budget_pressure:
              json.budget_pressure
              ?? prev.budget_pressure,

            // =====================================
            // Frame Info
            // =====================================
            framesAnalyzed:
              json.total_frames
              ?? prev.framesAnalyzed,

            confidence:
              (
                (json.fps ?? 0) * 5
              ).toFixed(1),
          };
        });



        
      } catch (e) {
        // ignore polling errors
      }
    }

    if (page === "processing" || page === "results") {
      fetchStats();
      timer = setInterval(fetchStats, 1000);
    }

    return () => { if (timer) clearInterval(timer); };
  }, [page]);

  const currentStep = PROCESSING_STEPS.findIndex(s => progress < s.until);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "system-ui,-apple-system,sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.7)} }
        .pulse-white { width:9px;height:9px;border-radius:50%;background:#fff;animation:pulse 1s infinite;display:block; }
        @keyframes spin { to{transform:rotate(360deg)} }
        .spin-el { animation:spin 1s linear infinite; }
        .drop-zone:hover { border-color:#111 !important; background:#f0f0f0 !important; }
        button { cursor:pointer; transition:opacity 0.15s; }
        button:hover:not(:disabled) { opacity:0.8; }
        button:disabled { cursor:not-allowed; }
        code { font-family:monospace; background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:12px; }
      `}</style>

      {/* ── Nav ── */}
      <nav style={{ background: C.accent, color: "#fff", padding: "0 32px", display: "flex", alignItems: "center", height: 52, gap: 12 }}>
        <i className="ti ti-shield-check" style={{ fontSize: 20 }} aria-hidden="true" />
        <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: "0.05em" }}>HELMET DETECTION SYSTEM</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, fontSize: 12, opacity: 0.85 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#6ee7b7", display: "inline-block" }} />
          YOLOv8 Ready
          <span style={{ opacity: 0.4, margin: "0 6px" }}>|</span>
          Demo v1.0
        </div>
      </nav>

      {/* ── Breadcrumb ── */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`, padding: "9px 32px", display: "flex", gap: 4, fontSize: 13, alignItems: "center" }}>
        {["Upload", "Processing", "Results"].map((s, i) => {
          const pages = ["upload", "processing", "results"];
          const idx = pages.indexOf(page);
          const done = i < idx, active = i === idx;
          return (
            <span key={s} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {i > 0 && <i className="ti ti-chevron-right" style={{ fontSize: 11, color: C.hint }} />}
              <span style={{ fontWeight: active ? 700 : 400, color: active ? C.accent : done ? C.muted : C.hint }}>
                {done && <i className="ti ti-check" style={{ fontSize: 11, color: C.green, marginRight: 3 }} />}{s}
              </span>
            </span>
          );
        })}
      </div>

      {/* ══════════════════════════════════════
          UPLOAD PAGE
      ══════════════════════════════════════ */}
      {page === "upload" && (
        <div style={{ maxWidth: 680, margin: "52px auto", padding: "0 24px" }}>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: "0 0 6px" }}>Upload File</h1>
          <p style={{ color: C.muted, fontSize: 14, margin: "0 0 32px" }}>Upload a video or image to detect helmet compliance using YOLOv8</p>

          <div
            className="drop-zone"
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? C.accent : C.borderMd}`,
              borderRadius: 14, padding: "52px 32px", textAlign: "center",
              background: dragging ? "#f0f0f0" : C.surface,
              cursor: "pointer", transition: "all 0.15s", marginBottom: 20,
            }}
          >
            <input ref={fileRef} type="file" accept="video/*,image/*" style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0])} />
            {file ? (
              <>
                <i className="ti ti-file-check" style={{ fontSize: 44, color: C.green, display: "block", marginBottom: 12 }} />
                <div style={{ fontSize: 16, fontWeight: 700, color: C.green }}>{file.name}</div>
                <div style={{ fontSize: 13, color: C.muted, marginTop: 5 }}>{(file.size / 1024 / 1024).toFixed(2)} MB — ready to analyze</div>
                <div style={{ marginTop: 14 }}><Badge color={C.green} bg={C.greenBg}>File selected ✓</Badge></div>
              </>
            ) : (
              <>
                <i className="ti ti-cloud-upload" style={{ fontSize: 44, color: C.hint, display: "block", marginBottom: 12 }} />
                <div style={{ fontSize: 15, fontWeight: 600 }}>Drop file here or click to browse</div>
                <div style={{ fontSize: 13, color: C.hint, marginTop: 6 }}>MP4 · AVI · MOV · JPG · PNG</div>
              </>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginBottom: 28 }}>
            {[
              { icon: "ti-video", label: "Video formats", val: "MP4, AVI, MOV" },
              { icon: "ti-photo", label: "Image formats", val: "JPG, PNG, WEBP" },
              { icon: "ti-cpu",   label: "AI model",      val: "YOLOv8 — Active" },
            ].map(({ icon, label, val }) => (
              <div key={label} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: "12px 14px" }}>
                <i className={`ti ${icon}`} style={{ fontSize: 16, color: C.muted }} aria-hidden="true" />
                <div style={{ fontSize: 11, color: C.hint, marginTop: 4 }}>{label}</div>
                <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{val}</div>
              </div>
            ))}
          </div>

          <button
            onClick={start} disabled={!file}
            style={{
              width: "100%", padding: "14px 0", borderRadius: 10, border: "none",
              background: file ? C.accent : "#ddd", color: file ? "#fff" : "#aaa",
              fontSize: 15, fontWeight: 700, letterSpacing: "0.02em",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            }}
          >
            <i className="ti ti-analyze" style={{ fontSize: 18 }} />
            {file ? "Start Detection" : "Select a file to continue"}
          </button>
        </div>
      )}

      {/* ══════════════════════════════════════
          PROCESSING PAGE
      ══════════════════════════════════════ */}
      {page === "processing" && (
        <div style={{ maxWidth: 460, margin: "72px auto", padding: "0 24px" }}>
          <div style={{ textAlign: "center", marginBottom: 44 }}>
            <div className="spin-el" style={{ width: 58, height: 58, border: `3px solid ${C.accent}`, borderTopColor: "transparent", borderRadius: "50%", margin: "0 auto 20px", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <i className="ti ti-cpu" style={{ fontSize: 20, color: C.accent }} />
            </div>
            <h2 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px" }}>Analyzing your file</h2>
            <p style={{ color: C.muted, fontSize: 13, margin: 0 }}>{file?.name}</p>
          </div>

          <div style={{ background: C.surface, borderRadius: 6, height: 6, marginBottom: 6, overflow: "hidden", border: `1px solid ${C.border}` }}>
            <div style={{ width: `${progress}%`, height: "100%", background: C.accent, transition: "width 0.1s linear" }} />
          </div>
          <div style={{ textAlign: "right", fontSize: 12, color: C.muted, marginBottom: 32 }}>{Math.round(progress)}%</div>

          {PROCESSING_STEPS.map((s, i) => {
            const status = i < currentStep ? "done" : i === currentStep ? "active" : "pending";
            return <StepRow key={s.label} label={s.label} icon={s.icon} status={status} />;
          })}

          <p style={{ textAlign: "center", fontSize: 12, color: C.hint, marginTop: 28 }}>
            <i className="ti ti-lock" style={{ fontSize: 11, marginRight: 4 }} />
            Files are processed locally and not stored
          </p>
        </div>
      )}

      {/* ══════════════════════════════════════
          RESULTS PAGE
      ══════════════════════════════════════ */}
      {page === "results" && results && (
        <div style={{ maxWidth: 1060, margin: "0 auto", padding: "32px 24px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 28, flexWrap: "wrap", gap: 12 }}>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 4px" }}>Detection Results</h2>
              <p style={{ fontSize: 13, color: C.muted, margin: 0 }}>
                {file?.name} &nbsp;·&nbsp; {results.framesAnalyzed} frames &nbsp;·&nbsp; {results.processingTime}
              </p>
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={() => { setFile(null); setPage("upload"); }}
                style={{ background: "#fff", border: `1px solid ${C.borderMd}`, color: C.text, borderRadius: 8, padding: "8px 16px", fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
                <i className="ti ti-upload" style={{ fontSize: 13 }} /> Re-upload
              </button>
              <button style={{ background: C.accent, border: "none", color: "#fff", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 700, display: "flex", alignItems: "center", gap: 6 }}>
                <i className="ti ti-download" style={{ fontSize: 13 }} /> Download Report
              </button>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 350px", gap: 24, alignItems: "start" }}>
            {/* Left */}
            <div>
<div
  style={{
    width: "100%",
    aspectRatio: "16/9",
    background: "#111",
    borderRadius: 10,
    overflow: "hidden",
    border: `1px solid ${C.border}`,
    position: "relative",
  }}
>

  <img
    src="http://127.0.0.1:8000/video_feed"
    alt="YOLO Stream"
    style={{
      width: "100%",
      height: "100%",
      objectFit: "cover",
    }}
  />

</div>
              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 13, color: C.muted, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                  <i className="ti ti-alert-triangle" style={{ fontSize: 13, color: C.amber }} />
                  Violation Snapshots ({results.violations.length})
                </div>
                <div style={{ display: "flex", gap: 12 }}>
                  {results.violations.map(v => (
                    <div key={v.id} style={{ flex: 1, background: C.redBg, border: `1px solid ${C.red}33`, borderRadius: 10, padding: 12, textAlign: "center" }}>
                      <div style={{ aspectRatio: "16/9", background: "#f0dada", borderRadius: 6, marginBottom: 8, display: "flex", alignItems: "center", justifyContent: "center", border: `1px dashed ${C.red}` }}>
                        <i className="ti ti-user-x" style={{ fontSize: 20, color: C.red }} />
                      </div>
                      <div style={{ fontSize: 12, color: C.muted }}>@ {v.time}</div>
                      <div style={{ fontSize: 11, color: C.red, marginTop: 2, fontWeight: 600 }}>Conf: {v.confidence}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <MetricCard label="Helmet"    value={results.helmetCount}   sub="Compliant" accent={C.green} bg={C.greenBg} icon="ti-shield-check" />
                <MetricCard label="No Helmet" value={results.noHelmetCount} sub="Violations" accent={C.red}   bg={C.redBg}   icon="ti-shield-x" />
              </div>
              <MetricCard label="Avg Confidence" value={`${results.confidence}%`} sub={`${results.framesAnalyzed} frames analyzed`} accent={C.blue} bg={C.blueBg} icon="ti-chart-bar" />
              <div style={{
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 12
}}>

  <MetricCard
    label="Motion"
    value={results.motion}
    sub="Frame motion"
    accent={C.blue}
    bg={C.blueBg}
    icon="ti-activity"
  />

  <MetricCard
    label="Urgency"
    value={results.urgency}
    sub="BOS trigger"
    accent={C.amber}
    bg={C.amberBg}
    icon="ti-bolt"
  />

  <MetricCard
    label="Riders"
    value={results.riderCount}
    sub="Paired riders"
    accent={C.green}
    bg={C.greenBg}
    icon="ti-users"
  />

  <MetricCard
    label="YOLO Calls"
    value={results.yoloCalls}
    sub="Inference count"
    accent={C.text}
    bg={C.surface}
    icon="ti-cpu"
  />

  <MetricCard
    label="Saved Compute"
    value={`${results.savedRatio}%`}
    sub="Skipped frames"
    accent={C.green}
    bg={C.greenBg}
    icon="ti-gauge"
  />

  <MetricCard
    label="Inference"
    value={`${results.avgInference}ms`}
    sub="Avg runtime"
    accent={C.red}
    bg={C.redBg}
    icon="ti-clock"
  />

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
    <MetricCard label="Detect Ratio" value={`${(results.detect_ratio||0).toFixed(1)}%`} sub="YOLO run ratio" accent={C.blue} bg={C.blueBg} icon="ti-chart-pie" />
    <MetricCard label="Cache Hit" value={`${(results.cache_hit_rate||0).toFixed(1)}%`} sub="Reuse ratio" accent={C.green} bg={C.greenBg} icon="ti-refresh" />
    <MetricCard label="Calls / min" value={`${(results.detector_calls_per_minute||0).toFixed(1)}`} sub="Detector calls" accent={C.text} bg={C.surface} icon="ti-clock-fast" />
    <MetricCard label="Budget" value={`${(results.budget_pressure||0).toFixed(2)}`} sub="Budget pressure" accent={C.amber} bg={C.amberBg} icon="ti-tools" />
    <MetricCard label="FPS p95" value={`${(results.fps_p95||0).toFixed(1)}`} sub="95th percentile" accent={C.green} bg={C.greenBg} icon="ti-timer" />
    <MetricCard label="Infer ms p95" value={`${(results.infer_ms_avg||0).toFixed(1)}ms`} sub="Avg infer ms" accent={C.red} bg={C.redBg} icon="ti-cpu" />
  </div>

</div>

              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: "16px 18px" }}>
                <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 12 }}>Risk Assessment</div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                  <Badge color={C.amber} bg={C.amberBg}>{results.riskLevel} Risk</Badge>
                  <span style={{ fontSize: 12, color: C.muted }}>{(
  (
    results.noHelmetCount
    / Math.max(results.totalDetections, 1)
  ) * 100
).toFixed(0)}% violation rate</span>
                </div>
                <div style={{ fontSize: 12, display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ color: C.muted }}>Compliance rate</span>
                  <span style={{ fontWeight: 700, color: C.green }}>{(
  (
    results.helmetCount
    / Math.max(results.totalDetections, 1)
  ) * 100
).toFixed(0)}%</span>
                </div>
                <div style={{ background: C.surface, borderRadius: 4, height: 8, overflow: "hidden", border: `1px solid ${C.border}` }}>
                  <div
  style={{
    width: `${
      (
        results.helmetCount
        / Math.max(results.totalDetections, 1)
      ) * 100
    }%`,
    height: "100%",
    background: C.green
  }}
/>
                </div>
              </div>

              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: "16px 18px" }}>
                <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 12 }}>Detection Summary</div>
                {[
                  { label: "Total detections", value: results.totalDetections },
                  { label: "Frames analyzed",  value: results.framesAnalyzed  },
                  { label: "Processing time",  value: results.processingTime  },
                  { label: "Model",            value: "YOLOv8"               },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${C.border}`, fontSize: 13 }}>
                    <span style={{ color: C.muted }}>{label}</span>
                    <span style={{ fontWeight: 600 }}>{value}</span>
                  </div>
                ))}
              </div>

              {/* API 串接提示 */}
              <div style={{ background: "#fffbe6", border: "1px dashed #d4aa00", borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 12, color: "#7a6000", display: "flex", alignItems: "flex-start", gap: 7 }}>
                  <i className="ti ti-code" style={{ fontSize: 13, marginTop: 1, flexShrink: 0 }} />
                  <span>
                    <strong>TODO for 同學：</strong> 找到 <code>runDetection()</code> 函式，把 mock 換成 <code>fetch()</code> 指向你的 YOLOv8 後端即可
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

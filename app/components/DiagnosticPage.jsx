/* Phyto-Diagnostic Suite */

const LeafUpload = ({ imgUrl, fileName, fileSize, onFile, onClear }) => {
  const inputRef = React.useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    onFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="tool-block">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Plant Specimen</h3>
        <span className={"pill " + (imgUrl ? "live" : "")}>
          {imgUrl ? "PhytoNet · ready" : "Awaiting upload"}
        </span>
      </div>
      <p className="tool-block-sub">Upload a close-up of a single leaf. Avoid multiple species in frame.</p>

      <input ref={inputRef} type="file" accept="image/*" style={{display:"none"}}
        onChange={e => onFile(e.target.files[0])} />

      <div className="upload-area">
        {!imgUrl ? (
          <div className="upload-drop"
            onClick={() => inputRef.current.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}>
            <Icon name="upload" size={24} />
            <div className="upload-drop-main">Drop leaf image · or click to browse</div>
            <div className="upload-drop-sub">PNG, JPG up to 10MB</div>
          </div>
        ) : (
          <div className="upload-preview">
            <div className="upload-preview-img leaf-square">
              <img src={imgUrl} alt="Leaf specimen" loading="eager" decoding="sync"
                style={{width:"100%",height:"100%",objectFit:"contain",display:"block",imageRendering:"auto"}} />
              <div className="upload-preview-chip">
                <Icon name="check" size={12} />
                <span>Valid leaf detected · ready for diagnosis</span>
              </div>
            </div>
            <div className="upload-meta">
              <div className="upload-meta-file">
                <div className="upload-meta-thumb leaf"></div>
                <div>
                  <div className="upload-meta-name">{fileName}</div>
                  <div className="upload-meta-sub num">{fileSize}</div>
                </div>
              </div>
              <button className="topbar-icon" onClick={onClear}><Icon name="x" size={13} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const DetectionResult = () => {
  return (
    <div className="detect-result">
      <div className="detect-result-head">
        <span className="eyebrow">Detection result</span>
        <span className="pill warn">Pathogen detected</span>
      </div>
      <h3 className="display detect-name">Apple Scab</h3>
      <div className="detect-latin">Venturia inaequalis · fungal</div>
      <div className="detect-confidence">
        <div className="detect-confidence-head">
          <span className="label">Confidence level</span>
          <span className="num detect-confidence-value">96.8%</span>
        </div>
        <div className="range-bar">
          <div className="fill" style={{ width: "96.8%" }} />
        </div>
      </div>
    </div>
  );
};

const TreatmentPlan = () => {
  return (
    <div className="treatment">
      <div className="treatment-head">
        <Icon name="flask" size={14} />
        <span className="eyebrow">Treatment plan</span>
      </div>
      <div className="treatment-item">
        <div className="treatment-item-head">
          <span className="treatment-num">01</span>
          <span className="treatment-label">Primary treatment</span>
        </div>
        <p className="treatment-body">Apply Mancozeb 75% WP @ 2.5 g/L, or Captan 50% WP @ 2 g/L at 10-day intervals.</p>
      </div>
      <div className="treatment-item">
        <div className="treatment-item-head">
          <span className="treatment-num">02</span>
          <span className="treatment-label">Fertilizer adjustment</span>
        </div>
        <p className="treatment-body">Reduce excess nitrogen; apply balanced NPK 12:32:16 @ 8 g/L. Supplement Calcium (Ca(NO₃)₂) @ 3 g/L to strengthen cell walls.</p>
      </div>
      <div className="treatment-item">
        <div className="treatment-item-head">
          <span className="treatment-num">03</span>
          <span className="treatment-label">Cultural practices</span>
        </div>
        <p className="treatment-body">Prune infected twigs; rake fallen leaves; ensure canopy airflow to reduce humidity.</p>
      </div>
    </div>
  );
};

const TopPredictions = () => {
  const preds = [
    { c: "Apple — Apple scab", v: 96.8, accent: true },
    { c: "Apple — Black rot", v: 1.4 },
    { c: "Apple — Cedar apple rust", v: 0.9 },
    { c: "Apple — healthy", v: 0.6 },
    { c: "Blueberry — healthy", v: 0.3 },
  ];
  return (
    <div className="preds">
      <div className="preds-head">
        <span className="eyebrow">Top-5 predictions</span>
        <span className="label num">Softmax</span>
      </div>
      {preds.map((p, i) => (
        <div key={i} className="pred-row">
          <div className="pred-label">{p.c}</div>
          <div className="pred-bar">
            <div className="pred-bar-fill" style={{ width: p.v + "%", background: p.accent ? "var(--sage)" : "var(--ink-4)" }} />
          </div>
          <div className="pred-val num">{p.v.toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
};

const DiagnosticPage = () => {
  const [imgUrl, setImgUrl] = React.useState(null);
  const [fileName, setFileName] = React.useState("");
  const [fileSize, setFileSize] = React.useState("");
  const [running, setRunning] = React.useState(false);
  const [done, setDone] = React.useState(false);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    if (imgUrl) URL.revokeObjectURL(imgUrl);
    setImgUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setFileSize((file.size / 1024).toFixed(1) + " KB");
    setDone(false);
  };

  const clearFile = () => {
    if (imgUrl) URL.revokeObjectURL(imgUrl);
    setImgUrl(null); setFileName(""); setFileSize(""); setDone(false);
  };

  const runDiagnosis = () => {
    if (!imgUrl || running) return;
    setRunning(true); setDone(false);
    setTimeout(() => { setRunning(false); setDone(true); }, 1800);
  };

  return (
    <div className="page-tool" data-screen-label="03 Phyto-Diagnostic Suite">
      <div className="tool-header">
        <div>
          <Reveal><span className="eyebrow">Module · Neural Vision</span></Reveal>
          <Reveal delay={60}>
            <h1 className="display tool-page-title">Phyto-Diagnostic Suite</h1>
          </Reveal>
          <Reveal delay={120}>
            <p className="tool-page-sub">
              Upload a leaf photograph. PhytoNet-v2 resolves pathogen identity across 38 classes
              and returns a precision treatment protocol in a single forward pass.
            </p>
          </Reveal>
        </div>
      </div>

      <div className="diag-grid">
        <Reveal>
          <LeafUpload imgUrl={imgUrl} fileName={fileName} fileSize={fileSize}
            onFile={handleFile} onClear={clearFile} />
        </Reveal>
        <Reveal delay={80}>
          <div className="diag-right">
            {done ? (
              <>
                <DetectionResult />
                <TreatmentPlan />
                <TopPredictions />
              </>
            ) : (
              <div className="detect-result">
                <div className="detect-result-head">
                  <span className="eyebrow">Detection result</span>
                  <span className="pill">{running ? "Running…" : "Awaiting specimen"}</span>
                </div>
                <p style={{fontSize:"13px",color:"var(--ink-3)",marginTop:"12px"}}>
                  {imgUrl
                    ? (running ? "PhytoNet-v2 inference in progress…" : "Preview ready · run diagnosis to see results here.")
                    : "Upload a leaf image and run diagnosis to see results here."}
                </p>
                {running && <div className="tool-analyze-bar" style={{marginTop:"20px",borderRadius:"999px",overflow:"hidden"}}><div /></div>}
              </div>
            )}
          </div>
        </Reveal>
      </div>

      <div className="diag-actions">
        <button className={"btn btn-large " + (done ? "btn-ghost" : "btn-sage") + (running ? " analyzing" : "")}
          onClick={runDiagnosis} disabled={!imgUrl || running}>
          {running
            ? <><span className="spinner" /> Running PhytoNet-v2…</>
            : <><Icon name="microscope" size={15} /> Run neural diagnosis <Icon name="arrowRight" size={14} className="arrow" /></>}
        </button>
        {done && (
          <button className="btn btn-ghost">
            <Icon name="download" size={14} /> Export report
          </button>
        )}
      </div>
    </div>
  );
};

window.DiagnosticPage = DiagnosticPage;

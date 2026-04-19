/* Phyto-Diagnostic Suite */

const LeafUpload = () => {
  return (
    <div className="tool-block">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Plant Specimen</h3>
        <span className="pill live">PhytoNet · ready</span>
      </div>
      <p className="tool-block-sub">Upload a close-up of a single leaf. Avoid multiple species in frame.</p>

      <div className="upload-area">
        <div className="upload-preview">
          <div className="upload-preview-img leaf">
            <svg viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice">
              <defs>
                <radialGradient id="leafG" cx="0.5" cy="0.5" r="0.7">
                  <stop offset="0%" stopColor="#5a8a3a" />
                  <stop offset="60%" stopColor="#2d4a2b" />
                  <stop offset="100%" stopColor="#14240f" />
                </radialGradient>
                <filter id="leafTex">
                  <feTurbulence baseFrequency="0.4" numOctaves="3" seed="2" />
                  <feColorMatrix values="0 0 0 0 0.2  0 0 0 0 0.3  0 0 0 0 0.15  0 0 0 0.4 0" />
                  <feComposite in2="SourceGraphic" operator="in" />
                </filter>
              </defs>
              <rect width="400" height="300" fill="url(#leafG)" />
              <ellipse cx="200" cy="150" rx="140" ry="95" fill="#4a7a2a" />
              <ellipse cx="200" cy="150" rx="140" ry="95" fill="url(#leafTex)" />
              <path d="M 60 150 Q 200 140 340 150" stroke="#2a4a1a" strokeWidth="1" fill="none" />
              {Array.from({length: 8}).map((_, i) => (
                <path key={i} d={`M 200 150 Q ${120 + i * 25} ${100 + (i%2)*100} ${80 + i * 35} ${110 + (i%2)*80}`}
                  stroke="#2a4a1a" strokeWidth="0.8" fill="none" opacity="0.6"/>
              ))}
              {/* Disease spots */}
              {[[140,120,8],[220,160,10],[170,180,6],[260,130,7],[190,210,5]].map(([x,y,r], i) => (
                <g key={i}>
                  <circle cx={x} cy={y} r={r} fill="#3a2014" opacity="0.8"/>
                  <circle cx={x} cy={y} r={r-2} fill="#5a3a20" opacity="0.6"/>
                </g>
              ))}
              {/* Detection annotation */}
              <g>
                <rect x="120" y="95" width="60" height="40" fill="none" stroke="#e8c989" strokeWidth="1.2" strokeDasharray="3 3"/>
                <text x="184" y="110" fill="#e8c989" fontSize="9" fontFamily="monospace">Lesion · 0.91</text>
                <rect x="215" y="150" width="55" height="35" fill="none" stroke="#e8c989" strokeWidth="1.2" strokeDasharray="3 3"/>
                <text x="275" y="165" fill="#e8c989" fontSize="9" fontFamily="monospace">Lesion · 0.88</text>
              </g>
            </svg>
            <div className="upload-preview-chip">
              <Icon name="check" size={12} />
              <span>Valid leaf detected · ready for diagnosis</span>
            </div>
          </div>
          <div className="upload-meta">
            <div className="upload-meta-file">
              <div className="upload-meta-thumb leaf"></div>
              <div>
                <div className="upload-meta-name">apple_leaf_sample.jpg</div>
                <div className="upload-meta-sub num">1.8 MB · 2048×1536</div>
              </div>
            </div>
            <button className="topbar-icon"><Icon name="x" size={13} /></button>
          </div>
        </div>
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
  return (
    <div className="page-tool" data-screen-label="03 Phyto-Diagnostic Suite">
      <div className="tool-header">
        <div>
          <Reveal><span className="eyebrow">Module · Neural Vision</span></Reveal>
          <Reveal delay={80}>
            <h1 className="display tool-page-title">Phyto-Diagnostic Suite</h1>
          </Reveal>
          <Reveal delay={160}>
            <p className="tool-page-sub">
              Upload a leaf photograph. PhytoNet-v2 resolves pathogen identity across 38 classes
              and returns a precision treatment protocol in a single forward pass.
            </p>
          </Reveal>
        </div>
      </div>

      <div className="diag-grid">
        <Reveal><LeafUpload /></Reveal>
        <Reveal delay={100}>
          <div className="diag-right">
            <DetectionResult />
            <TreatmentPlan />
            <TopPredictions />
          </div>
        </Reveal>
      </div>

      <div className="diag-actions">
        <button className="btn btn-sage btn-large">
          <Icon name="microscope" size={15} /> Run neural diagnosis <Icon name="arrowRight" size={14} className="arrow" />
        </button>
        <button className="btn btn-ghost">
          <Icon name="download" size={14} /> Export report
        </button>
      </div>
    </div>
  );
};

window.DiagnosticPage = DiagnosticPage;

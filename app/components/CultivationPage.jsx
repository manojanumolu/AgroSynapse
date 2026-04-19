/* Predictive Cultivation — the main tool */

const SoilUpload = () => {
  const [imgUrl, setImgUrl] = React.useState(null);
  const [fileName, setFileName] = React.useState("");
  const [fileSize, setFileSize] = React.useState("");
  const inputRef = React.useRef(null);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    if (imgUrl) URL.revokeObjectURL(imgUrl);
    setImgUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setFileSize((file.size / 1024).toFixed(0) + " KB");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const clear = () => {
    if (imgUrl) URL.revokeObjectURL(imgUrl);
    setImgUrl(null); setFileName(""); setFileSize("");
  };

  return (
    <div className="tool-block">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Soil Specimen</h3>
        <span className={"pill " + (imgUrl ? "live" : "")}>
          {imgUrl ? "Vision ready" : "Awaiting upload"}
        </span>
      </div>
      <p className="tool-block-sub">Upload a clear close-up of the soil sample. Avoid leaves, hands, or moisture artifacts.</p>

      <input ref={inputRef} type="file" accept="image/*" style={{display:"none"}}
        onChange={e => handleFile(e.target.files[0])} />

      <div className="upload-area">
        {!imgUrl ? (
          <div className="upload-drop"
            onClick={() => inputRef.current.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}>
            <Icon name="upload" size={24} />
            <div className="upload-drop-main">Drop image · or click to browse</div>
            <div className="upload-drop-sub">PNG, JPG up to 10MB</div>
          </div>
        ) : (
          <div className="upload-preview">
            <div className="upload-preview-img">
              <img src={imgUrl} alt="Soil specimen" />
              <div className="upload-preview-chip">
                <Icon name="check" size={12} />
                <span>Valid specimen · ready for synthesis</span>
              </div>
            </div>
            <div className="upload-meta">
              <div className="upload-meta-file">
                <div className="upload-meta-thumb"></div>
                <div>
                  <div className="upload-meta-name">{fileName}</div>
                  <div className="upload-meta-sub num">{fileSize}</div>
                </div>
              </div>
              <button className="topbar-icon" title="Remove" onClick={clear}><Icon name="x" size={13} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Tooltip = ({ text }) => (
  <div className="tooltip-wrap">
    <Icon name="info" size={12} />
    <div className="tooltip-box">{text}</div>
  </div>
);

const NumericField = ({ label, unit, value, setValue, range, step = 0.1, optimal, tooltip }) => {
  const pct = Math.max(0, Math.min(100, ((value - range[0]) / (range[1] - range[0])) * 100));
  const optLeft = optimal ? ((optimal[0] - range[0]) / (range[1] - range[0])) * 100 : 0;
  const optWidth = optimal ? ((optimal[1] - optimal[0]) / (range[1] - range[0])) * 100 : 0;
  const inOptimal = optimal && value >= optimal[0] && value <= optimal[1];
  return (
    <div className="numeric-field">
      <div className="numeric-field-head">
        <div className="numeric-field-label-row">
          <span className="label">{label}</span>
          {tooltip && <Tooltip text={tooltip} />}
        </div>
        <span className="label numeric-unit">{unit}</span>
      </div>
      <div className="stepper">
        <input type="number" value={value} step={step}
          onChange={(e) => setValue(parseFloat(e.target.value) || 0)} />
        <button onClick={() => setValue(+(value - step).toFixed(2))}>−</button>
        <button onClick={() => setValue(+(value + step).toFixed(2))}>+</button>
      </div>
      <div className="range-bar">
        {optimal && <div className="opt" style={{ left: optLeft + "%", width: optWidth + "%" }} />}
        <div className="fill" style={{ width: pct + "%", background: inOptimal ? "var(--sage)" : "var(--earth-2)" }} />
      </div>
      {optimal && (
        <div className="numeric-hint num">
          {inOptimal ? <span className="ok">● Optimal · {optimal[0]}–{optimal[1]}</span>
                     : <span className="warn">○ Outside optimal · {optimal[0]}–{optimal[1]}</span>}
        </div>
      )}
    </div>
  );
};

const ChemicalProfile = () => {
  const [n, setN] = React.useState(90);
  const [p, setP] = React.useState(35);
  const [k, setK] = React.useState(54);
  const [ph, setPh] = React.useState(6.5);
  return (
    <div className="tool-block">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Chemical Profile</h3>
        <span className="pill">NPK · pH</span>
      </div>
      <p className="tool-block-sub">Known values from lab report, or estimates from field tests.</p>
      <div className="chem-grid">
        <NumericField label="Nitrogen (N)" unit="mg/kg" value={n} setValue={setN} range={[0, 200]} step={1} optimal={[60, 140]}
          tooltip="Available nitrogen in soil. Low N causes yellowing; high N causes excessive leaf growth. Optimal: 60–140 mg/kg." />
        <NumericField label="Phosphorus (P)" unit="mg/kg" value={p} setValue={setP} range={[0, 100]} step={1} optimal={[20, 60]}
          tooltip="Supports root development and flowering. Deficiency shows as purple leaf undersides. Optimal: 20–60 mg/kg." />
        <NumericField label="Potassium (K)" unit="mg/kg" value={k} setValue={setK} range={[0, 200]} step={1} optimal={[30, 120]}
          tooltip="Regulates water uptake and disease resistance. Low K causes leaf-edge browning. Optimal: 30–120 mg/kg." />
        <NumericField label="Soil pH" unit="pH" value={ph} setValue={setPh} range={[3, 10]} step={0.1} optimal={[6, 7.2]}
          tooltip="Acidity/alkalinity scale. Most crops thrive at 6.0–7.2. Outside this range, nutrients become unavailable regardless of quantity." />
      </div>
    </div>
  );
};

const ClimateBlock = () => {
  const [state, setState] = React.useState("Andhra Pradesh");
  const [district, setDistrict] = React.useState("Guntur");
  const [village, setVillage] = React.useState("Rawada");
  const [filled, setFilled] = React.useState(true);

  return (
    <div className="tool-block wide">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Climate Synthesis</h3>
        <span className={"pill " + (filled ? "live" : "")}>{filled ? "Auto-filled" : "Pending"}</span>
      </div>
      <p className="tool-block-sub">District-grade vectors pulled from 12-year IMD historical series.</p>

      <div className="climate-row">
        <div className="field">
          <span className="label">State</span>
          <select value={state} onChange={(e) => setState(e.target.value)}>
            <option>Andhra Pradesh</option><option>Telangana</option><option>Karnataka</option><option>Tamil Nadu</option>
          </select>
        </div>
        <div className="field">
          <span className="label">District</span>
          <select value={district} onChange={(e) => setDistrict(e.target.value)}>
            <option>Guntur</option><option>Krishna</option><option>Nellore</option>
          </select>
        </div>
        <div className="field">
          <span className="label">Village / Town</span>
          <input type="text" value={village} onChange={(e) => setVillage(e.target.value)} />
        </div>
        <button className="btn btn-sage" onClick={() => setFilled(true)}>
          <Icon name="sparkles" size={13} /> Fetch vectors
        </button>
      </div>

      <div className="climate-tiles">
        <div className="climate-tile">
          <div className="climate-tile-icon"><Icon name="thermometer" size={16} /></div>
          <div className="climate-tile-body">
            <div className="label">Temperature</div>
            <div className="climate-tile-value">27.2<small>°C</small></div>
          </div>
          <div className="climate-tile-spark">
            <svg viewBox="0 0 60 20"><polyline points="0,14 10,10 20,12 30,6 40,8 50,4 60,7" fill="none" stroke="#c44536" strokeWidth="1.5"/></svg>
          </div>
        </div>
        <div className="climate-tile">
          <div className="climate-tile-icon"><Icon name="droplet" size={16} /></div>
          <div className="climate-tile-body">
            <div className="label">Humidity</div>
            <div className="climate-tile-value">75.3<small>%</small></div>
          </div>
          <div className="climate-tile-spark">
            <svg viewBox="0 0 60 20"><polyline points="0,8 10,12 20,10 30,14 40,9 50,12 60,10" fill="none" stroke="#5a8a3a" strokeWidth="1.5"/></svg>
          </div>
        </div>
        <div className="climate-tile">
          <div className="climate-tile-icon"><Icon name="wind" size={16} /></div>
          <div className="climate-tile-body">
            <div className="label">Rainfall</div>
            <div className="climate-tile-value">1,302<small>mm</small></div>
          </div>
          <div className="climate-tile-spark">
            <svg viewBox="0 0 60 20"><polyline points="0,16 10,12 20,14 30,8 40,10 50,5 60,9" fill="none" stroke="#d4a373" strokeWidth="1.5"/></svg>
          </div>
        </div>
      </div>
    </div>
  );
};

const FarmDetails = () => {
  const [yieldLast, setYieldLast] = React.useState(2083);
  const [fert, setFert] = React.useState(118);
  const [season, setSeason] = React.useState("Kharif (Monsoon)");
  const [irrigation, setIrrigation] = React.useState("Canal Irrigation");
  const [prevCrop, setPrevCrop] = React.useState("Cotton");
  const [region, setRegion] = React.useState("Central Zone");

  return (
    <div className="tool-block">
      <div className="tool-block-head">
        <h3 className="display tool-block-title">Farm Context</h3>
        <span className="pill earth">History · systems</span>
      </div>
      <p className="tool-block-sub">Historical yield + cultivation system. Used to weight output probability.</p>

      <div className="farm-grid">
        <NumericField label="Yield · last season" unit="kg/ha" value={yieldLast} setValue={setYieldLast} range={[0, 5000]} step={10}
          tooltip="Your previous crop's yield in kilograms per hectare. Used to calibrate the model's baseline. 1 acre ≈ 0.4 ha." />
        <NumericField label="Fertilizer used" unit="kg/ha" value={fert} setValue={setFert} range={[0, 400]} step={1}
          tooltip="Total fertilizer applied last season. Helps model assess soil depletion vs excess. Typical range: 60–200 kg/ha." />

        <div className="field">
          <span className="label">Current season</span>
          <select value={season} onChange={(e) => setSeason(e.target.value)}>
            <option>Kharif (Monsoon)</option><option>Rabi (Winter)</option><option>Zaid (Summer)</option>
          </select>
        </div>
        <div className="field">
          <span className="label">Irrigation system</span>
          <select value={irrigation} onChange={(e) => setIrrigation(e.target.value)}>
            <option>Canal Irrigation</option><option>Borewell</option><option>Drip</option><option>Rainfed</option>
          </select>
        </div>
        <div className="field">
          <span className="label">Previous crop</span>
          <select value={prevCrop} onChange={(e) => setPrevCrop(e.target.value)}>
            <option>Cotton</option><option>Rice</option><option>Sugarcane</option><option>Groundnut</option>
          </select>
        </div>
        <div className="field">
          <span className="label">Geographic zone</span>
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            <option>Central Zone</option><option>Coastal</option><option>Arid</option>
          </select>
        </div>
      </div>
    </div>
  );
};

const UnitGuide = () => {
  return (
    <aside className="unit-guide">
      <div className="unit-guide-head">
        <Icon name="info" size={14} />
        <span className="eyebrow">Farmer unit guide</span>
      </div>
      <div className="unit-guide-body">
        <div className="ug-row"><span>Yield</span><span className="num">t/ha</span></div>
        <div className="ug-row"><span>NPK</span><span className="num">mg/kg</span></div>
        <div className="ug-row"><span>Area</span><span className="num">1 acre ≈ 0.4 ha</span></div>
        <div className="ug-row"><span>Temp</span><span className="num">°C</span></div>
        <div className="ug-row"><span>Rainfall</span><span className="num">mm/yr</span></div>
      </div>
      <div className="unit-guide-foot">
        <span className="label">Need help?</span>
        <a className="unit-guide-link">Open glossary →</a>
      </div>
    </aside>
  );
};

const CultivationPage = ({ setPage }) => {
  const [analyzing, setAnalyzing] = React.useState(false);

  const runAnalysis = () => {
    setAnalyzing(true);
    setTimeout(() => {
      setAnalyzing(false);
      setPage("dashboard");
    }, 2400);
  };

  return (
    <div className="page-tool" data-screen-label="02 Predictive Cultivation">
      <div className="tool-header">
        <div>
          <Reveal><span className="eyebrow">Module · Agricultural Core</span></Reveal>
          <Reveal delay={80}>
            <h1 className="display tool-page-title">Predictive Cultivation</h1>
          </Reveal>
          <Reveal delay={160}>
            <p className="tool-page-sub">
              Synthesize soil specimen, chemical profile, climate vectors, and farm history into
              laboratory-grade crop recommendations — probability-ranked, protocol-complete.
            </p>
          </Reveal>
        </div>
        <Reveal delay={240}>
          <div className="tool-header-actions">
            <UnitGuide />
          </div>
        </Reveal>
      </div>

      <div className="tool-grid">
        <Reveal><SoilUpload /></Reveal>
        <Reveal delay={80}><ChemicalProfile /></Reveal>
        <Reveal delay={120}><ClimateBlock /></Reveal>
        <Reveal delay={160}><FarmDetails /></Reveal>
      </div>

      <div className="tool-analyze">
        <div className="tool-analyze-inner">
          <div>
            <h3 className="display tool-analyze-title">Ready to synthesize.</h3>
            <p className="tool-analyze-sub">All four vectors complete. Expected synthesis time: ~2.4 seconds.</p>
          </div>
          <button className={"btn btn-primary btn-large " + (analyzing ? "analyzing" : "")} onClick={runAnalysis} disabled={analyzing}>
            {analyzing ? (
              <><span className="spinner" /> Synthesizing…</>
            ) : (
              <><Icon name="sparkles" size={15} /> Analyze & predict crop <Icon name="arrowRight" size={14} className="arrow" /></>
            )}
          </button>
        </div>
        {analyzing && <div className="tool-analyze-bar"><div /></div>}
      </div>
    </div>
  );
};

window.CultivationPage = CultivationPage;

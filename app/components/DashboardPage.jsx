/* Dashboard / Results page */

const RingProgress = ({ value, size = 140, stroke = 6 }) => {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const [offset, setOffset] = React.useState(c);
  const ref = React.useRef(null);
  React.useEffect(() => {
    const el = ref.current;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) setOffset(c * (1 - value / 100)); });
    }, { threshold: 0.3 });
    if (el) io.observe(el);
    return () => io.disconnect();
  }, [value, c]);
  return (
    <div ref={ref} className="progress-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle className="track" cx={size/2} cy={size/2} r={r} strokeWidth={stroke} fill="none" />
        <circle className="fill" cx={size/2} cy={size/2} r={r} strokeWidth={stroke} fill="none"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round" />
      </svg>
      <div className="center"><span className="num">{value}<small style={{fontSize:'14px'}}>%</small></span></div>
    </div>
  );
};

const RecommendedCrop = () => {
  return (
    <div className="rec-crop">
      <div className="rec-crop-img">
        <svg viewBox="0 0 300 300" preserveAspectRatio="xMidYMid slice">
          <defs>
            <radialGradient id="claygrad" cx="0.5" cy="0.5" r="0.7">
              <stop offset="0%" stopColor="#d4533f"/><stop offset="100%" stopColor="#8a2a1a"/>
            </radialGradient>
          </defs>
          <rect width="300" height="300" fill="url(#claygrad)"/>
          {Array.from({length:6}).map((_,i)=>(
            <ellipse key={i}
              cx={100 + (i%3)*50 + (i>2?25:0)}
              cy={160 + Math.floor(i/3)*50}
              rx="22" ry="16" fill="#d4a373" stroke="#8b6f47" strokeWidth="1.5"/>
          ))}
          <text x="20" y="280" fill="rgba(255,255,255,0.4)" fontSize="10" fontFamily="monospace">GROUNDNUT · ARACHIS HYPOGAEA</text>
        </svg>
      </div>
      <div className="rec-crop-body">
        <span className="eyebrow">Primary recommendation</span>
        <h2 className="display rec-crop-name">Groundnut</h2>
        <p className="rec-crop-desc">
          Synaptic triangulation indicates Groundnut as the optimal rotation for
          the upcoming Kharif season. Soil profile aligns with 14,200 validated training pairs.
        </p>
        <div className="rec-crop-divider" />
        <div className="rec-crop-meta">
          <div><span className="label">NPK protocol</span><span className="num rec-crop-val">18:46:32</span></div>
          <div><span className="label">Expected yield</span><span className="num rec-crop-val">3,140 kg/ha</span></div>
          <div><span className="label">Confidence</span><span className="num rec-crop-val">93%</span></div>
        </div>
      </div>
      <div className="rec-crop-ring">
        <RingProgress value={93} size={140} stroke={6} />
        <span className="label">Synaptic score</span>
      </div>
    </div>
  );
};

const AltCrops = () => {
  const alts = [
    { name: "Maize", score: 68, n: "18:46:32", y: "2,890 kg/ha", tag: "Safe alt" },
    { name: "Chilli", score: 54, n: "19:19:19", y: "1,420 kg/ha", tag: "Cash crop" },
    { name: "Sorghum", score: 47, n: "12:32:16", y: "2,100 kg/ha", tag: "Drought-safe" },
  ];
  return (
    <div className="alts">
      <div className="alts-head">
        <span className="eyebrow">Alternative crops</span>
        <span className="label">Ranked K=2–4</span>
      </div>
      {alts.map((a, i) => (
        <div key={i} className="alt">
          <div className="alt-rank num">{String(i+2).padStart(2,"0")}</div>
          <div className="alt-body">
            <div className="alt-name">{a.name}</div>
            <div className="alt-meta"><span className="num">{a.n}</span> · <span className="num">{a.y}</span></div>
          </div>
          <div className="alt-tag">{a.tag}</div>
          <div className="alt-score num">{a.score}%</div>
        </div>
      ))}
    </div>
  );
};

const ProbabilityBreakdown = () => {
  const bars = [
    { k: "Groundnut", v: 93, color: "var(--sage)" },
    { k: "Maize", v: 68, color: "var(--earth)" },
    { k: "Chilli", v: 54, color: "var(--earth-2)" },
    { k: "Sorghum", v: 47, color: "var(--ink-4)" },
    { k: "Cotton", v: 38, color: "var(--ink-4)" },
    { k: "Paddy", v: 22, color: "var(--ink-4)" },
    { k: "Sugarcane", v: 12, color: "var(--ink-4)" },
  ];
  return (
    <div className="probs">
      <div className="probs-head">
        <div>
          <span className="eyebrow">Soil probability breakdown</span>
          <h3 className="display probs-title">Synaptic confidence, per crop.</h3>
        </div>
        <div className="seg">
          <button className="active">Soil</button><button>Climate</button><button>Combined</button>
        </div>
      </div>
      <div className="probs-chart">
        {bars.map((b, i) => (
          <div key={i} className="prob-row">
            <div className="prob-k">{b.k}</div>
            <div className="prob-track">
              <div className="prob-fill" style={{ width: b.v + "%", background: b.color, animationDelay: (i * 80) + "ms" }} />
            </div>
            <div className="prob-v num">{b.v}%</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const Timeline = () => {
  const months = ["Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"];
  const rows = [
    { phase: "Sowing", span: [0, 1], color: "var(--sage)" },
    { phase: "Vegetative", span: [0.8, 3], color: "var(--sage-2)" },
    { phase: "Flowering", span: [3, 4.5], color: "var(--earth)" },
    { phase: "Pod development", span: [4, 6], color: "var(--earth-2)" },
    { phase: "Harvest", span: [6, 7.5], color: "var(--clay)" },
  ];
  return (
    <div className="timeline">
      <div className="timeline-head">
        <span className="eyebrow">Cultivation timeline · 8 months</span>
        <span className="pill earth">Kharif 2026</span>
      </div>
      <div className="timeline-months">
        {months.map(m => <div key={m} className="timeline-m label">{m}</div>)}
      </div>
      <div className="timeline-body">
        {rows.map((r, i) => (
          <div key={i} className="timeline-row">
            <div className="timeline-phase">{r.phase}</div>
            <div className="timeline-track">
              <div className="timeline-bar" style={{
                left: `${(r.span[0] / 8) * 100}%`,
                width: `${((r.span[1] - r.span[0]) / 8) * 100}%`,
                background: r.color,
              }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const DashboardPage = () => {
  return (
    <div className="page-dashboard" data-screen-label="04 Dashboard">
      <div className="dash-header">
        <div>
          <Reveal><span className="eyebrow">Result analysis · field T-047</span></Reveal>
          <Reveal delay={80}><h1 className="display tool-page-title">Synthesis complete.</h1></Reveal>
          <Reveal delay={160}>
            <p className="tool-page-sub">
              Generated Apr 18, 2026 · 14:22 IST. Output verified across 4 model heads;
              confidence-weighted with historical yield prior.
            </p>
          </Reveal>
        </div>
        <Reveal delay={200}>
          <div className="dash-header-actions">
            <button className="btn btn-ghost"><Icon name="download" size={14} /> Export PDF</button>
            <button className="btn btn-primary"><Icon name="arrowUpRight" size={14} /> Share report</button>
          </div>
        </Reveal>
      </div>

      <Reveal delay={100}><RecommendedCrop /></Reveal>

      <div className="dash-grid">
        <Reveal delay={80} className="dash-col-wide"><ProbabilityBreakdown /></Reveal>
        <Reveal delay={160}><AltCrops /></Reveal>
      </div>

      <Reveal delay={200}><Timeline /></Reveal>

      <div className="dash-insights">
        <Reveal><div className="section-head">
          <span className="eyebrow">Advisory · notes</span>
          <h2 className="display section-title">What the model <em>noticed.</em></h2>
        </div></Reveal>
        <div className="insights-grid">
          {[
            { k: "Soil moisture elevated", v: "72.4% · 9pts above optimal. Reduce next irrigation cycle.", icon: "droplet" },
            { k: "Nitrogen balanced", v: "90 mg/kg. Within optimal band; maintain current dosage.", icon: "flask" },
            { k: "Trend: climate", v: "Rainfall +14% YoY for this zone. Groundnut outperforms maize under wetter monsoons.", icon: "wind" },
            { k: "Alert: fungal window", v: "Humidity + temperature align with phyto-stress curve in weeks 6–8.", icon: "microscope" },
          ].map((it, i) => (
            <Reveal key={i} delay={i * 100}>
              <div className="insight">
                <div className="insight-icon"><Icon name={it.icon} size={16} /></div>
                <div>
                  <div className="insight-k">{it.k}</div>
                  <p className="insight-v">{it.v}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </div>
  );
};

window.DashboardPage = DashboardPage;
window.RingProgress = RingProgress;

/* Home / Landing page — editorial hero + module preview */

const useReveal = (delay = 0) => {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          setTimeout(() => el.classList.add("in"), delay);
          io.unobserve(el);
        }
      });
    }, { threshold: 0.15 });
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);
  return ref;
};

const Reveal = ({ children, delay = 0, className = "" }) => {
  const ref = useReveal(delay);
  return <div ref={ref} className={"reveal " + className}>{children}</div>;
};

const MaskLine = ({ children, delay = 0 }) => {
  const ref = useReveal(delay);
  return (
    <span ref={ref} className="reveal-mask"><span>{children}</span></span>
  );
};

const HeroVisual = () => {
  // Keep hero motion CSS-driven to avoid expensive full React rerenders every frame.
  const sway = (amp, period, phase = 0) => Math.sin(phase / period) * amp;

  return (
    <div className="hero-visual">
      <svg viewBox="0 0 600 600" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#f5e8cf" />
            <stop offset="40%" stopColor="#e8c989" />
            <stop offset="75%" stopColor="#c68f4c" />
            <stop offset="100%" stopColor="#5a3a1a" />
          </linearGradient>
          <radialGradient id="sun" cx="0.72" cy="0.38" r="0.3">
            <stop offset="0%" stopColor="#fff3d1" stopOpacity="1" />
            <stop offset="40%" stopColor="#fcd982" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#f5a94e" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="field" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#4a5a2a" />
            <stop offset="100%" stopColor="#1a2812" />
          </linearGradient>
          <linearGradient id="field2" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#8b6f3e" />
            <stop offset="100%" stopColor="#3a2812" />
          </linearGradient>
          <filter id="grain">
            <feTurbulence baseFrequency="0.9" numOctaves="2" />
            <feColorMatrix values="0 0 0 0 0.2  0 0 0 0 0.15  0 0 0 0 0.1  0 0 0 0.15 0"/>
            <feComposite in2="SourceGraphic" operator="in"/>
          </filter>
        </defs>

        {/* Sky */}
        <rect width="600" height="600" fill="url(#sky)" />
        {/* Sun */}
        <rect width="600" height="600" fill="url(#sun)" />

        {/* Distant ridge */}
        <path d={`M0 ${380 + sway(1, 4)} Q 150 ${360} 300 ${375} T 600 ${370} L 600 600 L 0 600 Z`} fill="#6a4a2a" opacity="0.7" />
        <path d={`M0 ${420} Q 200 ${400} 400 ${415} T 600 ${410} L 600 600 L 0 600 Z`} fill="#3e2a18" opacity="0.85" />

        {/* Field */}
        <path d={`M0 450 Q 300 435 600 450 L 600 600 L 0 600 Z`} fill="url(#field)" />
        <path d={`M0 490 Q 300 475 600 490 L 600 600 L 0 600 Z`} fill="url(#field2)" opacity="0.7" />

        {/* Wheat stalks — foreground, swaying */}
        {Array.from({ length: 22 }).map((_, i) => {
          const x = (i / 22) * 600 + sway(2, 2, i);
          const h = 120 + (i % 5) * 20;
          const bend = sway(3 + (i % 3), 2, i * 0.7);
          return (
            <g key={i} opacity={0.55 + (i % 3) * 0.15}>
              <path d={`M ${x} 600 Q ${x + bend} ${600 - h / 2} ${x + bend * 2} ${600 - h}`}
                stroke="#3a2812" strokeWidth="1.3" fill="none" />
              <ellipse cx={x + bend * 2} cy={600 - h} rx="3" ry="10"
                fill="#c89a5c" transform={`rotate(${bend * 2} ${x + bend * 2} ${600 - h})`} />
            </g>
          );
        })}

        {/* Foreground blur stalks */}
        {Array.from({ length: 8 }).map((_, i) => {
          const x = (i / 8) * 700 - 40 + sway(4, 2.5, i);
          const h = 200 + (i % 3) * 40;
          const bend = sway(6, 2, i);
          return (
            <g key={"fg" + i} opacity="0.8" filter="blur(2px)">
              <path d={`M ${x} 600 Q ${x + bend} ${600 - h / 2} ${x + bend * 1.5} ${600 - h}`}
                stroke="#2a1c0a" strokeWidth="4" fill="none" strokeLinecap="round" />
            </g>
          );
        })}

        {/* Grain overlay */}
        <rect width="600" height="600" filter="url(#grain)" opacity="0.4" />

        {/* Vignette */}
        <radialGradient id="vig" cx="0.5" cy="0.5" r="0.75">
          <stop offset="60%" stopColor="#000" stopOpacity="0" />
          <stop offset="100%" stopColor="#000" stopOpacity="0.5" />
        </radialGradient>
        <rect width="600" height="600" fill="url(#vig)" />
      </svg>
    </div>
  );
};

const LiveMetrics = () => {
  const metrics = [
    { label: "Fields analyzed", value: 24781, suffix: "", icon: "map" },
    { label: "Accuracy, phyto", value: 96.3, suffix: "%", decimals: 1, icon: "microscope" },
    { label: "Yield uplift, avg", value: 18.4, suffix: "%", decimals: 1, icon: "seedling" },
    { label: "Regions covered", value: 11, suffix: "", icon: "chart" },
  ];
  return (
    <div className="metrics-row">
      {metrics.map((m, i) => (
        <Reveal key={i} delay={i * 80}>
          <div className="metric">
            <div className="metric-label">
              <Icon name={m.icon} size={13} />
              <span>{m.label}</span>
            </div>
            <div className="metric-value">
              <Counter value={m.value} decimals={m.decimals || 0} />
              <span className="metric-suffix">{m.suffix}</span>
            </div>
          </div>
        </Reveal>
      ))}
    </div>
  );
};

const Counter = ({ value, decimals = 0, duration = 1600 }) => {
  const [n, setN] = React.useState(0);
  const ref = React.useRef(null);
  const started = React.useRef(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting && !started.current) {
          started.current = true;
          const start = performance.now();
          const tick = (now) => {
            const p = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            setN(value * eased);
            if (p < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      });
    }, { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, [value, duration]);
  const display = decimals ? n.toFixed(decimals) : Math.round(n).toLocaleString();
  return <span ref={ref} className="num">{display}</span>;
};

const ModuleCard = ({ eyebrow, title, description, accent, icon, onClick }) => {
  return (
    <div className={"module-card " + (accent === "earth" ? "earth" : "")} onClick={onClick}>
      <div className="module-card-header">
        <div className="module-icon">
          <Icon name={icon} size={22} />
        </div>
        <span className="eyebrow">{eyebrow}</span>
      </div>
      <h3 className="module-title display">{title}</h3>
      <p className="module-desc">{description}</p>
      <div className="module-cta">
        <span>Open module</span>
        <div className="module-cta-icon"><Icon name="arrowRight" size={14} /></div>
      </div>
    </div>
  );
};

const PipelineSection = ({ kind }) => {
  const soil = {
    eyebrow: "Pipeline · soil to crop",
    title: "From soil, a crop.",
    sub: "Four stages of multimodal synthesis converge into a probability-ranked cultivation protocol.",
    accent: "sage",
    steps: [
      { n: "01", t: "Specimen vision", d: "Upload soil imagery. Computer vision extracts texture, aggregation, color-moisture indices in under 400ms.", icon: "scan" },
      { n: "02", t: "Chemical profile", d: "NPK + pH triangulated against optimal bands. Outliers flagged; dosages computed.", icon: "flask" },
      { n: "03", t: "Climate synthesis", d: "Auto-fills 180+ district-grade weather, humidity, and rainfall vectors from geographic selection.", icon: "wind" },
      { n: "04", t: "Crop recommendation", d: "Multimodal fusion returns top-K crops with NPK protocol, timeline, and confidence ranking.", icon: "seedling" },
    ],
  };
  const leaf = {
    eyebrow: "Pipeline · leaf to cure",
    title: "From leaf, a cure.",
    sub: "A single forward pass resolves pathogen identity and returns a dosage-precise treatment plan.",
    accent: "earth",
    steps: [
      { n: "01", t: "Leaf specimen", d: "Upload a close-up of a single leaf. Auto-validation checks framing, focus, and species ambiguity.", icon: "upload" },
      { n: "02", t: "Neural pathology", d: "Convolutional network scans lesions across 38 pathogen classes and one healthy baseline.", icon: "microscope" },
      { n: "03", t: "Dosage synthesis", d: "Confidence-weighted treatment plan: primary chemistry, fertilizer correction, cultural practices.", icon: "flask" },
      { n: "04", t: "Action report", d: "Export-ready PDF with ranked predictions, intervals, and field-specific cultural recommendations.", icon: "download" },
    ],
  };
  const data = kind === "leaf" ? leaf : soil;

  return (
    <section className={"pipeline pipeline-" + data.accent}>
      <Reveal>
        <div className="section-head">
          <span className="eyebrow">{data.eyebrow}</span>
          <h2 className="display section-title">{data.title.split(",")[0]}, <em>{data.title.split(",")[1]}</em></h2>
          <p className="pipeline-sub">{data.sub}</p>
        </div>
      </Reveal>
      <div className="pipeline-grid">
        {data.steps.map((s, i) => (
          <Reveal key={i} delay={i * 100}>
            <div className="pipeline-step">
              <div className="pipeline-step-head">
                <div className="pipeline-step-icon"><Icon name={s.icon} size={17} /></div>
                <span className="pipeline-step-num">{s.n}</span>
              </div>
              <h4 className="pipeline-step-title">{s.t}</h4>
              <p className="pipeline-step-desc">{s.d}</p>
              {i < data.steps.length - 1 && <div className="pipeline-arrow"><Icon name="arrowRight" size={14} /></div>}
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
};

const WorkflowStrip = () => {
  const steps = [
    { n: "01", t: "Specimen", d: "Upload soil + leaf imagery" },
    { n: "02", t: "Climate", d: "Auto-fill district vectors" },
    { n: "03", t: "Synthesis", d: "Neural triangulation runs" },
    { n: "04", t: "Recommendation", d: "Top-K crops + protocols" },
  ];
  return (
    <section className="workflow">
      <Reveal>
        <div className="section-head">
          <span className="eyebrow">From field to recommendation</span>
          <h2 className="display section-title">A four-step synthesis, <em>end-to-end.</em></h2>
        </div>
      </Reveal>
      <div className="workflow-track">
        {steps.map((s, i) => (
          <Reveal key={i} delay={i * 150} className="workflow-step-wrap">
            <div className="workflow-step">
              <div className="workflow-num">{s.n}</div>
              <div className="workflow-body">
                <h5>{s.t}</h5>
                <p>{s.d}</p>
              </div>
            </div>
            {i < steps.length - 1 && <div className="workflow-line" />}
          </Reveal>
        ))}
      </div>
    </section>
  );
};

const HomePage = ({ setPage }) => {
  return (
    <div className="page-home" data-screen-label="01 Home">
      {/* HERO */}
      <section className="hero">
        <HeroVisual />
        <div className="hero-overlay" />
        <div className="hero-grid hero-grid-solo">
          <div className="hero-copy">
            <Reveal delay={60}>
              <div className="hero-brandname display">Agro<em>Synapse</em></div>
            </Reveal>
            <h1 className="display hero-title">
              <MaskLine delay={180}>Laboratory-grade</MaskLine>
              <MaskLine delay={300}>agronomy, delivered</MaskLine>
              <MaskLine delay={420}>to every <em>acre.</em></MaskLine>
            </h1>
            <Reveal delay={620}>
              <p className="hero-lede">
                Fusing soil vision, climate synthesis, and phyto-diagnostic
                neural nets into a single recommendation engine —
                tuned for your field, not the average of everyone else's.
              </p>
            </Reveal>
            <Reveal delay={760}>
              <div className="hero-actions">
                <button className="btn btn-primary" onClick={() => setPage("cultivation")}>
                  Begin synthesis <Icon name="arrowRight" size={14} className="arrow" />
                </button>
                <button className="btn btn-ghost" onClick={() => setPage("diagnostic")}>
                  <Icon name="microscope" size={13} /> Diagnose a leaf
                </button>
              </div>
            </Reveal>
            <Reveal delay={900}>
              <div className="hero-badges">
                <div className="hero-badge"><Icon name="seedling" size={13}/><span>Predictive Cultivation</span></div>
                <div className="hero-badge"><Icon name="microscope" size={13}/><span>Phyto-Diagnostic Suite</span></div>
              </div>
            </Reveal>
          </div>
        </div>
        <div className="hero-tick">
          <span className="label">Scroll</span>
          <Icon name="arrowDown" size={14} />
        </div>
      </section>

      {/* MODULES */}
      <section className="modules">
        <Reveal>
          <div className="section-head">
            <span className="eyebrow">Modules · active</span>
            <h2 className="display section-title">Two instruments, <em>calibrated</em> for your land.</h2>
          </div>
        </Reveal>
        <div className="modules-grid">
          <Reveal delay={120}>
            <ModuleCard
              eyebrow="Agricultural · core"
              title="Predictive Cultivation"
              description="Multimodal soil-to-crop fusion. Fuses specimen imagery, chemical profile, climate vectors, and farm history to return top-K crops with protocol."
              icon="seedling"
              onClick={() => setPage("cultivation")}
            />
          </Reveal>
          <Reveal delay={240}>
            <ModuleCard
              eyebrow="Neural · vision"
              title="Phyto-Diagnostic Suite"
              description="Leaf-to-cure vision. Convolutional pathology engine trained across 38 classes, resolving pathogen identity and treatment dosage in a single pass."
              icon="microscope"
              accent="earth"
              onClick={() => setPage("diagnostic")}
            />
          </Reveal>
        </div>
      </section>

      {/* SOIL PIPELINE */}
      <PipelineSection kind="soil" />

      {/* LEAF PIPELINE */}
      <PipelineSection kind="leaf" />

      {/* FOOTER CTA */}
      <section className="cta-block">
        <Reveal>
          <div className="cta-inner">
            <span className="eyebrow">Ready when you are</span>
            <h2 className="display cta-title">Put a <em>lab</em> behind every field.</h2>
            <p className="cta-sub">Specimen imagery in, protocol out. No login gymnastics. No field experts required.</p>
            <div className="cta-actions">
              <button className="btn btn-sage" onClick={() => setPage("cultivation")}>
                Start with a soil scan <Icon name="arrowRight" size={14} className="arrow" />
              </button>
            </div>
          </div>
        </Reveal>
      </section>

      <footer className="site-footer">
        <div className="footer-row">
          <div className="footer-brand">
            <span className="eyebrow">AgroSynapse AI · 2026</span>
            <p>Synaptic agronomy, built by Manoj Anumolu. Academic project; not a replacement for certified agronomist advice.</p>
          </div>
          <div className="footer-meta">
            <span className="label">v0.4 · Alpha</span>
            <span className="label">Models: SoilNet-v3 · PhytoNet-v2</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

window.HomePage = HomePage;
window.Reveal = Reveal;
window.MaskLine = MaskLine;
window.Counter = Counter;
window.useReveal = useReveal;

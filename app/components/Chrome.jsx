/* Side rail + top bar — persistent chrome */

const Rail = ({ page, setPage }) => {
  const items = [
    { key: "home", icon: "home", label: "Home" },
    { key: "cultivation", icon: "seedling", label: "Cultivation" },
    { key: "diagnostic", icon: "scan", label: "Diagnostic" },
    { key: "dashboard", icon: "chart", label: "Dashboard" },
  ];
  return (
    <aside className="rail">
      <div className="rail-logo" title="AgroSynapse AI">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 20c0-9 7-16 16-16 0 9-7 16-16 16Z" fill="currentColor" fillOpacity="0.2"/>
          <path d="M4 20 12 12"/>
        </svg>
      </div>
      {items.map(it => (
        <button
          key={it.key}
          className={"rail-btn " + (page === it.key ? "active" : "")}
          onClick={() => setPage(it.key)}
          title={it.label}
        >
          <Icon name={it.icon} size={18} />
        </button>
      ))}
      <div className="rail-spacer" />
      <button className="rail-btn" title="Settings"><Icon name="settings" size={17} /></button>
      <div className="rail-user">MA</div>
    </aside>
  );
};

const TopBar = ({ page, setPage, onTweaks }) => {
  const crumbs = {
    home: "",
    cultivation: "Predictive Cultivation",
    diagnostic: "Phyto-Diagnostic Suite",
    dashboard: "Analytics Dashboard",
  };
  const items = [
    { key: "home", label: "Home" },
    { key: "cultivation", label: "Cultivation" },
    { key: "diagnostic", label: "Diagnostic" },
    { key: "dashboard", label: "Dashboard" },
  ];
  return (
    <header className="topbar">
      <div className="topbar-crumb">
        <span className="dot" />
        <span>AgroSynapse{page !== "home" ? ` · ${crumbs[page]}` : ""}</span>
      </div>
      <div className="topbar-spacer" />
      <nav className="topbar-nav">
        {items.map(it => (
          <button key={it.key} className={page === it.key ? "active" : ""} onClick={() => setPage(it.key)}>{it.label}</button>
        ))}
      </nav>
      <button className="topbar-icon" title="Search"><Icon name="search" size={15} /></button>
      <button className="topbar-icon" title="Notifications"><Icon name="bell" size={15} /></button>
      <button className="topbar-icon" title="Tweaks" onClick={onTweaks}><Icon name="adjust" size={15} /></button>
    </header>
  );
};

window.Rail = Rail;
window.TopBar = TopBar;

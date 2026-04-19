/* Icons — centralized SVG set */
const Icon = ({ name, className = "", size = 16 }) => {
  const paths = {
    home: <path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1v-9.5Z" />,
    leaf: <path d="M4 20c0-9 7-16 16-16 0 9-7 16-16 16Zm0 0 8-8" />,
    seedling: <><path d="M12 22V10" /><path d="M12 10c0-3 2-6 6-6 0 3-2 6-6 6Z" /><path d="M12 12c0-2.5-2-5-6-5 0 2.5 2 5 6 5Z" /></>,
    scan: <><path d="M3 7V4a1 1 0 0 1 1-1h3" /><path d="M17 3h3a1 1 0 0 1 1 1v3" /><path d="M21 17v3a1 1 0 0 1-1 1h-3" /><path d="M7 21H4a1 1 0 0 1-1-1v-3" /><path d="M3 12h18" /></>,
    chart: <><path d="M3 3v18h18" /><path d="M7 14l4-4 4 4 5-6" /></>,
    bell: <><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10 21a2 2 0 0 0 4 0" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" /></>,
    arrowRight: <path d="M5 12h14M13 5l7 7-7 7" />,
    arrowUpRight: <><path d="M7 17 17 7" /><path d="M8 7h9v9" /></>,
    arrowDown: <path d="M12 5v14M5 12l7 7 7-7" />,
    upload: <><path d="M12 16V4m0 0-4 4m4-4 4 4" /><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" /></>,
    check: <path d="m5 12 5 5L20 7" />,
    x: <path d="M6 6l12 12M18 6 6 18" />,
    sparkles: <><path d="M12 3v6m0 6v6M3 12h6m6 0h6" /><path d="M5 5l3 3m8 8 3 3M5 19l3-3m8-8 3-3" /></>,
    water: <path d="M12 3s7 8 7 13a7 7 0 1 1-14 0c0-5 7-13 7-13Z" />,
    thermometer: <><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4 4 0 1 0 5 0Z" /></>,
    wind: <><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2" /><path d="M9.6 4.6A2 2 0 1 1 11 8H2" /><path d="M12.6 19.4A2 2 0 1 0 14 16H2" /></>,
    microscope: <><path d="M6 18h8" /><path d="M3 22h18" /><path d="M14 22a7 7 0 1 0 0-14h-1" /><path d="M9 14h2" /><path d="M9 12a2 2 0 0 1-2-2V6h4v4a2 2 0 0 1-2 2Z" /><path d="M12 6H6" /><path d="M10 2h4" /></>,
    flask: <><path d="M9 3h6" /><path d="M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3" /></>,
    plant: <><path d="M12 22v-8" /><path d="M12 14a5 5 0 0 1-5-5V6h5Z" /><path d="M12 14a5 5 0 0 0 5-5V6h-5Z" /></>,
    map: <><path d="M9 3 3 6v15l6-3 6 3 6-3V3l-6 3-6-3Z" /><path d="M9 3v15M15 6v15" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 8h.01M11 12h1v5h1" /></>,
    play: <path d="M8 5v14l11-7-11-7Z" />,
    download: <><path d="M12 4v12m0 0 4-4m-4 4-4-4" /><path d="M4 20h16" /></>,
    adjust: <><circle cx="8" cy="8" r="2" /><path d="M4 8h2m4 0h10" /><circle cx="16" cy="16" r="2" /><path d="M4 16h10m4 0h2" /></>,
    droplet: <path d="M12 3s7 8 7 13a7 7 0 1 1-14 0c0-5 7-13 7-13Z" />,
  };
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"
      className={className}>
      {paths[name]}
    </svg>
  );
};

window.Icon = Icon;

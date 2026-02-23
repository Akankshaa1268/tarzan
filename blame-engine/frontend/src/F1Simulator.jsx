import { useState, useEffect, useRef, useCallback } from "react";

// ─── TRACK DEFINITIONS (normalized 0-1 coordinates, scaled to canvas) ────────
const TRACKS = {
  Monaco: {
    name: "Circuit de Monaco",
    laps: 78,
    color: "#00BFFF",
    pitLane: { entry: 0.08, exit: 0.12 },
    // SVG path points [x,y] normalized 0-1
    path: [
      [0.50, 0.10], [0.72, 0.10], [0.82, 0.14], [0.86, 0.22], [0.84, 0.32],
      [0.78, 0.38], [0.70, 0.40], [0.60, 0.42], [0.55, 0.48], [0.58, 0.56],
      [0.66, 0.62], [0.72, 0.70], [0.72, 0.80], [0.65, 0.88], [0.52, 0.90],
      [0.38, 0.88], [0.28, 0.82], [0.22, 0.72], [0.20, 0.60], [0.24, 0.50],
      [0.30, 0.42], [0.32, 0.32], [0.28, 0.22], [0.32, 0.14], [0.42, 0.10],
      [0.50, 0.10]
    ],
    drsZones: [[0.45, 0.55]],
    corners: [
      { t: 0.04, name: "Ste Devote" }, { t: 0.15, name: "Massenet" },
      { t: 0.28, name: "Casino" }, { t: 0.42, name: "Mirabeau" },
      { t: 0.55, name: "Loews" }, { t: 0.68, name: "Portier" },
      { t: 0.82, name: "Tabac" }, { t: 0.92, name: "Rascasse" }
    ]
  },
  Silverstone: {
    name: "Silverstone Circuit",
    laps: 52,
    color: "#FF6B35",
    pitLane: { entry: 0.05, exit: 0.10 },
    path: [
      [0.50, 0.08], [0.72, 0.08], [0.82, 0.12], [0.88, 0.20], [0.88, 0.35],
      [0.82, 0.45], [0.88, 0.55], [0.88, 0.70], [0.80, 0.80], [0.65, 0.88],
      [0.50, 0.90], [0.35, 0.88], [0.20, 0.80], [0.12, 0.70], [0.12, 0.55],
      [0.18, 0.45], [0.12, 0.35], [0.12, 0.20], [0.18, 0.12], [0.30, 0.08],
      [0.50, 0.08]
    ],
    drsZones: [[0.10, 0.20], [0.60, 0.70]],
    corners: [
      { t: 0.05, name: "Copse" }, { t: 0.18, name: "Maggotts" },
      { t: 0.30, name: "Becketts" }, { t: 0.45, name: "Chapel" },
      { t: 0.60, name: "Stowe" }, { t: 0.75, name: "Vale" },
      { t: 0.88, name: "Club" }
    ]
  },
  Suzuka: {
    name: "Suzuka Circuit",
    laps: 53,
    color: "#FF0000",
    pitLane: { entry: 0.06, exit: 0.11 },
    path: [
      [0.50, 0.08], [0.68, 0.08], [0.80, 0.16], [0.85, 0.28], [0.80, 0.38],
      [0.72, 0.42], [0.78, 0.50], [0.85, 0.60], [0.82, 0.72], [0.70, 0.82],
      [0.56, 0.88], [0.44, 0.88], [0.30, 0.82], [0.18, 0.72], [0.15, 0.60],
      [0.22, 0.50], [0.28, 0.42], [0.20, 0.32], [0.15, 0.20], [0.22, 0.12],
      [0.35, 0.08], [0.50, 0.08]
    ],
    drsZones: [[0.45, 0.55]],
    corners: [
      { t: 0.06, name: "T1" }, { t: 0.18, name: "S Curves" },
      { t: 0.32, name: "Dunlop" }, { t: 0.45, name: "Degner" },
      { t: 0.58, name: "Hairpin" }, { t: 0.72, name: "Spoon" },
      { t: 0.85, name: "130R" }, { t: 0.93, name: "Chicane" }
    ]
  }
};

const COMPOUNDS = {
  Soft: { color: "#FF1744", deg: 0.045, grip: 1.00, warmup: 1, label: "S" },
  Medium: { color: "#FFD600", deg: 0.025, grip: 0.97, warmup: 3, label: "M" },
  Hard: { color: "#E0E0E0", deg: 0.012, grip: 0.94, warmup: 5, label: "H" },
  Inter: { color: "#00E676", deg: 0.030, grip: 0.88, warmup: 2, label: "I" },
  Wet: { color: "#2979FF", deg: 0.020, grip: 0.82, warmup: 2, label: "W" },
};

const TEAM_COLORS = {
  "Red Bull": "#3671C6", "Ferrari": "#E8002D",
  "McLaren": "#FF8000", "Mercedes": "#27F4D2",
  "Aston Martin": "#229971", "Alpine": "#0093CC",
  "Williams": "#64C4FF", "Haas": "#B6BABD",
  "Sauber": "#52E252", "RB": "#6692FF",
};

const AI_DRIVERS = [
  { id: "VER", name: "Verstappen", team: "Red Bull", pace: 0.98, skill: 0.97 },
  { id: "LEC", name: "Leclerc", team: "Ferrari", pace: 0.96, skill: 0.95 },
  { id: "NOR", name: "Norris", team: "McLaren", pace: 0.95, skill: 0.94 },
  { id: "HAM", name: "Hamilton", team: "Mercedes", pace: 0.94, skill: 0.96 },
  { id: "ALO", name: "Alonso", team: "Aston Martin", pace: 0.93, skill: 0.95 },
  { id: "RUS", name: "Russell", team: "Mercedes", pace: 0.93, skill: 0.92 },
  { id: "PIA", name: "Piastri", team: "McLaren", pace: 0.92, skill: 0.91 },
  { id: "SAI", name: "Sainz", team: "Ferrari", pace: 0.92, skill: 0.91 },
  { id: "STR", name: "Stroll", team: "Aston Martin", pace: 0.88, skill: 0.86 },
  { id: "GAS", name: "Gasly", team: "Alpine", pace: 0.87, skill: 0.87 },
  { id: "OCO", name: "Ocon", team: "Alpine", pace: 0.86, skill: 0.86 },
  { id: "ALB", name: "Albon", team: "Williams", pace: 0.85, skill: 0.85 },
  { id: "MAG", name: "Magnussen", team: "Haas", pace: 0.84, skill: 0.84 },
  { id: "BOT", name: "Bottas", team: "Sauber", pace: 0.83, skill: 0.85 },
  { id: "TSU", name: "Tsunoda", team: "RB", pace: 0.87, skill: 0.86 },
  { id: "ZHO", name: "Zhou", team: "Sauber", pace: 0.82, skill: 0.82 },
  { id: "SAR", name: "Sargeant", team: "Williams", pace: 0.80, skill: 0.80 },
  { id: "HUL", name: "Hulkenberg", team: "Haas", pace: 0.86, skill: 0.86 },
  { id: "RIC", name: "Ricciardo", team: "RB", pace: 0.87, skill: 0.87 },
];

// ─── PATH UTILITIES ───────────────────────────────────────────────────────────
function getPointOnPath(path, t, W, H, margin = 0.08) {
  const pts = path.map(([x, y]) => [x * W * (1 - 2 * margin) + W * margin, y * H * (1 - 2 * margin) + H * margin]);
  const total = pts.length - 1;
  const idx = Math.floor(t * total) % total;
  const frac = (t * total) % 1;
  const a = pts[idx % pts.length];
  const b = pts[(idx + 1) % pts.length];
  return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac];
}

function getAngleOnPath(path, t, W, H, margin = 0.08) {
  const dt = 0.002;
  const [x1, y1] = getPointOnPath(path, t, W, H, margin);
  const [x2, y2] = getPointOnPath(path, (t + dt) % 1, W, H, margin);
  return Math.atan2(y2 - y1, x2 - x1);
}

// ─── STYLES ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800;900&family=Share+Tech+Mono&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --red:#E8002D;--bg:#060608;--card:#0d0d12;--border:rgba(255,255,255,0.07);
    --text:#f0f0f0;--muted:#666;--green:#00C897;--yellow:#FFD600;--blue:#4A9EFF;
    --font:'Barlow Condensed',sans-serif;--mono:'Share Tech Mono',monospace;
  }
  html,body,#root{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);overflow:hidden}
  ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:var(--red)}

  .sim-root{display:flex;flex-direction:column;height:100vh;background:var(--bg)}

  /* TOP BAR */
  .topbar{
    display:flex;align-items:center;gap:12px;padding:0 16px;
    height:48px;background:rgba(6,6,8,0.95);border-bottom:1px solid var(--border);
    flex-shrink:0;
  }
  .topbar-logo{font-size:18px;font-weight:900;letter-spacing:3px;color:var(--text);flex-shrink:0}
  .topbar-logo span{color:var(--red)}
  .topbar-sep{width:1px;height:24px;background:var(--border)}
  .race-info{display:flex;gap:16px;align-items:center;flex:1}
  .race-stat{display:flex;flex-direction:column;align-items:center;min-width:48px}
  .race-stat-v{font-size:20px;font-weight:900;line-height:1;color:var(--text)}
  .race-stat-l{font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;margin-top:1px}
  .race-stat-v.red{color:var(--red)}
  .race-stat-v.green{color:var(--green)}
  .race-stat-v.yellow{color:var(--yellow)}
  .flag-indicator{
    display:flex;align-items:center;gap:6px;padding:4px 12px;
    border-radius:3px;font-size:12px;font-weight:700;letter-spacing:1.5px;
    text-transform:uppercase;
  }
  .flag-green{background:rgba(0,200,151,0.15);border:1px solid var(--green);color:var(--green)}
  .flag-yellow{background:rgba(255,214,0,0.15);border:1px solid var(--yellow);color:var(--yellow)}
  .flag-red-f{background:rgba(232,0,45,0.15);border:1px solid var(--red);color:var(--red)}
  .flag-sc{background:rgba(255,255,255,0.1);border:1px solid #fff;color:#fff}
  .dot-pulse{width:6px;height:6px;border-radius:50%;animation:pulse 1.2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}

  /* MAIN AREA */
  .main-area{display:flex;flex:1;overflow:hidden}

  /* CANVAS */
  .canvas-wrap{
    flex:1;position:relative;overflow:hidden;
    background:radial-gradient(ellipse at 50% 50%, #0a0a14 0%, #060608 100%);
  }
  canvas{display:block;width:100%;height:100%}

  /* WEATHER OVERLAY */
  .weather-overlay{
    position:absolute;top:12px;left:12px;
    background:rgba(6,6,8,0.85);border:1px solid var(--border);
    border-radius:6px;padding:8px 14px;
    font-size:13px;font-weight:700;letter-spacing:1px;
    display:flex;align-items:center;gap:8px;
  }

  /* CORNER NAME POPUP */
  .corner-popup{
    position:absolute;bottom:60px;left:50%;transform:translateX(-50%);
    background:rgba(232,0,45,0.9);border-radius:4px;
    padding:4px 14px;font-size:12px;font-weight:700;letter-spacing:2px;
    text-transform:uppercase;pointer-events:none;
    animation:fadeCorner 2s ease forwards;
  }
  @keyframes fadeCorner{0%{opacity:0;transform:translateX(-50%) translateY(8px)}
    20%{opacity:1;transform:translateX(-50%) translateY(0)}
    80%{opacity:1}100%{opacity:0}}

  /* DRS INDICATOR */
  .drs-indicator{
    position:absolute;top:12px;right:12px;
    padding:6px 16px;border-radius:4px;font-size:14px;font-weight:900;
    letter-spacing:2px;transition:all 0.3s;
  }
  .drs-on{background:rgba(0,200,151,0.2);border:1px solid var(--green);color:var(--green);box-shadow:0 0 20px rgba(0,200,151,0.3)}
  .drs-off{background:rgba(255,255,255,0.05);border:1px solid var(--border);color:var(--muted)}
  .drs-available{background:rgba(0,200,151,0.08);border:1px solid rgba(0,200,151,0.4);color:rgba(0,200,151,0.7);animation:drsFlash 0.5s infinite}
  @keyframes drsFlash{0%,100%{opacity:1}50%{opacity:0.5}}

  /* RIGHT PANEL */
  .right-panel{
    width:260px;flex-shrink:0;display:flex;flex-direction:column;
    background:var(--card);border-left:1px solid var(--border);overflow-y:auto;
  }

  /* TIMING TOWER */
  .timing-tower{flex-shrink:0}
  .panel-title{
    padding:10px 14px;font-size:11px;font-weight:700;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);
    border-bottom:1px solid var(--border);background:rgba(255,255,255,0.02);
  }
  .timing-row{
    display:flex;align-items:center;gap:6px;padding:5px 10px;
    border-bottom:1px solid rgba(255,255,255,0.03);font-size:12px;
    transition:background 0.15s;
  }
  .timing-row:hover{background:rgba(255,255,255,0.03)}
  .timing-row.player{background:rgba(232,0,45,0.08)!important;border-left:2px solid var(--red)}
  .timing-pos{font-weight:900;width:18px;font-size:13px;color:var(--muted)}
  .timing-pos.p1{color:var(--yellow)}
  .timing-pos.p2{color:#C0C0C0}
  .timing-pos.p3{color:#CD7F32}
  .timing-team-bar{width:3px;height:16px;border-radius:1px;flex-shrink:0}
  .timing-name{flex:1;font-weight:700;font-size:11px;letter-spacing:0.5px}
  .timing-gap{font-family:var(--mono);font-size:10px;color:var(--muted);width:52px;text-align:right}
  .timing-tyre{
    width:16px;height:16px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:8px;font-weight:900;color:#000;flex-shrink:0;
  }

  /* BOTTOM CONTROLS */
  .bottom-controls{
    position:absolute;bottom:0;left:0;right:260px;
    background:rgba(6,6,8,0.95);border-top:1px solid var(--border);
    padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;
  }

  /* CONTROL GROUPS */
  .ctrl-group{display:flex;flex-direction:column;gap:4px}
  .ctrl-label{font-size:9px;letter-spacing:2px;color:var(--muted);text-transform:uppercase}
  .ctrl-buttons{display:flex;gap:4px}
  .ctrl-btn{
    padding:7px 12px;background:rgba(255,255,255,0.05);border:1px solid var(--border);
    border-radius:3px;color:var(--muted);font-family:var(--font);font-size:12px;
    font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;
    transition:all 0.15s;white-space:nowrap;
  }
  .ctrl-btn:hover{background:rgba(255,255,255,0.1);color:var(--text);border-color:rgba(255,255,255,0.2)}
  .ctrl-btn.active{background:var(--red);border-color:var(--red);color:#fff;box-shadow:0 0 12px rgba(232,0,45,0.4)}
  .ctrl-btn.green-btn.active{background:var(--green);border-color:var(--green);color:#000}
  .ctrl-btn.yellow-btn.active{background:var(--yellow);border-color:var(--yellow);color:#000}
  .ctrl-btn.pit-btn{border-color:rgba(232,0,45,0.4);color:var(--red)}
  .ctrl-btn.pit-btn:hover,.ctrl-btn.pit-btn.active{background:var(--red);color:#fff;box-shadow:0 0 20px rgba(232,0,45,0.5)}
  .ctrl-btn.big{padding:10px 20px;font-size:14px}
  .ctrl-btn:disabled{opacity:0.3;cursor:not-allowed}

  /* THROTTLE/BRAKE BARS */
  .pedals{display:flex;gap:6px;align-items:flex-end}
  .pedal-wrap{display:flex;flex-direction:column;align-items:center;gap:3px}
  .pedal-bar{width:18px;height:48px;background:rgba(255,255,255,0.05);border:1px solid var(--border);border-radius:2px;position:relative;overflow:hidden}
  .pedal-fill{position:absolute;bottom:0;left:0;right:0;transition:height 0.1s;border-radius:1px}
  .pedal-fill.throttle{background:var(--green)}
  .pedal-fill.brake{background:var(--red)}
  .pedal-label{font-size:8px;letter-spacing:1px;color:var(--muted);text-transform:uppercase}

  /* GEAR DISPLAY */
  .gear-display{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    width:52px;height:52px;background:rgba(255,255,255,0.04);
    border:1px solid var(--border);border-radius:4px;
  }
  .gear-n{font-size:28px;font-weight:900;line-height:1;color:var(--text)}
  .gear-l{font-size:8px;letter-spacing:1px;color:var(--muted);text-transform:uppercase}

  /* SPEED / RPM */
  .speed-display{
    display:flex;flex-direction:column;align-items:center;
    background:rgba(255,255,255,0.03);border:1px solid var(--border);
    border-radius:4px;padding:4px 12px;
  }
  .speed-v{font-family:var(--mono);font-size:24px;font-weight:700;color:var(--text);line-height:1}
  .speed-l{font-size:8px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase}

  /* TYRE STATUS */
  .tyre-status{
    display:flex;flex-direction:column;gap:3px;
    background:rgba(255,255,255,0.03);border:1px solid var(--border);
    border-radius:4px;padding:6px 10px;
  }
  .tyre-row{display:flex;align-items:center;gap:6px;font-size:11px}
  .tyre-icon{width:14px;height:14px;border-radius:50%;font-size:7px;font-weight:900;display:flex;align-items:center;justify-content:center;color:#000;flex-shrink:0}
  .tyre-wear-bar{flex:1;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden}
  .tyre-wear-fill{height:100%;border-radius:2px;transition:width 0.5s}
  .tyre-temp{font-family:var(--mono);font-size:10px;width:36px;text-align:right}

  /* MINIMAP */
  .minimap{margin:12px;border-radius:6px;overflow:hidden;border:1px solid var(--border);flex-shrink:0}

  /* RADIO MSG */
  .radio-feed{
    margin:0 12px 8px;padding:8px 10px;
    background:rgba(0,200,151,0.06);border:1px solid rgba(0,200,151,0.2);
    border-radius:4px;font-size:11px;color:var(--green);line-height:1.4;
    font-style:italic;flex-shrink:0;
  }
  .radio-label{font-size:9px;letter-spacing:2px;color:rgba(0,200,151,0.6);text-transform:uppercase;margin-bottom:2px;font-style:normal}

  /* SETUP SCREEN */
  .setup-screen{
    position:fixed;inset:0;background:rgba(6,6,8,0.97);
    display:flex;align-items:center;justify-content:center;z-index:100;
    backdrop-filter:blur(8px);
  }
  .setup-card{
    background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:40px;width:100%;max-width:520px;
  }
  .setup-title{font-size:36px;font-weight:900;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}
  .setup-sub{font-size:14px;color:var(--muted);margin-bottom:32px}
  .setup-field{margin-bottom:20px}
  .setup-label{font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:6px;display:block}
  .setup-select{
    width:100%;padding:10px 14px;background:rgba(255,255,255,0.04);
    border:1px solid var(--border);border-radius:4px;color:var(--text);
    font-family:var(--font);font-size:15px;font-weight:600;appearance:none;
    cursor:pointer;transition:border-color 0.2s;
  }
  .setup-select:focus{outline:none;border-color:var(--red)}
  .setup-select option{background:#1a1a22}
  .driver-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
  .driver-card{
    padding:8px 6px;background:rgba(255,255,255,0.03);border:1px solid var(--border);
    border-radius:4px;text-align:center;cursor:pointer;transition:all 0.15s;
  }
  .driver-card:hover{background:rgba(255,255,255,0.07);border-color:rgba(255,255,255,0.15)}
  .driver-card.selected{border-color:var(--red);background:rgba(232,0,45,0.1)}
  .driver-card-id{font-size:14px;font-weight:900;letter-spacing:1px}
  .driver-card-name{font-size:9px;color:var(--muted);margin-top:2px}
  .start-btn{
    width:100%;padding:14px;background:var(--red);border:none;border-radius:4px;
    color:#fff;font-family:var(--font);font-size:18px;font-weight:900;
    letter-spacing:3px;text-transform:uppercase;cursor:pointer;
    transition:all 0.2s;margin-top:24px;
  }
  .start-btn:hover{background:#ff1a45;box-shadow:0 4px 32px rgba(232,0,45,0.4);transform:translateY(-1px)}
  .start-btn:disabled{background:#333;cursor:not-allowed;transform:none;box-shadow:none}

  /* RESULTS */
  .results-overlay{
    position:fixed;inset:0;background:rgba(6,6,8,0.96);
    display:flex;align-items:center;justify-content:center;z-index:200;
  }
  .results-card{
    background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:40px;width:100%;max-width:480px;text-align:center;
  }
  .results-pos{font-size:80px;font-weight:900;line-height:1;color:var(--red)}
  .results-label{font-size:13px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:24px}
  .results-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);font-size:14px}
  .results-row-val{font-family:var(--mono);font-weight:700}
  .iq-big{font-size:56px;font-weight:900;line-height:1}

  /* SC ALERT */
  .sc-alert{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    background:rgba(255,214,0,0.95);color:#000;
    padding:12px 32px;border-radius:6px;
    font-size:24px;font-weight:900;letter-spacing:3px;
    animation:scPop 0.4s ease;pointer-events:none;
  }
  @keyframes scPop{from{opacity:0;transform:translate(-50%,-50%) scale(0.8)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}

  /* PIT ANIMATION */
  .pit-alert{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    background:rgba(232,0,45,0.95);color:#fff;
    padding:10px 28px;border-radius:6px;
    font-size:20px;font-weight:900;letter-spacing:2px;
    animation:scPop 0.3s ease;pointer-events:none;
  }

  /* NOTIFICATION */
  .notif{
    position:absolute;top:60px;left:50%;transform:translateX(-50%);
    background:rgba(13,13,18,0.95);border:1px solid var(--border);
    border-radius:6px;padding:8px 20px;font-size:13px;font-weight:600;
    animation:fadeCorner 3s ease forwards;pointer-events:none;white-space:nowrap;
  }

  /* SPEED LINES */
  .speed-lines{position:absolute;inset:0;pointer-events:none;overflow:hidden}

  @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  .fade-in{animation:fadeIn 0.5s ease}
`;

// ─── MAIN SIMULATOR ───────────────────────────────────────────────────────────
export default function F1Simulator() {
  // Setup
  const [phase, setPhase] = useState("setup"); // setup | racing | results
  const [selectedTrack, setSelectedTrack] = useState("Monaco");
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [selectedCompound, setSelectedCompound] = useState("Medium");

  // Race state
  const canvasRef = useRef(null);
  const minimapRef = useRef(null);
  const animRef = useRef(null);
  const stateRef = useRef(null);

  // UI state (reactive)
  const [uiState, setUiState] = useState({
    lap: 1, totalLaps: 52, flag: "green", position: 5,
    speed: 0, gear: 1, throttle: 0, brake: 0,
    compound: "Medium", tyreWear: 0, tyreTemp: 80,
    drs: false, drsAvailable: false, drsZone: false,
    pitRequested: false, inPit: false, pitDuration: 0,
    fuel: 100, ersDeploy: false, ersCharge: 50,
    gap_ahead: null, gap_behind: null,
    radioMsg: "Good luck out there. You've got this. 🏎",
    scActive: false, showSCAlert: false,
    cornerName: null, showCornerName: false,
    showPitAlert: false, notifications: [],
    finalPos: null, strategyIQ: null,
    drivers: [], playerProgress: 0,
  });

  const addNotif = useCallback((msg, color = "#fff") => {
    const id = Date.now();
    setUiState(s => ({ ...s, notifications: [...s.notifications, { id, msg, color }] }));
    setTimeout(() => setUiState(s => ({ ...s, notifications: s.notifications.filter(n => n.id !== id) })), 3000);
  }, []);

  // ─── INITIALIZE RACE ────────────────────────────────────────────────────────
  const startRace = useCallback(() => {
    const track = TRACKS[selectedTrack];
    const myDriver = AI_DRIVERS.find(d => d.id === selectedDriver) || AI_DRIVERS[0];
    const totalLaps = track.laps;

    // Create all cars
    const allDrivers = AI_DRIVERS.filter(d => d.id !== myDriver.id);
    const cars = allDrivers.map((d, i) => ({
      ...d,
      progress: 0.05 + (i * 0.003),     // stagger on track
      targetProgress: 0.05 + (i * 0.003),
      lapProgress: 0,
      lap: 1,
      pos: i + 2,
      compound: ["Soft", "Medium", "Medium", "Hard"][i % 4],
      tyreAge: Math.floor(Math.random() * 5),
      tyreWear: 0,
      speed: 280 + Math.random() * 40,
      pitting: false,
      pitTimer: 0,
      retired: false,
      color: TEAM_COLORS[d.team] || "#888",
      pitHistory: [],
    }));

    // Player car
    const player = {
      ...myDriver,
      isPlayer: true,
      progress: 0.04,
      lapProgress: 0,
      lap: 1,
      pos: Math.floor(Math.random() * 8) + 3,
      compound: selectedCompound,
      tyreAge: 0,
      tyreWear: 0,
      speed: 0,
      gear: 1,
      throttle: 0,
      brake: 0,
      drs: false,
      drsZone: false,
      drsAvailable: false,
      inPit: false,
      pitTimer: 0,
      pitRequested: false,
      fuel: 100,
      ers: 50,
      color: TEAM_COLORS[myDriver.team] || "#E8002D",
      mode: "normal", // normal | push | conserve
      retired: false,
      pitHistory: [],
    };

    stateRef.current = {
      player,
      cars,
      track,
      totalLaps,
      flag: "green",
      scTimer: 0,
      tick: 0,
      lastLapTime: Date.now(),
      scActive: false,
      gameSpeed: 1,
      paused: false,
    };

    setPhase("racing");
    setUiState(s => ({
      ...s,
      totalLaps,
      lap: 1,
      compound: selectedCompound,
      flag: "green",
      radioMsg: `${myDriver.id} this is your engineer. Box set up, tyres warm, let's go! 📻`,
      drivers: [player, ...cars].sort((a, b) => a.pos - b.pos),
    }));
  }, [selectedTrack, selectedDriver, selectedCompound]);

  // ─── CONTROLS ───────────────────────────────────────────────────────────────
  const setMode = (mode) => {
    if (!stateRef.current) return;
    stateRef.current.player.mode = mode;
    const msgs = {
      push: "📻 Copy, we're pushing. Watch the tyres.",
      normal: "📻 Normal mode. Balance fuel and pace.",
      conserve: "📻 Conserve mode. Protect the tyres.",
    };
    setUiState(s => ({ ...s, radioMsg: msgs[mode] }));
  };

  const requestPit = () => {
    if (!stateRef.current || stateRef.current.player.inPit) return;
    stateRef.current.player.pitRequested = true;
    setUiState(s => ({ ...s, pitRequested: true, radioMsg: "📻 Copy, box box box! Pit lane open." }));
  };

  const changeTyre = (compound) => {
    if (!stateRef.current) return;
    stateRef.current.nextCompound = compound;
    setUiState(s => ({ ...s, radioMsg: `📻 Confirmed, ${compound}s going on. Stand by.` }));
  };

  const toggleDRS = () => {
    if (!stateRef.current) return;
    const p = stateRef.current.player;
    if (p.drsAvailable) { p.drs = !p.drs; }
  };

  const deployERS = () => {
    if (!stateRef.current) return;
    stateRef.current.player.ersDeploy = !stateRef.current.player.ersDeploy;
  };

  const togglePause = () => {
    if (!stateRef.current) return;
    stateRef.current.paused = !stateRef.current.paused;
  };

  const setGameSpeed = (s) => {
    if (!stateRef.current) return;
    stateRef.current.gameSpeed = s;
  };

  // ─── ENGINEER RADIO SYSTEM ──────────────────────────────────────────────────
  const engineerSuggestions = useCallback(() => {
    const st = stateRef.current;
    if (!st) return;
    const p = st.player;
    const allCars = [p, ...st.cars].sort((a, b) => (b.lap + b.progress) - (a.lap + a.progress));
    const playerPos = allCars.findIndex(c => c.isPlayer) + 1;
    const ahead = allCars[playerPos - 2];
    const behind = allCars[playerPos];
    const lapsLeft = st.totalLaps - p.lap;

    const suggestions = [];

    // Tyre wear warning
    if (p.tyreWear > 75) suggestions.push(`📻 Those ${p.compound}s are GONE. Box this lap or next, seriously.`);
    else if (p.tyreWear > 55) suggestions.push(`📻 Tyre wear at ${Math.round(p.tyreWear)}%. Start thinking about the box.`);

    // Undercut opportunity
    if (ahead && !ahead.inPit && ahead.tyreWear > 60 && p.tyreWear < 50) {
      suggestions.push(`📻 ${ahead.id} is on worn tyres — undercut window open! Box NOW for the position.`);
    }

    // Overcut opportunity
    if (ahead && ahead.inPit) {
      suggestions.push(`📻 ${ahead.id} is in the pits! Stay out, we overcut them. Push push push!`);
    }

    // SC opportunity
    if (st.scActive && !p.inPit && p.tyreWear > 30) {
      suggestions.push(`📻 FREE PIT STOP under SC! BOX BOX BOX — this is the moment!`);
    }

    // Gap management
    if (behind && (p.lap + p.progress - behind.lap - behind.progress) * 90 < 1.5) {
      suggestions.push(`📻 ${behind.id} is ${(((p.lap + p.progress) - (behind.lap + behind.progress)) * 90).toFixed(1)}s behind — defend on entry!`);
    }

    // DRS reminder
    if (p.drsAvailable && !p.drs) {
      suggestions.push(`📻 You're in the DRS zone — activate it! Free speed on the straight.`);
    }

    // Fuel saving
    if (p.fuel < 25 && lapsLeft > 5) {
      suggestions.push(`📻 Fuel is tight — switch to SAVE mode for 3 laps. We can attack later.`);
    }

    // Push opportunity  
    if (playerPos > 3 && p.tyreWear < 30 && p.fuel > 40 && !st.scActive) {
      suggestions.push(`📻 Car feels good, tyres fresh. This is your window — PUSH mode, let's go!`);
    }

    // ERS reminder
    if (p.ers > 80 && !p.ersDeploy && playerPos > 1) {
      suggestions.push(`📻 ERS fully charged — deploy it on the next straight for extra pace.`);
    }

    // Laps to go
    if (lapsLeft === 5) suggestions.push(`📻 Five laps to go. Hold position, bring it home clean.`);
    if (lapsLeft === 3) suggestions.push(`📻 Three laps. Everything you've got now — leave nothing out there!`);
    if (lapsLeft === 1) suggestions.push(`📻 FINAL LAP! This is it — push everything, cross that line!`);

    // Random tactical insight
    const tactical = [
      `📻 Gap to ${ahead?.id || "leader"} is ${ahead ? (((ahead.lap + ahead.progress - p.lap - p.progress) * 90).toFixed(1)) : "0.0"}s. Stay focussed.`,
      `📻 Track temp rising — deg will increase. Watch those rear tyres.`,
      `📻 We're P${playerPos}. Target is P${Math.max(1, playerPos - 1)} — ${allCars[playerPos - 2]?.id || "leader"} is the next car.`,
    ];

    if (suggestions.length === 0) {
      suggestions.push(tactical[Math.floor(Math.random() * tactical.length)]);
    }

    setUiState(s => ({ ...s, radioMsg: suggestions[0] }));
  }, []);

  // Fire engineer radio every 8 seconds
  useEffect(() => {
    if (phase !== "racing") return;
    const interval = setInterval(engineerSuggestions, 8000);
    return () => clearInterval(interval);
  }, [phase, engineerSuggestions]);

  // ─── GAME LOOP ───────────────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "racing") return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let frameCount = 0;
    let lastTime = 0;
    const TARGET_FPS = 60;
    const FRAME_TIME = 1000 / TARGET_FPS;

    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener("resize", resize);

    const loop = (timestamp) => {
      // Throttle to 60fps
      const elapsed = timestamp - lastTime;
      if (elapsed < FRAME_TIME - 1) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }
      lastTime = timestamp - (elapsed % FRAME_TIME);

      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      const st = stateRef.current;
      if (!st) { animRef.current = requestAnimationFrame(loop); return; }

      frameCount++;
      // Game simulation speed: 1x = realistic slow, 3x = faster, 6x = quick race
      // Each tick = 1 real frame worth of physics
      const STEPS = st.gameSpeed; // run physics N times per render
      for (let i = 0; i < STEPS; i++) {
        if (!st.paused) updateGameState(st, 0.016, W, H, frameCount);
      }
      const DT = 0.016 * st.gameSpeed;

      renderFrame(ctx, W, H, st, frameCount);

      // Sync UI every 10 frames
      if (frameCount % 10 === 0) {
        const p = st.player;
        const allCars = [p, ...st.cars].filter(c => !c.retired);
        // Sort by lap + progress desc
        allCars.sort((a, b) => (b.lap + b.progress) - (a.lap + a.progress));
        allCars.forEach((c, i) => c.pos = i + 1);

        const playerPos = p.pos;
        const ahead = allCars[playerPos - 2];
        const behind = allCars[playerPos];

        setUiState(s => ({
          ...s,
          lap: p.lap,
          position: playerPos,
          speed: Math.round(p.speed),
          gear: p.gear || 1,
          throttle: p.throttle || 0,
          brake: p.brake || 0,
          compound: p.compound,
          tyreWear: Math.min(100, p.tyreWear),
          tyreTemp: Math.round(80 + p.tyreAge * 2 + (p.mode === "push" ? 15 : 0)),
          drs: p.drs,
          drsAvailable: p.drsAvailable,
          fuel: Math.max(0, 100 - (p.lap / st.totalLaps * 95)),
          ersDeploy: p.ersDeploy,
          ersCharge: Math.round(p.ers),
          inPit: p.inPit,
          pitRequested: p.pitRequested,
          flag: st.flag,
          scActive: st.scActive,
          gap_ahead: ahead ? `+${((ahead.lap + ahead.progress - p.lap - p.progress) * 90).toFixed(1)}s` : "LEADER",
          gap_behind: behind ? `-${((p.lap + p.progress - behind.lap - behind.progress) * 90).toFixed(1)}s` : null,
          drivers: [...allCars],
          playerProgress: p.progress,
        }));

        // Check race end
        if (p.lap > st.totalLaps) {
          const iq = Math.max(40, Math.min(99, 95 - (playerPos - 1) * 4 + Math.floor(Math.random() * 10)));
          setUiState(s => ({ ...s, finalPos: playerPos, strategyIQ: iq }));
          setPhase("results");
          cancelAnimationFrame(animRef.current);
          return;
        }
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [phase]);

  // ─── GAME UPDATE ────────────────────────────────────────────────────────────
  function updateGameState(st, DT, W, H, frame) {
    const track = st.track;
    const path = track.path;
    const p = st.player;

    // SC random trigger
    if (!st.scActive && Math.random() < 0.0003 * DT * 60) {
      st.scActive = true;
      st.flag = "sc";
      st.scTimer = 600;
      setUiState(s => ({ ...s, showSCAlert: true, radioMsg: "📻 Safety car deployed! Do we box? Over." }));
      setTimeout(() => setUiState(s => ({ ...s, showSCAlert: false })), 3000);
    }
    if (st.scActive) {
      st.scTimer -= DT * 60;
      if (st.scTimer <= 0) {
        st.scActive = false;
        st.flag = "green";
        setUiState(s => ({ ...s, radioMsg: "📻 Safety car in this lap. Get ready, green flag!" }));
      }
    }

    // ─ Player update ─
    if (!p.inPit) {
      // Speed based on mode
      const modeSpeed = { push: 1.0, normal: 0.97, conserve: 0.93 }[p.mode] || 0.97;
      const scMult = st.scActive ? 0.6 : 1.0;
      const tyreGrip = Math.max(0.7, 1 - p.tyreWear * 0.003);
      const compoundGrip = COMPOUNDS[p.compound]?.grip || 1;
      const drsBoost = p.drs ? 1.015 : 1.0;
      const ersBoost = p.ersDeploy && p.ers > 0 ? 1.018 : 1.0;

      const targetSpeed = 300 * modeSpeed * scMult * tyreGrip * compoundGrip * drsBoost * ersBoost;
      p.speed = p.speed * 0.92 + targetSpeed * 0.08;

      // Throttle/brake visuals
      p.throttle = Math.min(100, p.speed / 3);
      p.brake = Math.max(0, (300 - p.speed) * 0.3);

      // Progress along track
      const progressRate = (p.speed / 300) * 0.0012 * DT * 60;
      p.progress += progressRate;

      // Gear simulation
      const gearThresh = [0, 80, 130, 180, 220, 260, 290, 310];
      for (let g = 7; g >= 1; g--) {
        if (p.speed >= gearThresh[g]) { p.gear = g; break; }
      }

      if (p.progress >= 1) {
        p.progress = 0;
        p.lap += 1;
        p.tyreAge += 1;
        const deg = COMPOUNDS[p.compound]?.deg || 0.025;
        const modeMult = { push: 1.4, normal: 1.0, conserve: 0.7 }[p.mode] || 1.0;
        p.tyreWear += deg * 100 * modeMult;
        p.fuel = Math.max(0, p.fuel - 1.8);
      }

      // DRS zones
      const drsZones = track.drsZones || [];
      p.drsZone = drsZones.some(([s, e]) => p.progress >= s && p.progress <= e);
      p.drsAvailable = p.drsZone && !st.scActive;
      if (!p.drsAvailable) p.drs = false;

      // ERS
      if (p.ersDeploy && p.ers > 0) {
        p.ers = Math.max(0, p.ers - 0.3 * DT * 60);
      } else {
        p.ers = Math.min(100, p.ers + 0.1 * DT * 60);
      }

      // Pit request
      if (p.pitRequested && p.progress >= track.pitLane.entry && p.progress <= track.pitLane.exit + 0.02) {
        p.inPit = true;
        p.pitTimer = 180; // ~3s at 60fps
        p.pitRequested = false;
        p.speed = 80;
        const nextComp = stateRef.current.nextCompound || p.compound;
        p.compound = nextComp;
        p.tyreWear = 0;
        p.tyreAge = 0;
        setUiState(s => ({ ...s, showPitAlert: true, radioMsg: `📻 Pit stop! ${nextComp}s going on. Clear! Clear!` }));
        setTimeout(() => setUiState(s => ({ ...s, showPitAlert: false })), 2000);
      }
    } else {
      // In pit
      p.speed = 60;
      p.pitTimer -= DT * 60;
      if (p.pitTimer <= 0) {
        p.inPit = false;
        p.progress = track.pitLane.exit;
        setUiState(s => ({ ...s, radioMsg: `📻 Good stop! ${p.compound}s on. Go go go!` }));
      }
    }

    // ─ AI cars update ─
    st.cars.forEach(car => {
      if (car.retired) return;
      if (!car.inPit) {
        const skillMult = car.skill;
        const scMult = st.scActive ? 0.6 : 1.0;
        const tyreGrip = Math.max(0.7, 1 - car.tyreWear * 0.003);
        const compoundGrip = COMPOUNDS[car.compound]?.grip || 1;
        const targetSpeed = 295 * skillMult * scMult * tyreGrip * compoundGrip;
        car.speed = car.speed * 0.9 + targetSpeed * 0.1 + (Math.random() - 0.5) * 2;

        const progressRate = (car.speed / 300) * 0.00115 * DT * 60;
        car.progress += progressRate + (Math.random() - 0.5) * 0.0001;

        if (car.progress >= 1) {
          car.progress = 0;
          car.lap += 1;
          car.tyreAge += 1;
          const deg = COMPOUNDS[car.compound]?.deg || 0.025;
          car.tyreWear += deg * 100;

          // AI pit strategy
          if (car.tyreWear > 70 && car.lap < stateRef.current.totalLaps - 5) {
            car.inPit = true;
            car.pitTimer = 200 + Math.random() * 60;
            const compounds = ["Soft", "Medium", "Hard"];
            car.compound = compounds[Math.floor(Math.random() * 3)];
            car.tyreWear = 0;
            car.tyreAge = 0;
          }
        }
      } else {
        car.pitTimer -= DT * 60;
        if (car.pitTimer <= 0) {
          car.inPit = false;
          car.progress = track.pitLane.exit + Math.random() * 0.01;
        }
      }
    });
  }

  // ─── RENDER ─────────────────────────────────────────────────────────────────
  function renderFrame(ctx, W, H, st, frame) {
    // Clear
    ctx.clearRect(0, 0, W, H);

    // Background
    const bgGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
    bgGrad.addColorStop(0, "#0a0a14");
    bgGrad.addColorStop(1, "#060608");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Grid pattern
    ctx.strokeStyle = "rgba(255,255,255,0.015)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const track = st.track;
    const path = track.path;

    // ─ Draw track ─
    const margin = 0.08;
    const pts = path.map(([x, y]) => [x * W * (1 - 2 * margin) + W * margin, y * H * (1 - 2 * margin) + H * margin]);

    // Track shadow/glow
    ctx.shadowColor = track.color;
    ctx.shadowBlur = 20;
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 40;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Tarmac
    ctx.strokeStyle = "#1a1a22";
    ctx.lineWidth = 36;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();

    // Track edge (white lines)
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();

    // Center line (dashed)
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 12]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // DRS zones (highlighted track section)
    const drsZones = track.drsZones || [];
    drsZones.forEach(([start, end]) => {
      ctx.strokeStyle = "rgba(0,200,151,0.25)";
      ctx.lineWidth = 36;
      ctx.beginPath();
      const steps = 40;
      for (let i = 0; i <= steps; i++) {
        const t = start + (end - start) * (i / steps);
        const [x, y] = getPointOnPath(path, t, W, H, margin);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      // DRS label
      const [lx, ly] = getPointOnPath(path, (start + end) / 2, W, H, margin);
      ctx.fillStyle = "rgba(0,200,151,0.7)";
      ctx.font = "bold 9px 'Barlow Condensed'";
      ctx.textAlign = "center";
      ctx.fillText("DRS", lx, ly - 24);
    });

    // Start/finish line
    const [sfx, sfy] = getPointOnPath(path, 0, W, H, margin);
    const sfAngle = getAngleOnPath(path, 0, W, H, margin);
    ctx.save();
    ctx.translate(sfx, sfy);
    ctx.rotate(sfAngle + Math.PI / 2);
    for (let i = -3; i <= 3; i++) {
      ctx.fillStyle = (Math.abs(i) % 2 === 0) ? "#fff" : "#000";
      ctx.fillRect(i * 4 - 2, -18, 4, 6);
      ctx.fillStyle = (Math.abs(i) % 2 === 1) ? "#fff" : "#000";
      ctx.fillRect(i * 4 - 2, -12, 4, 6);
    }
    ctx.restore();

    // Pit lane indicator
    const [px, py] = getPointOnPath(path, track.pitLane.entry, W, H, margin);
    ctx.fillStyle = "rgba(232,0,45,0.8)";
    ctx.font = "bold 9px 'Barlow Condensed'";
    ctx.textAlign = "center";
    ctx.fillText("PIT", px, py - 22);

    // ─ Draw corner markers ─
    (track.corners || []).forEach(corner => {
      const [cx, cy] = getPointOnPath(path, corner.t, W, H, margin);
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
    });

    // ─ Draw AI cars ─
    st.cars.forEach(car => {
      if (car.retired) return;
      const [cx, cy] = getPointOnPath(path, car.progress, W, H, margin);
      const angle = getAngleOnPath(path, car.progress, W, H, margin);

      if (car.inPit) return; // hide in pit

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);

      // Car glow
      ctx.shadowColor = car.color;
      ctx.shadowBlur = 8;

      // Car body
      ctx.fillStyle = car.color;
      ctx.fillRect(-9, -3, 18, 6);

      // Cockpit
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(-2, -2.5, 8, 5);

      // Front wing
      ctx.fillStyle = car.color;
      ctx.fillRect(7, -4, 3, 8);

      // Rear wing
      ctx.fillRect(-10, -4, 2, 8);

      ctx.shadowBlur = 0;
      ctx.restore();

      // Driver label
      if (car.speed > 100) {
        ctx.fillStyle = car.color;
        ctx.font = "bold 8px 'Barlow Condensed'";
        ctx.textAlign = "center";
        ctx.fillText(car.id, cx, cy - 12);
      }
    });

    // ─ Draw Player Car ─
    const pl = st.player;
    if (!pl.inPit) {
      const [plx, ply] = getPointOnPath(path, pl.progress, W, H, margin);
      const plAngle = getAngleOnPath(path, pl.progress, W, H, margin);

      ctx.save();
      ctx.translate(plx, ply);
      ctx.rotate(plAngle);

      // DRS effect
      if (pl.drs) {
        ctx.shadowColor = "#00C897";
        ctx.shadowBlur = 20;
      }

      // Exhaust/speed trail
      const trailLen = Math.floor(pl.speed / 30);
      for (let i = 0; i < trailLen; i++) {
        const t2 = ((pl.progress - i * 0.0005) + 1) % 1;
        const [tx, ty] = getPointOnPath(path, t2, W, H, margin);
        ctx.fillStyle = `rgba(255,100,0,${0.15 - i * 0.015})`;
        ctx.beginPath();
        ctx.arc(tx - plx, ty - ply, 3 - i * 0.3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Car body (bigger, distinct)
      ctx.shadowColor = pl.color;
      ctx.shadowBlur = 15;
      ctx.fillStyle = pl.color;
      ctx.fillRect(-11, -4, 22, 8);

      // Cockpit
      ctx.fillStyle = "rgba(0,0,0,0.8)";
      ctx.fillRect(-1, -3, 9, 6);

      // Front wing
      ctx.fillStyle = pl.color;
      ctx.fillRect(9, -5, 3, 10);

      // Rear wing
      ctx.fillRect(-12, -5, 3, 10);

      // Helmet
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(2, 0, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.restore();

      // Player name tag
      ctx.fillStyle = "rgba(6,6,8,0.8)";
      ctx.fillRect(plx - 18, ply - 24, 36, 14);
      ctx.fillStyle = pl.color;
      ctx.font = "bold 10px 'Barlow Condensed'";
      ctx.textAlign = "center";
      ctx.fillText(`★ ${pl.id}`, plx, ply - 13);
    }

    // ─ SC overlay ─
    if (st.scActive) {
      ctx.fillStyle = "rgba(255,214,0,0.06)";
      ctx.fillRect(0, 0, W, H);
    }

    // ─ Speed lines when pushing ─
    if (pl.mode === "push" && pl.speed > 260 && frame % 3 === 0) {
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 8; i++) {
        const x = Math.random() * W;
        const len = 20 + Math.random() * 60;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x + len * 0.3, H);
        ctx.stroke();
      }
    }

    // ─ Track name watermark ─
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.font = `bold ${Math.min(W / 8, 60)}px 'Barlow Condensed'`;
    ctx.textAlign = "center";
    ctx.fillText(track.name.toUpperCase(), W / 2, H / 2 + 20);
  }

  // ─── SETUP SCREEN ────────────────────────────────────────────────────────────
  if (phase === "setup") {
    return (
      <>
        <style>{css}</style>
        <div className="setup-screen">
          <div className="setup-card fade-in">
            <div className="setup-title">BLAME<span style={{ color: "var(--red)" }}>ENGINE</span></div>
            <div className="setup-sub">F1 Strategy Simulator — 2D Race Mode</div>

            <div className="setup-field">
              <label className="setup-label">Circuit</label>
              <select className="setup-select" value={selectedTrack} onChange={e => setSelectedTrack(e.target.value)}>
                {Object.keys(TRACKS).map(t => <option key={t} value={t}>{TRACKS[t].name}</option>)}
              </select>
            </div>

            <div className="setup-field">
              <label className="setup-label">Starting Compound</label>
              <div style={{ display: "flex", gap: 8 }}>
                {Object.entries(COMPOUNDS).slice(0, 3).map(([name, c]) => (
                  <button key={name} className={`ctrl-btn ${selectedCompound === name ? "active" : ""}`}
                    style={{
                      flex: 1, borderColor: selectedCompound === name ? c.color : undefined,
                      background: selectedCompound === name ? c.color + "33" : undefined,
                      color: selectedCompound === name ? c.color : undefined
                    }}
                    onClick={() => setSelectedCompound(name)}>
                    {name}
                  </button>
                ))}
              </div>
            </div>

            <div className="setup-field">
              <label className="setup-label">Drive As</label>
              <div className="driver-grid">
                {AI_DRIVERS.map(d => (
                  <div key={d.id}
                    className={`driver-card ${selectedDriver === d.id ? "selected" : ""}`}
                    style={{ borderColor: selectedDriver === d.id ? TEAM_COLORS[d.team] : undefined }}
                    onClick={() => setSelectedDriver(d.id)}>
                    <div className="driver-card-id" style={{ color: TEAM_COLORS[d.team] }}>{d.id}</div>
                    <div className="driver-card-name">{d.name.split(" ").pop()}</div>
                  </div>
                ))}
              </div>
            </div>

            <button className="start-btn" disabled={!selectedDriver} onClick={startRace}>
              🏁 LIGHTS OUT — LET'S RACE
            </button>
          </div>
        </div>
      </>
    );
  }

  // ─── RESULTS SCREEN ──────────────────────────────────────────────────────────
  if (phase === "results") {
    const { finalPos, strategyIQ } = uiState;
    const iqColor = strategyIQ >= 85 ? "#00C897" : strategyIQ >= 70 ? "#FFD600" : "#E8002D";
    return (
      <>
        <style>{css}</style>
        <div className="results-overlay fade-in">
          <div className="results-card">
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, letterSpacing: 3, color: "var(--muted)", marginBottom: 8 }}>RACE RESULT</div>
            <div className="results-pos">P{finalPos}</div>
            <div className="results-label">Final Position</div>
            <div style={{ marginBottom: 24 }}>
              <div className="results-row">
                <span style={{ color: "var(--muted)" }}>Strategy IQ</span>
                <span className="results-row-val" style={{ color: iqColor, fontSize: 20 }}>{strategyIQ}</span>
              </div>
              <div className="results-row">
                <span style={{ color: "var(--muted)" }}>Track</span>
                <span className="results-row-val">{TRACKS[selectedTrack].name}</span>
              </div>
              <div className="results-row">
                <span style={{ color: "var(--muted)" }}>Driver</span>
                <span className="results-row-val">{selectedDriver}</span>
              </div>
              <div className="results-row">
                <span style={{ color: "var(--muted)" }}>Starting Tyre</span>
                <span className="results-row-val">{selectedCompound}</span>
              </div>
              <div className="results-row" style={{ border: "none" }}>
                <span style={{ color: "var(--muted)" }}>Rating</span>
                <span className="results-row-val" style={{ color: iqColor }}>
                  {strategyIQ >= 85 ? "ELITE STRATEGIST 🏆" : strategyIQ >= 70 ? "SOLID CALL 👍" : "BOX EARLIER NEXT TIME 📻"}
                </span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="ctrl-btn big" style={{ flex: 1 }} onClick={() => setPhase("setup")}>▶ Race Again</button>
              <button className="ctrl-btn big pit-btn" style={{ flex: 1 }} onClick={() => window.location.reload()}>🏠 Menu</button>
            </div>
          </div>
        </div>
      </>
    );
  }

  // ─── RACING SCREEN ───────────────────────────────────────────────────────────
  const { lap, totalLaps, flag, position, speed, gear, throttle, brake,
    compound, tyreWear, tyreTemp, drs, drsAvailable, fuel, ersCharge,
    inPit, pitRequested, scActive, showSCAlert, showPitAlert,
    radioMsg, drivers, gap_ahead, gap_behind, notifications } = uiState;

  const compData = COMPOUNDS[compound] || COMPOUNDS.Medium;
  const wearColor = tyreWear > 70 ? "#E8002D" : tyreWear > 40 ? "#FFD600" : "#00C897";
  const flagClass = { green: "flag-green", yellow: "flag-yellow", red: "flag-red-f", sc: "flag-sc" }[flag] || "flag-green";
  const flagLabel = { green: "GREEN FLAG", yellow: "YELLOW", red: "RED FLAG", sc: "SAFETY CAR" }[flag];

  return (
    <>
      <style>{css}</style>
      <div className="sim-root">
        {/* TOP BAR */}
        <div className="topbar">
          <div className="topbar-logo">B<span>E</span></div>
          <div className="topbar-sep" />
          <div className="race-info">
            <div className="race-stat">
              <div className="race-stat-v red">P{position}</div>
              <div className="race-stat-l">Position</div>
            </div>
            <div className="race-stat">
              <div className="race-stat-v">{lap}<span style={{ fontSize: 12, color: "var(--muted)" }}>/{totalLaps}</span></div>
              <div className="race-stat-l">Lap</div>
            </div>
            <div className="race-stat">
              <div className="race-stat-v green">{speed}</div>
              <div className="race-stat-l">km/h</div>
            </div>
            {gap_ahead && (
              <div className="race-stat">
                <div className="race-stat-v" style={{ fontSize: 14, color: "var(--blue)" }}>{gap_ahead === "LEADER" ? "P1" : gap_ahead}</div>
                <div className="race-stat-l">Gap Ahead</div>
              </div>
            )}
            {gap_behind && (
              <div className="race-stat">
                <div className="race-stat-v" style={{ fontSize: 14, color: "var(--muted)" }}>{gap_behind}</div>
                <div className="race-stat-l">Behind</div>
              </div>
            )}
            <div className="topbar-sep" />
            <div className={`flag-indicator ${flagClass}`}>
              <div className="dot-pulse" style={{ background: flag === "green" ? "var(--green)" : flag === "sc" ? "#fff" : flag === "red" ? "var(--red)" : "var(--yellow)" }} />
              {flagLabel}
            </div>
            {scActive && <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--yellow)" }}>⚠ SC DEPLOYED</div>}
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
            <button className="ctrl-btn" onClick={togglePause}>⏸ PAUSE</button>
            <span style={{ fontSize: 9, letterSpacing: 1.5, color: "var(--muted)", textTransform: "uppercase" }}>Speed</span>
            {[1, 3, 6, 12].map(s => (
              <button key={s} className="ctrl-btn" style={{ padding: "5px 9px", fontSize: 11 }}
                onClick={() => setGameSpeed(s)}>{s}×</button>
            ))}
          </div>
        </div>

        <div className="main-area">
          {/* CANVAS */}
          <div className="canvas-wrap">
            <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />

            {/* Weather */}
            <div className="weather-overlay">
              <span>☀️</span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 12 }}>28°C · DRY</span>
            </div>

            {/* DRS */}
            <div className={`drs-indicator ${drs ? "drs-on" : drsAvailable ? "drs-available" : "drs-off"}`}
              onClick={toggleDRS} style={{ cursor: "pointer" }}>
              DRS {drs ? "OPEN" : drsAvailable ? "READY" : "—"}
            </div>

            {/* Alerts */}
            {showSCAlert && <div className="sc-alert">⚠ SAFETY CAR</div>}
            {showPitAlert && <div className="pit-alert">🔧 PIT STOP</div>}
            {inPit && <div className="pit-alert" style={{ top: "60%" }}>IN PIT LANE</div>}

            {/* Notifications */}
            {notifications.map(n => (
              <div key={n.id} className="notif" style={{ color: n.color }}>{n.msg}</div>
            ))}

            {/* BOTTOM CONTROLS */}
            <div className="bottom-controls">
              {/* Pedals */}
              <div className="pedals">
                <div className="pedal-wrap">
                  <div className="pedal-bar">
                    <div className="pedal-fill throttle" style={{ height: `${throttle}%` }} />
                  </div>
                  <div className="pedal-label">THR</div>
                </div>
                <div className="pedal-wrap">
                  <div className="pedal-bar">
                    <div className="pedal-fill brake" style={{ height: `${brake}%` }} />
                  </div>
                  <div className="pedal-label">BRK</div>
                </div>
              </div>

              {/* Gear */}
              <div className="gear-display">
                <div className="gear-n">{gear}</div>
                <div className="gear-l">GEAR</div>
              </div>

              {/* Speed */}
              <div className="speed-display">
                <div className="speed-v">{speed}</div>
                <div className="speed-l">KM/H</div>
              </div>

              <div style={{ width: 1, height: 52, background: "var(--border)" }} />

              {/* Mode */}
              <div className="ctrl-group">
                <div className="ctrl-label">Engine Mode</div>
                <div className="ctrl-buttons">
                  {["push", "normal", "conserve"].map(m => (
                    <button key={m} className={`ctrl-btn ${stateRef.current?.player?.mode === m ? "active" : ""}`}
                      onClick={() => setMode(m)}>
                      {m === "push" ? "🔥 PUSH" : m === "normal" ? "⚡ NORMAL" : "💧 SAVE"}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ width: 1, height: 52, background: "var(--border)" }} />

              {/* Pit */}
              <div className="ctrl-group">
                <div className="ctrl-label">Strategy</div>
                <div className="ctrl-buttons">
                  <button
                    className={`ctrl-btn pit-btn ${pitRequested || inPit ? "active" : ""}`}
                    onClick={requestPit} disabled={inPit}>
                    {inPit ? "🔧 IN PIT" : pitRequested ? "⏳ BOX CALLED" : "🔧 BOX BOX"}
                  </button>
                </div>
              </div>

              {/* Compound */}
              <div className="ctrl-group">
                <div className="ctrl-label">Next Tyre</div>
                <div className="ctrl-buttons">
                  {["Soft", "Medium", "Hard"].map(c => (
                    <button key={c} className="ctrl-btn"
                      style={{
                        color: COMPOUNDS[c].color, borderColor: stateRef.current?.nextCompound === c ? COMPOUNDS[c].color : "var(--border)",
                        background: stateRef.current?.nextCompound === c ? COMPOUNDS[c].color + "22" : undefined
                      }}
                      onClick={() => changeTyre(c)}>
                      {COMPOUNDS[c].label}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ width: 1, height: 52, background: "var(--border)" }} />

              {/* DRS button */}
              <div className="ctrl-group">
                <div className="ctrl-label">DRS</div>
                <button className={`ctrl-btn green-btn ${drs ? "active" : ""}`}
                  onClick={toggleDRS} disabled={!drsAvailable}>
                  DRS {drs ? "ON" : "OFF"}
                </button>
              </div>

              {/* ERS */}
              <div className="ctrl-group">
                <div className="ctrl-label">ERS {Math.round(ersCharge)}%</div>
                <button className={`ctrl-btn yellow-btn ${stateRef.current?.player?.ersDeploy ? "active" : ""}`}
                  onClick={deployERS} disabled={ersCharge < 5}>
                  ⚡ DEPLOY
                </button>
              </div>

              {/* Tyre wear */}
              <div className="tyre-status">
                <div style={{ fontFamily: "var(--mono)", fontSize: 9, letterSpacing: 1, color: "var(--muted)", marginBottom: 2 }}>TYRE STATUS</div>
                <div className="tyre-row">
                  <div className="tyre-icon" style={{ background: compData.color }}>{compData.label}</div>
                  <div className="tyre-wear-bar">
                    <div className="tyre-wear-fill" style={{ width: `${100 - tyreWear}%`, background: wearColor }} />
                  </div>
                  <div className="tyre-temp" style={{ color: tyreTemp > 100 ? "var(--red)" : tyreTemp > 90 ? "var(--yellow)" : "var(--green)" }}>
                    {tyreTemp}°
                  </div>
                </div>
                <div className="tyre-row">
                  <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted)" }}>FUEL</span>
                  <div className="tyre-wear-bar">
                    <div className="tyre-wear-fill" style={{ width: `${fuel}%`, background: "var(--blue)" }} />
                  </div>
                  <div className="tyre-temp" style={{ color: "var(--blue)" }}>{Math.round(fuel)}%</div>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL */}
          <div className="right-panel">
            <div className="panel-title">// TIMING TOWER</div>
            <div className="timing-tower">
              {drivers.slice(0, 20).map((d, i) => {
                const isPlayer = d.isPlayer;
                const tColor = TEAM_COLORS[d.team] || "#888";
                const comp = COMPOUNDS[d.compound] || COMPOUNDS.Medium;
                const gap = i === 0 ? "LEADER" : d.inPit ? "PIT" :
                  `+${((drivers[0].lap + drivers[0].progress - d.lap - d.progress) * 90).toFixed(1)}s`;
                return (
                  <div key={d.id} className={`timing-row ${isPlayer ? "player" : ""}`}>
                    <div className={`timing-pos ${i === 0 ? "p1" : i === 1 ? "p2" : i === 2 ? "p3" : ""}`}>{i + 1}</div>
                    <div className="timing-team-bar" style={{ background: tColor }} />
                    <div className="timing-name" style={{ color: isPlayer ? tColor : undefined }}>{d.id}</div>
                    <div className="timing-tyre" style={{ background: comp.color }}>{comp.label}</div>
                    <div className="timing-gap">{gap}</div>
                  </div>
                );
              })}
            </div>

            {/* Radio */}
            <div className="radio-feed">
              <div className="radio-label">📻 ENGINEER RADIO</div>
              {radioMsg}
            </div>

            {/* Minimap */}
            <div className="panel-title">// TRACK MAP</div>
            <div className="minimap" style={{ height: 160, position: "relative", background: "rgba(0,0,0,0.4)" }}>
              <svg width="100%" height="160" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                <polyline
                  points={TRACKS[selectedTrack].path.map(([x, y]) => `${x * 90 + 5},${y * 90 + 5}`).join(" ")}
                  fill="none" stroke="#333" strokeWidth="4" strokeLinejoin="round" strokeLinecap="round"
                />
                <polyline
                  points={TRACKS[selectedTrack].path.map(([x, y]) => `${x * 90 + 5},${y * 90 + 5}`).join(" ")}
                  fill="none" stroke={TRACKS[selectedTrack].color} strokeWidth="1.5"
                  strokeLinejoin="round" strokeLinecap="round" strokeOpacity="0.6"
                />
                {/* AI cars on minimap */}
                {drivers.filter(d => !d.isPlayer && !d.inPit).slice(0, 10).map(d => {
                  const [mx, my] = getPointOnPath(TRACKS[selectedTrack].path, d.progress, 90, 90, 0);
                  return <circle key={d.id} cx={mx + 5} cy={my + 5} r="2" fill={TEAM_COLORS[d.team] || "#888"} opacity="0.7" />;
                })}
                {/* Player on minimap */}
                {(() => {
                  const p = stateRef.current?.player;
                  if (!p) return null;
                  const [mx, my] = getPointOnPath(TRACKS[selectedTrack].path, p.progress, 90, 90, 0);
                  return <circle cx={mx + 5} cy={my + 5} r="4" fill="var(--red)" stroke="#fff" strokeWidth="1" />;
                })()}
              </svg>
            </div>

            {/* Lap info */}
            <div style={{ padding: "10px 14px", borderTop: "1px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 10, letterSpacing: 2, color: "var(--muted)", textTransform: "uppercase" }}>Lap</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700 }}>{lap} / {totalLaps}</span>
              </div>
              <div style={{ height: 3, background: "rgba(255,255,255,0.08)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${(lap / totalLaps) * 100}%`, background: "var(--red)", transition: "width 0.5s", borderRadius: 2 }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
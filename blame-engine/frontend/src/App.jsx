import { useState, useEffect, useRef, useCallback, createContext, useContext } from "react";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

import F1Simulator from './F1Simulator';
import News from "./news";
const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.className = theme === "dark" ? "dark-theme" : "light-theme";
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === "dark" ? "light" : "dark");

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

// ─── MOCK DATA ────────────────────────────────────────────────────────────────
const SEASONS = [2024, 2023, 2022, 2021, 2020];
const GRANDS_PRIX = [
  "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
  "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
  "Austria", "Britain", "Hungary", "Belgium", "Netherlands",
  "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
  "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
];
const DRIVERS = [
  { id: "VER", name: "Max Verstappen", team: "Red Bull Racing", color: "#3671C6" },
  { id: "PER", name: "Sergio Perez", team: "Red Bull Racing", color: "#3671C6" },
  { id: "HAM", name: "Lewis Hamilton", team: "Ferrari", color: "#E8002D" },
  { id: "LEC", name: "Charles Leclerc", team: "Ferrari", color: "#E8002D" },
  { id: "NOR", name: "Lando Norris", team: "McLaren", color: "#FF8000" },
  { id: "PIA", name: "Oscar Piastri", team: "McLaren", color: "#FF8000" },
  { id: "RUS", name: "George Russell", team: "Mercedes", color: "#27F4D2" },
  { id: "ANT", name: "Kimi Antonelli", team: "Mercedes", color: "#27F4D2" },
  { id: "ALO", name: "Fernando Alonso", team: "Aston Martin", color: "#229971" },
  { id: "STR", name: "Lance Stroll", team: "Aston Martin", color: "#229971" },
];

const COMPOUNDS = ["Soft", "Medium", "Hard", "Intermediate", "Wet"];

function generateRaceAutopsy(driver, gp, season) {
  const seed = driver.charCodeAt(0) + gp.length + season;
  const rng = (n) => ((seed * n * 1234567) % 1000) / 1000;
  return {
    qualifying: -(rng(1) * 1.8 + 0.2).toFixed(2),
    tireDeg: -(rng(2) * 2.5 + 0.5).toFixed(2),
    pitExecution: -(rng(3) * 1.2 + 0.1).toFixed(2),
    strategyError: -(rng(4) * 3.1 + 0.0).toFixed(2),
    carPace: -(rng(5) * 2.0 + 0.3).toFixed(2),
    incidentImpact: -(rng(6) * 1.5 + 0.0).toFixed(2),
    verdict: [
      `${driver} lost critical time through an aggressive ${gp} strategy that backfired — the ${season} tyre compounds weren't optimised for the track temps. The pit wall called it too early on lap 23, surrendering the undercut window and costing a net ${(rng(7) * 4 + 1).toFixed(1)}s to the cars ahead.`,
      `A textbook case of strategy destroying raw pace. ${driver}'s car was genuinely P${Math.ceil(rng(8) * 3 + 1)} fast but sequential errors in tyre management and a reactive pit approach meant they finished ${Math.ceil(rng(9) * 5 + 2)} places below potential.`,
      `The data tells a damning story. ${driver}'s qualifying deficit of ${(rng(1) * 1.8 + 0.2).toFixed(2)}s meant always fighting traffic, but the real culprit was the pit stop execution — ${(rng(3) * 1.2 + 0.1).toFixed(2)}s of avoidable loss turned a points finish into damage limitation.`
    ][Math.floor(rng(10) * 3)],
    position: { actual: Math.ceil(rng(11) * 15 + 1), optimal: Math.ceil(rng(12) * 10 + 1) },
    totalLoss: parseFloat((rng(1) * 1.8 + rng(2) * 2.5 + rng(3) * 1.2 + rng(4) * 3.1 + rng(5) * 2.0 + rng(6) * 1.5).toFixed(2)),
  };
}

function generateSimLap(lap, stint, userTyre, push, weather) {
  const baseLapTime = 90 + (stint * 0.3 * (userTyre === "Soft" ? 1.2 : userTyre === "Hard" ? 0.8 : 1.0));
  const weatherFactor = weather === "Rain" ? 8 : weather === "Damp" ? 3 : 0;
  const pushFactor = push ? -0.3 : 0.4;
  return (baseLapTime + pushFactor + weatherFactor + (Math.random() * 0.4 - 0.2)).toFixed(3);
}

const LEADERBOARD_DATA = [
  { rank: 1, user: "strategist_max", iq: 94, wins: 47, circuit: "Monaco", delta: "+2.3s" },
  { rank: 2, user: "pitwall_prophet", iq: 91, wins: 38, circuit: "Spa", delta: "+1.8s" },
  { rank: 3, user: "f1_oracle_x", iq: 88, wins: 35, circuit: "Suzuka", delta: "+1.2s" },
  { rank: 4, user: "undercut_king", iq: 86, wins: 29, circuit: "Silverstone", delta: "+0.9s" },
  { rank: 5, user: "tyre_whisperer", iq: 83, wins: 24, circuit: "Monza", delta: "+0.6s" },
  { rank: 6, user: "safetycar_pro", iq: 81, wins: 21, circuit: "Brazil", delta: "+0.4s" },
  { rank: 7, user: "degradation_god", iq: 79, wins: 18, circuit: "Bahrain", delta: "+0.1s" },
  { rank: 8, user: "delta_hunter_7", iq: 76, wins: 15, circuit: "Hungary", delta: "-0.3s" },
];

// ─── STYLES ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700;800;900&family=Barlow:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --font-display: 'Barlow Condensed', sans-serif;
    --font-body: 'Barlow', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
  }

  .dark-theme {
    --red: #E8002D;
    --red-dark: #b0001f;
    --red-rgb: 232, 0, 45;
    --red-glow: rgba(232, 0, 45, 0.3);
    --bg: #080808;
    --bg-card: #0f0f0f;
    --bg-elevated: #161616;
    --bg-hover: #1e1e1e;
    --border: rgba(255, 255, 255, 0.06);
    --border-bright: rgba(255, 255, 255, 0.12);
    --text: #f0f0f0;
    --text-muted: #888;
    --text-faint: #444;
    --yellow: #FFD700;
    --yellow-rgb: 255, 215, 0;
    --green: #00C897;
    --green-rgb: 0, 200, 151;
    --blue: #4A9EFF;
  }

  .light-theme {
    --red: #0066ff; /* Primary accent is Blue in light mode */
    --red-dark: #0052cc;
    --red-rgb: 0, 102, 255;
    --red-glow: rgba(0, 102, 255, 0.15);
    --bg: #ffffff;
    --bg-card: #f8f9fa;
    --bg-elevated: #f1f3f5;
    --bg-hover: #e9ecef;
    --border: rgba(0, 0, 0, 0.08);
    --border-bright: rgba(0, 0, 0, 0.15);
    --text: #1a1a1a;
    --text-muted: #666;
    --text-faint: #adb5bd;
    --yellow: #fcc419;
    --yellow-rgb: 252, 196, 25;
    --green: #2b8a3e;
    --green-rgb: 43, 138, 62;
    --blue: #0066ff;
  }

  html { scroll-behavior: smooth; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--red); border-radius: 2px; }

  /* NAV */
  .nav {
    position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 32px;
    background: rgba(8,8,8,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
  }
  .nav-logo {
    font-family: var(--font-display);
    font-size: 22px; font-weight: 900; letter-spacing: 2px;
    color: var(--text);
    cursor: pointer;
    display: flex; align-items: center; gap: 8px;
  }
  .nav-logo span { color: var(--red); }
  .nav-logo .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--red);
    box-shadow: 0 0 12px var(--red);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 8px var(--red); }
    50% { box-shadow: 0 0 20px var(--red), 0 0 40px var(--red-glow); }
  }
  .nav-links {
    display: flex; align-items: center; gap: 4px;
  }
  .nav-link {
    padding: 6px 14px;
    font-family: var(--font-display);
    font-size: 13px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    background: none; border: none;
    cursor: pointer; transition: color 0.2s;
    border-radius: 4px;
  }
  .nav-link:hover, .nav-link.active { color: var(--text); background: var(--bg-elevated); }
  .nav-link.active { color: var(--red); }
  .nav-cta {
    padding: 7px 18px;
    background: var(--red);
    color: white;
    font-family: var(--font-display);
    font-size: 13px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase;
    border: none; border-radius: 4px;
    cursor: pointer; transition: all 0.2s;
  }
  .nav-cta:hover { background: #ff1a45; box-shadow: 0 4px 20px var(--red-glow); }

  /* THEME TOGGLE */
  .theme-toggle {
    display: flex; align-items: center; gap: 8px;
    margin-right: 16px; padding: 4px 10px;
    border-radius: 100px; background: var(--bg-card);
    border: 1px solid var(--border); cursor: pointer;
    transition: all 0.2s; user-select: none;
  }
  .theme-toggle:hover { border-color: var(--border-bright); background: var(--bg-hover); }
  .theme-toggle-label {
    font-family: var(--font-mono); font-size: 9px;
    letter-spacing: 1px; text-transform: uppercase; color: var(--text-muted);
  }
  .theme-toggle-switch {
    width: 28px; height: 14px; background: var(--text-faint);
    border-radius: 7px; position: relative; transition: background 0.3s;
  }
  .theme-toggle.active .theme-toggle-switch { background: var(--red); }
  .theme-toggle-dot {
    width: 10px; height: 10px; background: white;
    border-radius: 50%; position: absolute; top: 2px; left: 2px;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }
  .theme-toggle.active .theme-toggle-dot { transform: translateX(14px); }


  /* HERO */
  .hero {
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 80px 24px 60px;
    position: relative; overflow: hidden;
    text-align: center;
  }
  .hero-bg {
    position: absolute; inset: 0; z-index: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 40%, rgba(232,0,45,0.08) 0%, transparent 70%);
  }
  .hero-grid {
    position: absolute; inset: 0;
    background-image: linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black, transparent);
  }
  .hero-stripe {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
  }
  .hero-content { position: relative; z-index: 1; max-width: 900px; }
  .hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px;
    border: 1px solid var(--red);
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 11px; letter-spacing: 2px;
    color: var(--red);
    margin-bottom: 32px;
    background: rgba(232,0,45,0.06);
  }
  .hero-title {
    font-family: var(--font-display);
    font-size: clamp(52px, 8vw, 96px);
    font-weight: 900; line-height: 0.95;
    letter-spacing: -1px;
    text-transform: uppercase;
    margin-bottom: 24px;
  }
  .hero-title .red { color: var(--red); }
  .hero-subtitle {
    font-size: 18px; font-weight: 400;
    color: var(--text-muted); line-height: 1.6;
    max-width: 640px; margin: 0 auto 48px;
  }
  .hero-subtitle strong { color: var(--text); font-weight: 600; }
  .hero-ctas {
    display: flex; flex-wrap: wrap; gap: 12px;
    justify-content: center; margin-bottom: 64px;
  }
  .btn {
    padding: 14px 28px;
    font-family: var(--font-display);
    font-size: 14px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase;
    border: none; border-radius: 3px;
    cursor: pointer; transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: var(--red); color: white;
    box-shadow: 0 0 0 rgba(232,0,45,0);
  }
  .btn-primary:hover {
    background: #ff1a45;
    box-shadow: 0 4px 32px var(--red-glow), 0 0 0 1px var(--red);
    transform: translateY(-1px);
  }
  .btn-secondary {
    background: transparent; color: var(--text);
    border: 1px solid var(--border-bright);
  }
  .btn-secondary:hover {
    background: var(--bg-elevated);
    border-color: var(--text-muted);
    transform: translateY(-1px);
  }
  .btn-ghost {
    background: transparent; color: var(--text-muted);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover { color: var(--text); background: var(--bg-card); border-color: var(--border-bright); }
  .btn-sm { padding: 8px 16px; font-size: 12px; }

  .hero-stats {
    display: flex; gap: 48px; justify-content: center; flex-wrap: wrap;
  }
  .hero-stat { text-align: center; }
  .hero-stat-n {
    font-family: var(--font-display);
    font-size: 42px; font-weight: 900;
    color: var(--text); line-height: 1;
  }
  .hero-stat-n .unit { font-size: 20px; color: var(--red); }
  .hero-stat-l {
    font-size: 12px; font-weight: 500; letter-spacing: 2px;
    text-transform: uppercase; color: var(--text-muted); margin-top: 4px;
  }

  /* SECTIONS */
  .section {
    padding: 80px 24px;
    max-width: 1200px; margin: 0 auto;
  }
  .section-label {
    font-family: var(--font-mono);
    font-size: 11px; letter-spacing: 3px; color: var(--red);
    text-transform: uppercase; margin-bottom: 12px;
  }
  .section-title {
    font-family: var(--font-display);
    font-size: clamp(32px, 5vw, 52px); font-weight: 800;
    text-transform: uppercase; letter-spacing: -0.5px;
    margin-bottom: 40px;
  }

  /* CARDS */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: var(--border-bright); }
  .card-title {
    font-family: var(--font-display);
    font-size: 18px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 8px;
  }
  .card-label {
    font-family: var(--font-mono);
    font-size: 10px; letter-spacing: 2px; color: var(--red);
    text-transform: uppercase; margin-bottom: 8px;
  }

  /* GRID */
  .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  @media (max-width: 768px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
  }

  /* FORM */
  .form-group { margin-bottom: 16px; }
  .form-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 10px; letter-spacing: 2px; color: var(--text-muted);
    text-transform: uppercase; margin-bottom: 6px;
  }
  select, input {
    width: 100%;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    transition: border-color 0.2s;
    appearance: none;
    -webkit-appearance: none;
  }
  select:focus, input:focus {
    outline: none; border-color: var(--red);
    box-shadow: 0 0 0 2px rgba(232,0,45,0.15);
  }
  select option { background: #1a1a1a; }

  /* PAGE LAYOUT */
  .page { padding-top: 56px; min-height: 100vh; }

  /* WATERFALL CHART */
  .waterfall { margin: 24px 0; }
  .waterfall-bar-row {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 8px;
  }
  .waterfall-label {
    font-family: var(--font-mono);
    font-size: 11px; color: var(--text-muted);
    width: 160px; flex-shrink: 0; text-align: right;
  }
  .waterfall-track {
    flex: 1; height: 32px;
    background: var(--bg-elevated);
    border-radius: 3px; position: relative; overflow: hidden;
  }
  .waterfall-fill {
    position: absolute; top: 0; bottom: 0;
    background: var(--red); border-radius: 2px;
    transition: width 1s ease-out;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 8px;
    right: 0;
  }
  .waterfall-fill-text {
    font-family: var(--font-mono); font-size: 11px; color: white;
    font-weight: 600;
  }
  .waterfall-total {
    border-top: 1px solid var(--border-bright); margin-top: 12px; padding-top: 12px;
  }
  .waterfall-total .waterfall-fill { background: var(--yellow); }

  /* VERDICT BOX */
  .verdict-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    border-radius: 6px; padding: 20px 24px;
    margin: 24px 0;
    font-size: 15px; line-height: 1.7;
    color: var(--text-muted);
  }
  .verdict-box strong { color: var(--text); }

  /* SIMULATION */
  .sim-track {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 20px;
    font-family: var(--font-mono);
    font-size: 13px; line-height: 1.8;
    max-height: 320px; overflow-y: auto;
  }
  .sim-lap { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; }
  .sim-lap:hover { background: var(--bg-hover); }
  .lap-num { color: var(--text-muted); width: 60px; }
  .lap-time { color: var(--green); }
  .lap-tyre { color: var(--yellow); }
  .lap-event { color: var(--red); font-size: 11px; }

  /* CONTROLS */
  .sim-controls {
    display: flex; flex-wrap: wrap; gap: 10px;
    margin: 16px 0;
  }
  .control-group {
    display: flex; flex-direction: column; gap: 4px; min-width: 120px;
  }
  .control-label {
    font-family: var(--font-mono); font-size: 10px;
    letter-spacing: 1px; color: var(--text-muted); text-transform: uppercase;
  }
  .toggle-row { display: flex; gap: 8px; }
  .toggle-btn {
    flex: 1; padding: 8px; background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-muted); font-family: var(--font-display);
    font-size: 12px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; cursor: pointer; transition: all 0.15s;
  }
  .toggle-btn.active { background: var(--red); border-color: var(--red); color: white; }
  .toggle-btn:hover:not(.active) { border-color: var(--border-bright); color: var(--text); }

  /* IQ SCORE */
  .iq-score {
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    padding: 32px;
  }
  .iq-ring {
    width: 140px; height: 140px; position: relative;
  }
  .iq-ring svg { transform: rotate(-90deg); }
  .iq-number {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-family: var(--font-display); font-size: 40px; font-weight: 900;
    color: var(--text); text-align: center; line-height: 1;
  }
  .iq-label {
    font-family: var(--font-mono); font-size: 11px;
    letter-spacing: 2px; color: var(--text-muted); text-transform: uppercase;
  }

  /* LEADERBOARD */
  .lb-row {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }
  .lb-row:hover { background: var(--bg-elevated); }
  .lb-rank {
    font-family: var(--font-display); font-size: 20px; font-weight: 900;
    width: 32px; text-align: center;
    color: var(--text-faint);
  }
  .lb-rank.top { color: var(--yellow); }
  .lb-user { flex: 1; font-size: 14px; font-weight: 600; }
  .lb-iq {
    font-family: var(--font-mono); font-size: 13px;
    color: var(--green); width: 40px; text-align: right;
  }
  .lb-delta {
    font-family: var(--font-mono); font-size: 12px;
    color: var(--text-muted); width: 60px; text-align: right;
  }

  /* METRIC TILES */
  .metric-tile {
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px 20px;
  }
  .metric-value {
    font-family: var(--font-display); font-size: 32px; font-weight: 900;
    line-height: 1;
  }
  .metric-value.red { color: var(--red); }
  .metric-value.green { color: var(--green); }
  .metric-value.yellow { color: var(--yellow); }
  .metric-desc { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

  /* TABS */
  .tabs { display: flex; gap: 2px; margin-bottom: 24px; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 10px 20px;
    font-family: var(--font-display); font-size: 13px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    background: none; border: none; color: var(--text-muted);
    cursor: pointer; border-bottom: 2px solid transparent;
    margin-bottom: -1px; transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--red); border-bottom-color: var(--red); }

  /* AUTH */
  .auth-modal {
    position: fixed; inset: 0; z-index: 2000;
    background: rgba(0,0,0,0.8); backdrop-filter: blur(8px);
    display: flex; align-items: center; justify-content: center;
    padding: 24px;
  }
  .auth-card {
    background: var(--bg-card); border: 1px solid var(--border-bright);
    border-radius: 12px; padding: 40px; width: 100%; max-width: 420px;
  }
  .auth-title {
    font-family: var(--font-display); font-size: 28px; font-weight: 800;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 8px;
  }
  .auth-sub { font-size: 14px; color: var(--text-muted); margin-bottom: 28px; }
  .auth-divider { text-align: center; color: var(--text-faint); font-size: 12px; margin: 20px 0; }
  .auth-footer { margin-top: 20px; text-align: center; font-size: 13px; color: var(--text-muted); }
  .auth-footer a { color: var(--red); cursor: pointer; }
  .auth-close {
    position: absolute; top: 16px; right: 16px;
    background: none; border: none; color: var(--text-muted);
    font-size: 20px; cursor: pointer;
  }

  /* PRO BADGE */
  .pro-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 8px; background: rgba(255,215,0,0.1);
    border: 1px solid var(--yellow); border-radius: 3px;
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 1px;
    color: var(--yellow); text-transform: uppercase;
  }

  /* LOADING */
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }
  .skeleton {
    background: linear-gradient(90deg, var(--bg-elevated) 25%, var(--bg-hover) 50%, var(--bg-elevated) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 20px; height: 20px; border: 2px solid var(--border);
    border-top-color: var(--red); border-radius: 50%;
    animation: spin 0.7s linear infinite; display: inline-block;
  }

  /* HEATMAP */
  .heatmap-grid {
    display: grid; grid-template-columns: repeat(10, 1fr); gap: 3px;
  }
  .heatmap-cell {
    aspect-ratio: 1; border-radius: 2px; transition: opacity 0.2s;
  }
  .heatmap-cell:hover { opacity: 0.7; }

  /* TOAST */
  .toast-container {
    position: fixed; top: 72px; right: 24px; z-index: 3000;
    display: flex; flex-direction: column; gap: 8px;
  }
  .toast {
    padding: 12px 20px;
    background: var(--bg-elevated); border: 1px solid var(--border-bright);
    border-radius: 6px; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
    animation: slideIn 0.3s ease;
    min-width: 280px;
  }
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  .toast-success { border-left: 3px solid var(--green); }
  .toast-error { border-left: 3px solid var(--red); }
  .toast-info { border-left: 3px solid var(--blue); }

  /* TAGS */
  .tag {
    display: inline-flex; align-items: center;
    padding: 3px 8px; border-radius: 3px;
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 1px;
    text-transform: uppercase; font-weight: 600;
  }
  .tag-red { background: rgba(232,0,45,0.15); color: var(--red); border: 1px solid rgba(232,0,45,0.3); }
  .tag-green { background: rgba(0,200,151,0.15); color: var(--green); border: 1px solid rgba(0,200,151,0.3); }
  .tag-yellow { background: rgba(255,215,0,0.15); color: var(--yellow); border: 1px solid rgba(255,215,0,0.3); }
  .tag-blue { background: rgba(74,158,255,0.15); color: var(--blue); border: 1px solid rgba(74,158,255,0.3); }

  /* DIVIDER */
  .divider { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

  /* PROGRESS BAR */
  .progress-bar {
    height: 4px; background: var(--bg-elevated); border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--red); border-radius: 2px;
    transition: width 0.8s ease-out;
  }
  .progress-fill.green { background: var(--green); }
  .progress-fill.yellow { background: var(--yellow); }

  /* RACE TIMELINE */
  .timeline { position: relative; padding-left: 24px; }
  .timeline::before {
    content: ''; position: absolute; left: 8px; top: 8px; bottom: 8px;
    width: 1px; background: var(--border);
  }
  .timeline-event {
    position: relative; padding: 12px 0 12px 24px;
    border-bottom: 1px solid var(--border);
  }
  .timeline-event::before {
    content: ''; position: absolute; left: -20px; top: 18px;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--bg); border: 2px solid var(--border);
  }
  .timeline-event.red::before { border-color: var(--red); background: var(--red); }
  .timeline-event.green::before { border-color: var(--green); background: var(--green); }
  .timeline-lap {
    font-family: var(--font-mono); font-size: 10px; color: var(--text-muted); margin-bottom: 2px;
  }
  .timeline-desc { font-size: 13px; }

  /* FEATURE CARDS */
  .feature-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 32px;
    transition: all 0.3s; cursor: pointer;
  }
  .feature-card:hover {
    border-color: var(--red);
    box-shadow: 0 8px 40px rgba(232,0,45,0.1);
    transform: translateY(-2px);
  }
  .feature-icon {
    font-size: 32px; margin-bottom: 16px;
  }
  .feature-title {
    font-family: var(--font-display); font-size: 22px; font-weight: 800;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;
  }
  .feature-desc { font-size: 14px; color: var(--text-muted); line-height: 1.6; }
  .feature-arrow {
    margin-top: 20px; color: var(--red);
    font-family: var(--font-display); font-size: 13px;
    font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    display: flex; align-items: center; gap: 6px;
  }

  /* TEAM COMPARISON */
  .driver-compare {
    display: flex; gap: 3px; margin: 16px 0;
  }
  .driver-col { flex: 1; }
  .driver-header {
    background: var(--bg-elevated); padding: 12px;
    border-radius: 6px 6px 0 0; text-align: center;
    font-family: var(--font-display); font-weight: 700;
    font-size: 15px; letter-spacing: 1px;
  }
  .compare-row {
    display: flex; padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    align-items: center; font-size: 13px;
  }
  .compare-row:nth-child(even) { background: rgba(255,255,255,0.02); }

  /* RESPONSIVE NAV */
  @media (max-width: 640px) {
    .nav-links .nav-link:not(.mobile-keep) { display: none; }
    .nav { padding: 0 16px; }
  }

  /* FOOTER */
  .footer {
    border-top: 1px solid var(--border);
    padding: 40px 32px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 16px;
  }
  .footer-logo {
    font-family: var(--font-display); font-size: 18px; font-weight: 900;
    letter-spacing: 2px;
  }
  .footer-links { display: flex; gap: 24px; flex-wrap: wrap; }
  .footer-link { font-size: 13px; color: var(--text-muted); cursor: pointer; }
  .footer-link:hover { color: var(--text); }
  .footer-copy { font-size: 12px; color: var(--text-faint); }

  /* FIX SELECT ARROW */
  .select-wrapper { position: relative; }
  .select-wrapper::after {
    content: '▼'; position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
    font-size: 9px; color: var(--text-muted); pointer-events: none;
  }

  /* ANIMATED ENTRY */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .fade-up { animation: fadeUp 0.5s ease both; }
  .delay-1 { animation-delay: 0.1s; }
  .delay-2 { animation-delay: 0.2s; }
  .delay-3 { animation-delay: 0.3s; }
  .delay-4 { animation-delay: 0.4s; }
  .delay-5 { animation-delay: 0.5s; }

  /* TRACK STATUS */
  .status-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
  }

  /* TEAM COLORS */
  .team-rb { color: #3671C6; }
  .team-fer { color: #E8002D; }
  .team-mc { color: #FF8000; }
  .team-mer { color: #27F4D2; }
  .team-am { color: #229971; }
`;

// ─── COMPONENTS ───────────────────────────────────────────────────────────────
function IQRing({ score }) {
  const r = 56;
  const circ = 2 * Math.PI * r;
  const fill = (score / 100) * circ;
  const { theme } = useTheme();

  // Theme-aware colors
  const red = theme === 'dark' ? "#E8002D" : "#0066ff"; // red color variable matches blue in light mode
  const green = theme === 'dark' ? "#00C897" : "#2b8a3e";
  const yellow = theme === 'dark' ? "#FFD700" : "#fcc419";

  const color = score >= 85 ? green : score >= 70 ? yellow : red;

  return (
    <div className="iq-ring">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={r} fill="none" stroke="var(--bg-elevated)" strokeWidth="10" />
        <circle
          cx="70" cy="70" r={r} fill="none"
          stroke={color} strokeWidth="10"
          strokeDasharray={`${fill} ${circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 1.2s ease-out" }}
        />
      </svg>
      <div className="iq-number">{score}<br /><span style={{ fontSize: 14, color: "var(--text-muted)", fontWeight: 400 }}>IQ</span></div>
    </div>
  );
}

function WaterfallChart({ data, animate }) {
  const max = Math.max(...data.map(d => Math.abs(d.value)));
  const { theme } = useTheme();

  return (
    <div className="waterfall">
      {data.map((item, i) => {
        const val = Math.abs(item.value);
        const pct = (val / max) * 100;
        const baseColor = item.total ? "var(--yellow)" : item.value > 0 ? "var(--green)" : "var(--red)";
        const glowColor = item.total ? "rgba(var(--yellow-rgb), 0.2)" : item.value > 0 ? "rgba(var(--green-rgb), 0.2)" : "rgba(var(--red-rgb), 0.2)";

        return (
          <div key={i} className={`waterfall-bar-row fade-up`} style={{ animationDelay: `${i * 0.1}s`, marginBottom: item.total ? 0 : 10, marginTop: item.total ? 12 : 0 }}>
            <div className="waterfall-label" style={{
              color: item.total ? "var(--text)" : "var(--text-muted)",
              fontWeight: item.total ? 800 : 400,
              fontSize: item.total ? 12 : 11
            }}>
              {item.label}
            </div>
            <div className="waterfall-track" style={{
              height: item.total ? 36 : 24,
              background: "var(--bg-elevated)",
              borderTop: item.total ? "1px solid var(--border-bright)" : "none"
            }}>
              <div
                className="waterfall-fill"
                style={{
                  width: animate ? `${pct}%` : "0%",
                  background: baseColor,
                  boxShadow: animate && pct > 5 ? `0 0 15px ${glowColor}` : "none",
                  transition: `width 1.5s cubic-bezier(0.19, 1, 0.22, 1) ${i * 0.1}s`,
                  height: "100%"
                }}
              >
                <span className="waterfall-fill-text" style={{
                  fontSize: item.total ? 13 : 11,
                  fontWeight: 700,
                  color: theme === 'dark' || item.total ? '#fff' : 'rgba(255,255,255,0.9)',
                  textShadow: "0 1px 2px rgba(0,0,0,0.3)"
                }}>
                  {item.value}s
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Heatmap({ title, data }) {
  const { theme } = useTheme();
  return (
    <div>
      <div className="card-label">{title}</div>
      <div className="heatmap-grid">
        {data.map((v, i) => (
          <div
            key={i}
            className="heatmap-cell"
            style={{
              background: `rgba(var(--red-rgb), ${v})`,
              border: `1px solid rgba(var(--red-rgb), ${Math.min(v + 0.1, 1)})`,
            }}
            title={`Decision ${i + 1}: ${(v * 100).toFixed(0)}% optimal`}
          />
        ))}
      </div>
    </div>
  );
}


function Toast({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          <span>{t.type === "success" ? "✓" : t.type === "error" ? "✗" : "ℹ"}</span>
          {t.message}
        </div>
      ))}
    </div>
  );
}

// ─── PAGES ────────────────────────────────────────────────────────────────────

function Landing({ setPage }) {
  return (
    <div>
      {/* HERO */}
      <div className="hero">
        <div className="hero-bg" />
        <div className="hero-grid" />
        <div className="hero-stripe" />
        <div className="hero-content">
          <div className="hero-tag fade-up">
            <span className="status-dot" />
            LIVE TELEMETRY INTELLIGENCE
          </div>
          <h1 className="hero-title fade-up delay-1">
            THE<br />
            <span className="red">TARZAN</span><br />
            
          </h1>
          <p className="hero-subtitle fade-up delay-2">
            The first F1 platform where <strong>fans generate intelligence teams actually need.</strong>{" "}
            Race autopsy, strategy simulation, crowdsourced pit wall decisions — powered by real telemetry.
          </p>
          <div className="hero-ctas fade-up delay-3">
            <button className="btn btn-primary" onClick={() => setPage("autopsy")}>
              🔬 Explore Race Autopsy
            </button>
            <button className="btn btn-secondary" onClick={() => setPage("simulator")}>
              🏎 Be The Strategist
            </button>
            <button className="btn btn-ghost" onClick={() => setPage("teams")}>
              🏁 For Teams
            </button>
          </div>
          <div className="hero-stats fade-up delay-4">
            {[
              ["2.4M+", "TELEMETRY LAPS"],
              ["847K", "FAN DECISIONS"],
              ["24", "CIRCUITS"],
              ["99.7%", "DATA ACCURACY"],
            ].map(([n, l]) => (
              <div className="hero-stat" key={l}>
                <div className="hero-stat-n">{n}</div>
                <div className="hero-stat-l">{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* FEATURES */}
      <div className="section">
        
        <h2 className="section-title">Built for the <span style={{ color: "var(--red)" }}>obsessed</span></h2>
        <div className="grid-3">
          {[
            {
              icon: "🔬",
              label: "RACE AUTOPSY",
              title: "Where Did The Time Go?",
              desc: "Six-category quantified breakdown of every second lost — qualifying, tyre deg, pit execution, strategy, car pace, and incidents. Real FastF1 telemetry, AI verdict.",
              action: () => setPage("autopsy"),
              cta: "Analyse a Race",
            },
            {
              icon: "🎮",
              label: "BE THE STRATEGIST",
              title: "You're On The Pit Wall",
              desc: "Replay historical races and make the calls. Pit timing, compound choice, push vs conserve, safety car reactions. Beat the real driver or the model's optimal strategy.",
              action: () => setPage("simulator"),
              cta: "Start Simulating",
            },
            {
              icon: "🏆",
              label: "STRATEGY IQ",
              title: "Your Intelligence Rated",
              desc: "Every decision scored against optimal timing. Global leaderboard. Your simulations feed a crowdsourced model that refines real-world strategy intelligence.",
              action: () => setPage("leaderboard"),
              cta: "View Leaderboard",
            },
          ].map((f) => (
            <div className="feature-card" key={f.label} onClick={f.action}>
              <div className="feature-icon">{f.icon}</div>
              <div className="card-label">{f.label}</div>
              <div className="feature-title">{f.title}</div>
              <p className="feature-desc">{f.desc}</p>
              <div className="feature-arrow">{f.cta} →</div>
            </div>
          ))}
        </div>
      </div>

      {/* TRUST BAR */}
      <div style={{ borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)", padding: "40px 24px", textAlign: "center" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: 2, color: "var(--text-muted)", marginBottom: 20, textTransform: "uppercase" }}>
          // Data Sources
        </div>
        <div style={{ display: "flex", gap: 48, justifyContent: "center", flexWrap: "wrap", alignItems: "center" }}>
          {["FastF1 v3.4", "Ergast API", "OpenF1", "LIVETIMING.formula1.com", "Official FIA Data"].map(s => (
            <div key={s} style={{ fontFamily: "var(--font-display)", fontSize: 15, fontWeight: 700, color: "var(--text-faint)", letterSpacing: 2, textTransform: "uppercase" }}>{s}</div>
          ))}
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="section">
        
        <h2 className="section-title">Real data. <span style={{ color: "var(--red)" }}>Real analysis.</span></h2>
        <div className="grid-2">
          <div>
            {[
              { n: "01", title: "Select your race", desc: "Choose any season, Grand Prix, and driver from our full historical database." },
              { n: "02", title: "Fetch telemetry", desc: "FastF1 pulls lap times, sector splits, tyre data, and timing events in real-time." },
              { n: "03", title: "Compute blame attribution", desc: "Our model quantifies time loss across 6 categories using regression and delta analysis." },
              { n: "04", title: "Get the verdict", desc: "Plain-English AI summary tells you exactly what went wrong and by how much." },
            ].map(step => (
              <div key={step.n} style={{ display: "flex", gap: 20, marginBottom: 28 }}>
                <div style={{ fontFamily: "var(--font-display)", fontSize: 36, fontWeight: 900, color: "var(--text-faint)", lineHeight: 1, flexShrink: 0 }}>{step.n}</div>
                <div>
                  <div style={{ fontFamily: "var(--font-display)", fontSize: 16, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.6 }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="card-label">// LIVE EXAMPLE — VER, MONACO 2023</div>
            <WaterfallChart animate={true} data={[
              { label: "Qualifying Cost", value: -0.42 },
              { label: "Tyre Degradation", value: -1.87 },
              { label: "Pit Execution", value: -0.31 },
              { label: "Strategy Error", value: -0.00 },
              { label: "Car Pace Deficit", value: -0.18 },
              { label: "Incident Impact", value: -0.00 },
              { label: "TOTAL LOSS", value: -2.78, total: true },
            ]} />
          </div>
        </div>
      </div>

      <footer className="footer">
        <div className="footer-logo"><span style={{ color: "var(--red)" }}>TARZAN</span></div>
        <div className="footer-links">
          {["API Docs", "Architecture", "GitHub", "Privacy", "Terms"].map(l => <div key={l} className="footer-link">{l}</div>)}
        </div>
        <div className="footer-copy">© 2025 Tarzan. Data: FastF1, Ergast API, OpenF1.</div>
      </footer>
    </div>
  );
}

function RaceAutopsy({ user, addToast, setShowAuth }) {
  const [season, setSeason] = useState(2023);
  const [gp, setGp] = useState("");
  const [driver, setDriver] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [animate, setAnimate] = useState(false);
  const [tab, setTab] = useState("overview");
  const { theme } = useTheme();

  const analyse = async () => {
    if (!gp || !driver) { addToast("Select a Grand Prix and driver", "error"); return; }
    setLoading(true); setResult(null); setAnimate(false);
    await new Promise(r => setTimeout(r, 1800));
    const d = DRIVERS.find(d => d.id === driver);
    const res = generateRaceAutopsy(driver, gp, season);
    setResult({ ...res, driver: d, gp, season });
    setLoading(false);
    setTimeout(() => setAnimate(true), 100);
    addToast("Telemetry analysis complete", "success");
  };

  const exportPDF = () => {
    if (!user) { setShowAuth(true); return; }
    addToast("Generating PDF report…", "info");
    setTimeout(() => addToast("PDF ready for download", "success"), 1500);
  };

  const shareLink = () => {
    addToast("Shareable link copied to clipboard", "success");
  };

  const waterfallData = result ? [
    { label: "Qualifying Cost", value: parseFloat(result.qualifying) },
    { label: "Tyre Degradation", value: parseFloat(result.tireDeg) },
    { label: "Pit Execution", value: parseFloat(result.pitExecution) },
    { label: "Strategy Error", value: parseFloat(result.strategyError) },
    { label: "Car Pace Deficit", value: parseFloat(result.carPace) },
    { label: "Incident Impact", value: parseFloat(result.incidentImpact) },
    { label: "TOTAL LOSS", value: -result.totalLoss, total: true },
  ] : [];

  const pieColors = theme === 'dark'
    ? ["#E8002D", "#FF8700", "#FFD000", "#00D2BE", "#4A9EFF", "#A259FF"]
    : ["#0066ff", "#ff922b", "#fab005", "#12b886", "#228be6", "#7950f2"];

  return (
    <div className="page">
      <div className="section">
        
        <h2 className="section-title">Where did the <span style={{ color: "var(--red)" }}>time go?</span></h2>

        {/* SELECTOR */}
        <div className="card" style={{ marginBottom: 32 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, alignItems: "end" }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Season</label>
              <div className="select-wrapper">
                <select value={season} onChange={e => setSeason(+e.target.value)}>
                  {SEASONS.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Grand Prix</label>
              <div className="select-wrapper">
                <select value={gp} onChange={e => setGp(e.target.value)}>
                  <option value="">Select Grand Prix…</option>
                  {GRANDS_PRIX.map(g => <option key={g}>{g}</option>)}
                </select>
              </div>
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Driver</label>
              <div className="select-wrapper">
                <select value={driver} onChange={e => setDriver(e.target.value)}>
                  <option value="">Select Driver…</option>
                  {DRIVERS.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </div>
            </div>
            <button className="btn btn-primary" onClick={analyse} disabled={loading} style={{ height: 44 }}>
              {loading ? <><span className="spinner" />Fetching…</> : "🔬 Analyse"}
            </button>
          </div>
        </div>

        {/* LOADING SKELETONS */}
        {loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {[1, 2, 3, 4].map(i => <div key={i} className="skeleton" style={{ height: 40, borderRadius: 6 }} />)}
          </div>
        )}

        {/* RESULTS */}
        {result && (
          <div className="fade-up">
            {/* HEADER */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16, marginBottom: 24 }}>
              <div>
                <div style={{ fontFamily: "var(--font-display)", fontSize: 28, fontWeight: 900, textTransform: "uppercase", letterSpacing: 1 }}>
                  {result.driver.name}
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: 14 }}>
                  {result.season} {result.gp} Grand Prix · <span style={{ color: result.driver.color }}>{result.driver.team}</span>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={shareLink}>🔗 Share</button>
                <button className="btn btn-secondary btn-sm" onClick={exportPDF}>📄 PDF</button>
              </div>
            </div>

            {/* METRIC TILES */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 24 }}>
              <div className="metric-tile">
                <div className="metric-value red">-{result.totalLoss}s</div>
                <div className="metric-desc">Total Time Lost</div>
              </div>
              <div className="metric-tile">
                <div className="metric-value yellow">P{result.position.actual}</div>
                <div className="metric-desc">Actual Finish</div>
              </div>
              <div className="metric-tile">
                <div className="metric-value green">P{result.position.optimal}</div>
                <div className="metric-desc">Optimal Possible</div>
              </div>
              <div className="metric-tile">
                <div className="metric-value">{result.position.actual - result.position.optimal}</div>
                <div className="metric-desc">Positions Lost</div>
              </div>
            </div>

            <div className="tabs">
              {["overview", "breakdown", "timeline", "raw data"].map(t => (
                <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                  {t.toUpperCase()}
                </button>
              ))}
            </div>

            {tab === "overview" && (
              <div>
                <div className="grid-2">
                  <div className="card">
                    <div className="card-label">// TIME LOSS BREAKDOWN</div>
                    <WaterfallChart data={waterfallData} animate={animate} />
                  </div>
                  <div>
                    <div className="card" style={{ marginBottom: 16 }}>
                      <div className="card-label">// AI VERDICT</div>
                      <div className="verdict-box" style={{ margin: 0, marginTop: 8 }}>
                        {result.verdict}
                      </div>
                    </div>
                    <div className="card">
                      <div className="card-label">// BLAME ATTRIBUTION %</div>
                      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                        {[
                          { label: "Strategy Error", val: Math.abs(result.strategyError) },
                          { label: "Tyre Degradation", val: Math.abs(result.tireDeg) },
                          { label: "Car Pace Deficit", val: Math.abs(result.carPace) },
                          { label: "Qualifying Cost", val: Math.abs(result.qualifying) },
                          { label: "Pit Execution", val: Math.abs(result.pitExecution) },
                          { label: "Incident Impact", val: Math.abs(result.incidentImpact) },
                        ].sort((a, b) => b.val - a.val).map((item, i) => {
                          const pct = (item.val / result.totalLoss * 100).toFixed(0);
                          return (
                            <div key={i}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
                                <span style={{ color: "var(--text-muted)" }}>{item.label}</span>
                                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text)" }}>{pct}%</span>
                              </div>
                              <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${pct}%`, transition: "width 1s ease-out" }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === "breakdown" && (
              <div className="fade-up">
                {/* SUMMARY VISUALIZATION */}
                <div className="card" style={{ marginBottom: 24 }}>
                  <div className="card-label">// IMPACT DISTRIBUTION & COMPARISON</div>

                  <div className="grid-2" style={{ marginTop: 24, minHeight: 320 }}>
                    {/* PIE CHART */}
                    <div style={{ height: 320 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                              { name: "Qualifying", value: Math.abs(parseFloat(result.qualifying)) },
                              { name: "Tyre Deg", value: Math.abs(parseFloat(result.tireDeg)) },
                              { name: "Pit Exec", value: Math.abs(parseFloat(result.pitExecution)) },
                              { name: "Strategy", value: Math.abs(parseFloat(result.strategyError)) },
                              { name: "Car Pace", value: Math.abs(parseFloat(result.carPace)) },
                              { name: "Incident", value: Math.abs(parseFloat(result.incidentImpact)) },
                            ].filter(d => d.value > 0)}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                            stroke="none"
                            label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                          >
                            {pieColors.map((color, index) => (
                              <Cell key={`cell-${index}`} fill={color} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
                            itemStyle={{ color: 'var(--text)' }}
                            formatter={(value) => [`${value}s`, 'Loss']}
                          />
                          <Legend
                            layout="horizontal" verticalAlign="bottom" align="center"
                            wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', paddingTop: '20px' }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    {/* BAR CHART */}
                    <div style={{ height: 320 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={[
                            { name: "Qualifying", value: Math.abs(parseFloat(result.qualifying)) },
                            { name: "Tyre Deg", value: Math.abs(parseFloat(result.tireDeg)) },
                            { name: "Pit Exec", value: Math.abs(parseFloat(result.pitExecution)) },
                            { name: "Strategy", value: Math.abs(parseFloat(result.strategyError)) },
                            { name: "Car Pace", value: Math.abs(parseFloat(result.carPace)) },
                            { name: "Incident", value: Math.abs(parseFloat(result.incidentImpact)) },
                          ].sort((a, b) => b.value - a.value)}
                          margin={{ left: 20, right: 30, top: 20, bottom: 20 }}
                        >
                          <XAxis type="number" hide />
                          <YAxis
                            dataKey="name"
                            type="category"
                            width={100}
                            tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                            axisLine={false}
                            tickLine={false}
                          />
                          <Tooltip
                            cursor={{ fill: 'rgba(var(--red-rgb), 0.05)' }}
                            contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
                            itemStyle={{ color: 'var(--red)' }}
                            formatter={(value) => [`${value}s`, 'Time Lost']}
                          />
                          <Bar dataKey="value" fill="var(--red)" radius={[0, 4, 4, 0]} barSize={20} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <hr className="divider" style={{ margin: '24px 0' }} />
                  <div style={{ marginTop: 16 }}>
                    <WaterfallChart data={waterfallData} animate={animate} />
                  </div>
                </div>

                {/* DETAILED CATEGORY ANALYSIS */}
                <div className="card">
                  <div className="card-label">// DETAILED CATEGORY ANALYSIS</div>
                  <div style={{ marginTop: 16 }}>
                    {[
                      { label: "Qualifying Cost", value: result.qualifying, desc: "Time delta from optimal grid position relative to race pace. Includes traffic cost in opening stint." },
                      { label: "Tyre Degradation", value: result.tireDeg, desc: "Lap time delta from compound drop-off vs modelled optimal degrade curve. Corrected for weather." },
                      { label: "Pit Execution", value: result.pitExecution, desc: "Combined stationary time excess plus rejoining traffic cost across all stops." },
                      { label: "Strategy Error", value: result.strategyError, desc: "Delta between chosen strategy and reconstructed optimal pit window using track position model." },
                      { label: "Car Pace Deficit", value: result.carPace, desc: "Fundamental performance gap vs reference car, isolated from driver and strategy effects." },
                      { label: "Incident Impact", value: result.incidentImpact, desc: "Time cost of on-track incidents, yellow flag sectors, and avoidance manoeuvres." },
                    ].map((cat, i) => {
                      const maxVal = Math.max(...[result.qualifying, result.tireDeg, result.pitExecution, result.strategyError, result.carPace, result.incidentImpact].map(v => Math.abs(parseFloat(v))));
                      const barPct = (Math.abs(parseFloat(cat.value)) / (maxVal || 1) * 100);

                      return (
                        <div key={i} style={{ padding: "20px 0", borderBottom: i === 5 ? "none" : "1px solid var(--border)", display: "flex", gap: 24, alignItems: "flex-start" }}>
                          <div style={{ width: 100, flexShrink: 0, textAlign: "right" }}>
                            <div style={{ fontFamily: "var(--font-display)", fontSize: 32, fontWeight: 900, color: "var(--red)", lineHeight: 1 }}>
                              {cat.value}s
                            </div>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "var(--text-faint)", marginTop: 4, letterSpacing: 1 }}>TIME LOST</div>
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: 16, letterSpacing: 0.5, textTransform: "uppercase" }}>{cat.label}</div>
                              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
                                {((Math.abs(cat.value) / result.totalLoss) * 100).toFixed(1)}% OF TOTAL
                              </div>
                            </div>
                            <div style={{ height: 4, background: "rgba(255,255,255,0.04)", marginBottom: 12, borderRadius: 1, overflow: "hidden" }}>
                              <div
                                style={{
                                  height: "100%",
                                  width: animate ? `${barPct}%` : "0%",
                                  background: "var(--red)",
                                  transition: `width 0.8s ease-out ${i * 0.05}s`
                                }}
                              />
                            </div>
                            <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>{cat.desc}</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {tab === "timeline" && (
              <div className="card">
                <div className="card-label">// RACE TIMELINE</div>
                <div className="timeline" style={{ marginTop: 16 }}>
                  {[
                    { lap: 1, desc: "Race start — clean getaway, 0.3s gained on opening lap", type: "green" },
                    { lap: 8, desc: "Tyre deg onset earlier than predicted — pace drop of 0.4s/lap", type: "red" },
                    { lap: 14, desc: "Pit window opens — strategist delays call by 3 laps", type: "" },
                    { lap: 23, desc: `Pit stop: ${Math.abs(result.pitExecution)}s stationary time + traffic rejoining cost`, type: "red" },
                    { lap: 31, desc: "Virtual Safety Car deployed — opportunity missed for free pit", type: "red" },
                    { lap: 47, desc: "Final stint — optimal pace but deficit too large to recover", type: "" },
                    { lap: 55, desc: `Chequered flag: P${result.position.actual}. Potential: P${result.position.optimal}`, type: result.position.actual === result.position.optimal ? "green" : "red" },
                  ].map((ev, i) => (
                    <div key={i} className={`timeline-event ${ev.type}`}>
                      <div className="timeline-lap">LAP {ev.lap}</div>
                      <div className="timeline-desc">{ev.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {tab === "raw data" && (
              <div className="card">
                <div className="card-label">// RAW TELEMETRY SAMPLE (FASTF1)</div>
                <div className="sim-track" style={{ marginTop: 12 }}>
                  <div style={{ color: "var(--text-muted)", marginBottom: 8, fontSize: 11 }}># FastF1 session data — {result.season} {result.gp} GP</div>
                  {Array.from({ length: 30 }, (_, i) => {
                    const lap = i + 1;
                    const base = 88 + Math.random() * 3;
                    return (
                      <div key={i} className="sim-lap">
                        <span className="lap-num">LAP {String(lap).padStart(2, "0")}</span>
                        <span className="lap-time">{base.toFixed(3)}s</span>
                        <span className="lap-tyre">{lap < 25 ? "SOFT" : "MEDIUM"}</span>
                        <span style={{ color: "#888", fontSize: 11 }}>S1:{(base * 0.28).toFixed(2)} S2:{(base * 0.38).toFixed(2)} S3:{(base * 0.34).toFixed(2)}</span>
                        <span style={{ color: "#555", fontSize: 11 }}>{lap === 23 ? "PIT" : lap === 31 ? "VSC" : "—"}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {!loading && !result && (
          <div style={{ textAlign: "center", padding: "80px 24px", color: "var(--text-muted)" }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🔬</div>
            <div style={{ fontFamily: "var(--font-display)", fontSize: 20, fontWeight: 700, letterSpacing: 1 }}>
              SELECT A RACE TO BEGIN ANALYSIS
            </div>
            <div style={{ fontSize: 14, marginTop: 8 }}>Choose a season, Grand Prix, and driver above</div>
          </div>
        )}
      </div>
    </div>
  );
}

function Simulator({ user, addToast, setShowAuth }) {
  const [setup, setSetup] = useState({ circuit: "", season: 2023, driver: "" });
  const [phase, setPhase] = useState("setup"); // setup | racing | results
  const [lap, setLap] = useState(0);
  const [laps, setLaps] = useState([]);
  const [tyre, setTyre] = useState("Soft");
  const [push, setPush] = useState(true);
  const [weather, setWeather] = useState("Dry");
  const [stintLaps, setStintLaps] = useState(0);
  const [position, setPosition] = useState(5);
  const [totalTime, setTotalTime] = useState(0);
  const [score, setScore] = useState(null);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef(null);
  const TOTAL_LAPS = 30;

  const startRace = () => {
    if (!setup.circuit || !setup.driver) { addToast("Select circuit and driver", "error"); return; }
    if (!user) { setShowAuth(true); return; }
    setPhase("racing");
    setLap(0); setLaps([]); setStintLaps(0);
    setTyre("Soft"); setPush(true); setWeather("Dry");
    setPosition(5); setTotalTime(0); setRunning(false);
    addToast("Race simulation started — you're on the pit wall!", "info");
  };

  const runNextLap = useCallback(() => {
    setLap(prev => {
      const next = prev + 1;
      if (next > TOTAL_LAPS) { setRunning(false); return prev; }

      const lapTime = generateSimLap(next, Math.floor(stintLaps / 5), tyre, push, weather);
      const event = next === 10 ? "⚠️ VSC" : next === 18 ? "🌧 Rain Start" : next === 22 ? "🚗 Safety Car" : null;
      if (event === "🌧 Rain Start") setWeather("Rain");

      const posDelta = push ? (Math.random() > 0.7 ? -1 : 0) : 0;
      setPosition(p => Math.max(1, Math.min(20, p + posDelta)));
      setTotalTime(t => t + parseFloat(lapTime));
      setStintLaps(s => s + 1);
      setLaps(l => [...l, { lap: next, time: lapTime, tyre, push, weather, event }]);

      if (next === TOTAL_LAPS) {
        setRunning(false);
        setTimeout(() => finishRace(), 500);
      }
      return next;
    });
  }, [tyre, push, weather, stintLaps]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(runNextLap, 400);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [running, runNextLap]);

  const pit = () => {
    if (!running) return;
    setStintLaps(0);
    addToast(`📻 Box box! Pitting for ${tyre}`, "info");
    setLaps(l => [...l, { lap: lap, time: "22.4 (PIT)", tyre, push: false, weather, event: "🔧 PIT STOP" }]);
  };

  const finishRace = () => {
    const iq = Math.floor(60 + Math.random() * 35);
    const realDriverTime = totalTime * (0.95 + Math.random() * 0.08);
    const optimalTime = totalTime * (0.93 + Math.random() * 0.04);
    setScore({ iq, realTime: realDriverTime, userTime: totalTime, optimalTime, finalPos: position });
    setPhase("results");
    addToast(`Race complete! Strategy IQ: ${iq}`, iq >= 80 ? "success" : "info");
  };

  if (phase === "results" && score) {
    const delta = (score.userTime - score.realTime).toFixed(1);
    const vsOptimal = (score.userTime - score.optimalTime).toFixed(1);
    return (
      <div className="page">
        <div className="section">
          
          <h2 className="section-title">How did you <span style={{ color: "var(--red)" }}>compare?</span></h2>
          <div className="grid-2" style={{ marginBottom: 32 }}>
            <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
              <IQRing score={score.iq} />
              <div className="iq-label" style={{ marginTop: 8 }}>Strategy IQ Score</div>
              <div style={{ marginTop: 16, textAlign: "center" }}>
                {score.iq >= 85 && <div className="tag tag-green" style={{ fontSize: 12 }}>ELITE STRATEGIST</div>}
                {score.iq >= 70 && score.iq < 85 && <div className="tag tag-yellow" style={{ fontSize: 12 }}>COMPETENT</div>}
                {score.iq < 70 && <div className="tag tag-red" style={{ fontSize: 12 }}>NEEDS WORK</div>}
              </div>
            </div>
            <div className="card">
              <div className="card-label">// PERFORMANCE DELTA</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 16 }}>
                <div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>vs Real Driver</div>
                  <div style={{ fontFamily: "var(--font-display)", fontSize: 36, fontWeight: 900, color: parseFloat(delta) < 0 ? "var(--green)" : "var(--red)" }}>
                    {parseFloat(delta) < 0 ? "+" : ""}{-parseFloat(delta).toFixed(1)}s {parseFloat(delta) < 0 ? "FASTER" : "SLOWER"}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>vs Optimal Strategy</div>
                  <div style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 800, color: "var(--text-muted)" }}>
                    +{Math.abs(parseFloat(vsOptimal)).toFixed(1)}s from perfect
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>Final Position</div>
                  <div style={{ fontFamily: "var(--font-display)", fontSize: 36, fontWeight: 900 }}>P{score.finalPos}</div>
                </div>
              </div>
            </div>
          </div>
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="card-label">// IQ BREAKDOWN</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginTop: 16 }}>
              {[
                { label: "Pit Timing", score: Math.floor(40 + Math.random() * 60) },
                { label: "Tyre Efficiency", score: Math.floor(40 + Math.random() * 60) },
                { label: "Race Outcome", score: Math.floor(40 + Math.random() * 60) },
                { label: "SC Reaction", score: Math.floor(40 + Math.random() * 60) },
              ].map(s => (
                <div key={s.label} className="metric-tile" style={{ textAlign: "center" }}>
                  <div style={{ fontFamily: "var(--font-display)", fontSize: 28, fontWeight: 900, color: s.score >= 70 ? "var(--green)" : s.score >= 50 ? "var(--yellow)" : "var(--red)" }}>{s.score}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4, textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", gap: 12 }}>
            <button className="btn btn-primary" onClick={() => setPhase("setup")}>▶ Race Again</button>
            <button className="btn btn-secondary" onClick={() => addToast("Simulation saved to leaderboard", "success")}>🏆 Submit Score</button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === "racing") {
    return (
      <div className="page">
        <div className="section">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
            <div>
              
              <h2 style={{ fontFamily: "var(--font-display)", fontSize: 28, fontWeight: 900, textTransform: "uppercase" }}>
                {DRIVERS.find(d => d.id === setup.driver)?.name || "Driver"}
              </h2>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 24, color: "var(--red)", fontWeight: 700 }}>
                LAP {lap}/{TOTAL_LAPS}
              </div>
              <div className="tag tag-green" style={{ fontSize: 13 }}>P{position}</div>
            </div>
          </div>

          <div className="grid-2" style={{ marginBottom: 20 }}>
            {/* CONTROLS */}
            <div className="card">
              <div className="card-label">// PIT WALL CONTROLS</div>
              <div className="sim-controls" style={{ marginTop: 12 }}>
                <div className="control-group">
                  <div className="control-label">Compound</div>
                  <div className="toggle-row">
                    {["Soft", "Medium", "Hard"].map(c => (
                      <button key={c} className={`toggle-btn ${tyre === c ? "active" : ""}`} onClick={() => setTyre(c)}>
                        {c[0]}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="control-group">
                  <div className="control-label">Mode</div>
                  <div className="toggle-row">
                    <button className={`toggle-btn ${push ? "active" : ""}`} onClick={() => setPush(true)}>PUSH</button>
                    <button className={`toggle-btn ${!push ? "active" : ""}`} onClick={() => setPush(false)}>SAVE</button>
                  </div>
                </div>
                <div className="control-group">
                  <div className="control-label">Weather</div>
                  <div className="toggle-row">
                    <button className={`toggle-btn ${weather === "Dry" ? "active" : ""}`} onClick={() => setWeather("Dry")}>DRY</button>
                    <button className={`toggle-btn ${weather === "Rain" ? "active" : ""}`} onClick={() => setWeather("Rain")}>WET</button>
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <button
                  className="btn btn-primary"
                  style={{ flex: 1 }}
                  onClick={() => setRunning(r => !r)}
                >
                  {running ? "⏸ Pause" : "▶ Run"}
                </button>
                <button className="btn btn-secondary" style={{ flex: 1 }} onClick={pit} disabled={!running}>
                  🔧 Box Box!
                </button>
              </div>
              {stintLaps > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Tyre Age: {stintLaps} laps</div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{
                      width: `${Math.min(100, stintLaps / 25 * 100)}%`,
                      background: stintLaps > 20 ? "var(--red)" : stintLaps > 12 ? "var(--yellow)" : "var(--green)",
                      transition: "width 0.3s"
                    }} />
                  </div>
                </div>
              )}
            </div>
            {/* STANDINGS */}
            <div className="card">
              <div className="card-label">// LIVE STANDINGS (REAL DATA)</div>
              <div style={{ marginTop: 8 }}>
                {DRIVERS.slice(0, 6).map((d, i) => {
                  const isUser = d.id === setup.driver;
                  const pos = i < position - 1 ? i + 1 : i + 2;
                  const gap = i === 0 ? "Leader" : `+${((i * 1.2 + Math.random() * 2)).toFixed(2)}s`;
                  return (
                    <div key={d.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--border)", background: isUser ? "rgba(232,0,45,0.05)" : "none" }}>
                      <div style={{ fontFamily: "var(--font-display)", fontWeight: 900, fontSize: 18, width: 28, color: i === 0 ? "var(--yellow)" : "var(--text-faint)" }}>
                        {isUser ? position : (pos <= position ? pos : pos - 1)}
                      </div>
                      <div style={{ width: 4, height: 24, background: d.color, borderRadius: 2 }} />
                      <div style={{ flex: 1, fontSize: 13, fontWeight: isUser ? 700 : 400 }}>
                        {isUser ? "★ " : ""}{d.id}
                      </div>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-muted)" }}>{gap}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* LAP LOG */}
          <div className="card">
            <div className="card-label">// LAP LOG</div>
            <div className="sim-track" style={{ marginTop: 8 }}>
              {laps.length === 0 && (
                <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 20 }}>Press ▶ Run to start the race</div>
              )}
              {[...laps].reverse().map((l, i) => (
                <div key={i} className="sim-lap">
                  <span className="lap-num">L{l.lap}</span>
                  <span className="lap-time">{l.time}s</span>
                  <span className="lap-tyre" style={{ fontSize: 10 }}>{l.tyre.toUpperCase()}</span>
                  <span style={{ color: l.push ? "var(--red)" : "var(--blue)", fontSize: 10 }}>{l.push ? "PUSH" : "SAVE"}</span>
                  {l.event && <span className="lap-event">{l.event}</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="section">
        
        <h2 className="section-title">You're on the <span style={{ color: "var(--red)" }}>pit wall.</span></h2>
        <div className="grid-2">
          <div>
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-label">// RACE SETUP</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 12 }}>
                <div>
                  <label className="form-label">Circuit</label>
                  <div className="select-wrapper">
                    <select value={setup.circuit} onChange={e => setSetup(s => ({ ...s, circuit: e.target.value }))}>
                      <option value="">Select Circuit…</option>
                      {GRANDS_PRIX.map(g => <option key={g}>{g}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="form-label">Season</label>
                  <div className="select-wrapper">
                    <select value={setup.season} onChange={e => setSetup(s => ({ ...s, season: +e.target.value }))}>
                      {SEASONS.map(s => <option key={s}>{s}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="form-label">Driver to Replace</label>
                  <div className="select-wrapper">
                    <select value={setup.driver} onChange={e => setSetup(s => ({ ...s, driver: e.target.value }))}>
                      <option value="">Select Driver…</option>
                      {DRIVERS.map(d => <option key={d.id} value={d.id}>{d.name} ({d.team})</option>)}
                    </select>
                  </div>
                </div>
                <button className="btn btn-primary" onClick={startRace}>
                  🏎 Start Simulation
                </button>
                {!user && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center" }}>
                    Requires login to submit scores
                  </div>
                )}
              </div>
            </div>
          </div>
          <div>
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-label">// SIMULATION ENGINE</div>
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 10 }}>
                {[
                  ["🏎", "Real telemetry for all other cars"],
                  ["📊", "Tyre degradation curves per compound"],
                  ["🔧", "Undercut/overcut delta model"],
                  ["🚗", "Safety car compression model"],
                  ["🌧", "Weather pace shift simulation"],
                  ["🧠", "Crowdsourced strategy weighting"],
                ].map(([icon, text]) => (
                  <div key={text} style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 13, color: "var(--text-muted)" }}>
                    <span>{icon}</span> {text}
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-label">// YOUR DECISIONS</div>
              <div style={{ marginTop: 8 }}>
                {[
                  { d: "Pit Timing", desc: "Choose your pit window" },
                  { d: "Compound Choice", desc: "Soft / Medium / Hard / Inter" },
                  { d: "Push vs Conserve", desc: "Manage pace and deg" },
                  { d: "SC Reaction", desc: "Pit or stay out under safety car" },
                ].map(item => (
                  <div key={item.d} style={{ padding: "8px 0", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>{item.d}</div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{item.desc}</div>
                    </div>
                    <span className="tag tag-blue">LIVE</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Leaderboard() {
  const [filter, setFilter] = useState("all");
  return (
    <div className="page">
      <div className="section">
        
        <h2 className="section-title">Strategy <span style={{ color: "var(--red)" }}>Intelligence</span> Rankings</h2>
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {["all", "week", "month", "monaco", "spa", "suzuka"].map(f => (
            <button key={f} className={`toggle-btn ${filter === f ? "active" : ""}`} style={{ width: "auto", padding: "8px 16px" }} onClick={() => setFilter(f)}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="grid-2">
          <div className="card" style={{ padding: 0 }}>
            {LEADERBOARD_DATA.map((row, i) => (
              <div key={i} className="lb-row">
                <div className={`lb-rank ${i < 3 ? "top" : ""}`}>{i < 3 ? ["🥇", "🥈", "🥉"][i] : row.rank}</div>
                <div>
                  <div className="lb-user">{row.user}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Best: {row.circuit}</div>
                </div>
                <div style={{ flex: 1 }} />
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>{row.wins}w</span>
                  <div className="lb-delta" style={{ color: parseFloat(row.delta) > 0 ? "var(--green)" : "var(--red)" }}>{row.delta}</div>
                  <div className="lb-iq">{row.iq}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="card">
              <div className="card-label">// CROWDSOURCED MODEL</div>
              <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.7, marginTop: 8 }}>
                Every simulation you run feeds the strategy intelligence model. Decisions that outperform real-world outcomes are weighted higher, gradually refining optimal pit windows per circuit.
              </div>
              <hr className="divider" style={{ margin: "16px 0" }} />
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  ["Monaco Optimal Pit", "Lap 22-26 window"],
                  ["Spa Undercut Delta", "+1.4s advantage"],
                  ["Bahrain Compound", "Medium starter: +3%"],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                    <span style={{ color: "var(--text-muted)" }}>{k}</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--green)" }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-label">// HEATMAP — PIT DECISIONS</div>
              <div style={{ marginTop: 12 }}>
                <Heatmap title="Lap Pit Frequency" data={Array.from({ length: 50 }, (_, i) => {
                  const peak1 = Math.exp(-Math.pow(i - 12, 2) / 20);
                  const peak2 = Math.exp(-Math.pow(i - 25, 2) / 25);
                  return Math.min(1, (peak1 + peak2) * 1.2);
                })} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TeamDashboard({ user, setShowAuth }) {
  const [locked] = useState(!user?.isPro);
  const [tab, setTab] = useState("compare");

  if (locked) {
    return (
      <div className="page">
        <div className="section" style={{ textAlign: "center", paddingTop: 120 }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔒</div>
          <div className="pro-badge" style={{ fontSize: 13, padding: "6px 14px", marginBottom: 16 }}>
            ⚡ PRO TEAMS ONLY
          </div>
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: 40, fontWeight: 900, textTransform: "uppercase", marginBottom: 16 }}>
            Professional Team Dashboard
          </h2>
          <p style={{ color: "var(--text-muted)", maxWidth: 480, margin: "0 auto 32px", lineHeight: 1.6 }}>
            Access cross-driver race autopsy comparisons, aggregated fan decision heatmaps, driver vs fan performance gap analytics, and automated blame report API.
          </p>
          <button className="btn btn-primary" onClick={() => setShowAuth(true)}>
            Contact Team Sales
          </button>
        </div>
      </div>
    );
  }

  const demoData = DRIVERS.slice(0, 4).map(d => ({
    ...d,
    ...generateRaceAutopsy(d.id, "Monaco", 2023)
  }));

  return (
    <div className="page">
      <div className="section">
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          
          <div className="pro-badge">⚡ PRO</div>
        </div>
        <h2 className="section-title">Intelligence <span style={{ color: "var(--red)" }}>Command</span></h2>

        <div className="tabs">
          {["compare", "heatmaps", "api"].map(t => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>

        {tab === "compare" && (
          <div>
            <div style={{ marginBottom: 16, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
              Cross-driver autopsy comparison — 2023 Monaco GP
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid var(--border)" }}>
                    <th style={{ textAlign: "left", padding: "10px 12px", fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: 2, color: "var(--text-muted)" }}>DRIVER</th>
                    {["QUALIFYING", "TYRE DEG", "PIT EXEC", "STRATEGY", "CAR PACE", "INCIDENT", "TOTAL"].map(h => (
                      <th key={h} style={{ textAlign: "right", padding: "10px 12px", fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: 1.5, color: "var(--text-muted)" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {demoData.map((d, i) => (
                    <tr key={d.id} style={{ borderBottom: "1px solid var(--border)", background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                      <td style={{ padding: "10px 12px", display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 3, height: 20, background: d.color, borderRadius: 2 }} />
                        <div>
                          <div style={{ fontWeight: 700 }}>{d.id}</div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{d.team}</div>
                        </div>
                      </td>
                      {[d.qualifying, d.tireDeg, d.pitExecution, d.strategyError, d.carPace, d.incidentImpact, -d.totalLoss].map((v, j) => (
                        <td key={j} style={{ textAlign: "right", padding: "10px 12px", fontFamily: "var(--font-mono)", color: parseFloat(v) < -1 ? "var(--red)" : parseFloat(v) < 0 ? "var(--yellow)" : "var(--green)" }}>
                          {v}s
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === "heatmaps" && (
          <div className="grid-2">
            <div className="card">
              <div className="card-label">// FAN PIT DECISION HEATMAP</div>
              <div style={{ marginTop: 12, marginBottom: 8, fontSize: 12, color: "var(--text-muted)" }}>When fans chose to pit — Monaco 2023</div>
              <Heatmap title="Lap Distribution" data={Array.from({ length: 60 }, (_, i) => {
                const peak = Math.exp(-Math.pow(i - 23, 2) / 30);
                return Math.min(1, peak * 1.3 + Math.random() * 0.1);
              })} />
            </div>
            <div className="card">
              <div className="card-label">// COMPOUND CHOICE DISTRIBUTION</div>
              <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 10 }}>
                {[["Soft", 62, "var(--red)"], ["Medium", 28, "var(--yellow)"], ["Hard", 10, "var(--text-faint)"]].map(([c, pct, color]) => (
                  <div key={c}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                      <span>{c}</span>
                      <span style={{ fontFamily: "var(--font-mono)", color }}>{pct}%</span>
                    </div>
                    <div className="progress-bar">
                      <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 2, transition: "width 0.8s ease-out" }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-label">// DRIVER vs FAN PERFORMANCE GAP</div>
              <div style={{ marginTop: 12 }}>
                {DRIVERS.slice(0, 5).map(d => {
                  const gap = (Math.random() * 4 - 1.5).toFixed(2);
                  return (
                    <div key={d.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                      <div style={{ width: 36, fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 14 }}>{d.id}</div>
                      <div className="progress-bar" style={{ flex: 1 }}>
                        <div style={{ height: "100%", width: `${50 + parseFloat(gap) * 10}%`, background: parseFloat(gap) > 0 ? "var(--green)" : "var(--red)", transition: "width 0.8s" }} />
                      </div>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, width: 60, textAlign: "right", color: parseFloat(gap) > 0 ? "var(--green)" : "var(--red)" }}>
                        {parseFloat(gap) > 0 ? "+" : ""}{gap}s
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="card">
              <div className="card-label">// CROWDSOURCED INSIGHTS</div>
              <div style={{ marginTop: 12, fontSize: 13, lineHeight: 1.7, color: "var(--text-muted)" }}>
                <strong style={{ color: "var(--text)" }}>847,293 fan decisions</strong> across 24 circuits have refined our pit window model. Current confidence: <span style={{ color: "var(--green)" }}>94.2%</span> accuracy vs real-world outcomes.
              </div>
              <hr className="divider" style={{ margin: "12px 0" }} />
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[["Undercut Window", "3 laps earlier optimal", "87%"], ["SC Reaction", "Pit if gap > 18s", "91%"], ["Wet→Dry Switch", "Lap after VSC end", "83%"]].map(([k, v, c]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <span style={{ color: "var(--text-muted)" }}>{k}</span>
                    <span style={{ color: "var(--text)" }}>{v}</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--green)" }}>{c}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "api" && (
          <div className="card">
            <div className="card-label">// API DOCUMENTATION</div>
            <div className="sim-track" style={{ marginTop: 12 }}>
              <div style={{ color: "#888", marginBottom: 12 }}># BlameEngine REST API — v1.0</div>
              {[
                ["GET", "/api/v1/autopsy/{year}/{gp}/{driver}", "Automated blame report"],
                ["GET", "/api/v1/optimal/{circuit}", "Crowdsourced optimal strategy"],
                ["POST", "/api/v1/simulate", "Run strategy simulation"],
                ["GET", "/api/v1/leaderboard", "Global IQ rankings"],
                ["GET", "/api/v1/telemetry/{year}/{gp}/{driver}", "Raw telemetry cache"],
                ["GET", "/api/v1/heatmap/{circuit}/{year}", "Fan decision heatmap"],
              ].map(([method, path, desc]) => (
                <div key={path} style={{ display: "flex", gap: 12, padding: "6px 0", borderBottom: "1px solid var(--border)", alignItems: "center" }}>
                  <span style={{ color: method === "GET" ? "var(--blue)" : "var(--yellow)", width: 40, fontFamily: "var(--font-mono)", fontSize: 11 }}>{method}</span>
                  <span style={{ color: "var(--green)", flex: 1, fontFamily: "var(--font-mono)", fontSize: 12 }}>{path}</span>
                  <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{desc}</span>
                </div>
              ))}
              <div style={{ marginTop: 20, color: "#888" }}># Example response — /api/v1/autopsy/2023/Monaco/VER</div>
              <pre style={{ color: "var(--text)", fontSize: 12, marginTop: 8, whiteSpace: "pre-wrap" }}>{JSON.stringify({
                driver: "VER", gp: "Monaco", season: 2023,
                blame: { qualifying: -0.42, tyre_deg: -1.87, pit_execution: -0.31, strategy_error: 0.00, car_pace: -0.18, incident: 0.00 },
                total_loss: -2.78, actual_position: 1, optimal_position: 1,
                verdict: "Verstappen's Monaco was effectively damage limitation…",
              }, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function AuthModal({ mode, setMode, onClose, onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  const submit = () => {
    if (!email || !password) return;
    onLogin({ email, name: name || email.split("@")[0], isPro: email.includes("team") });
  };

  return (
    <div className="auth-modal" onClick={onClose}>
      <div className="auth-card" style={{ position: "relative" }} onClick={e => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose}>✕</button>
        <div className="auth-title">{mode === "login" ? "Sign In" : "Join Tarzan"}</div>
        <div className="auth-sub">{mode === "login" ? "Access your pit wall." : "Start generating intelligence."}</div>
        {mode === "signup" && (
          <div className="form-group">
            <label className="form-label">Name</label>
            <input placeholder="Your name" value={name} onChange={e => setName(e.target.value)} />
          </div>
        )}
        <div className="form-group">
          <label className="form-label">Email</label>
          <input type="email" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="form-label">Password</label>
          <input type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }} onClick={submit}>
          {mode === "login" ? "Sign In" : "Create Account"}
        </button>
        <div className="auth-divider">or</div>
        <button className="btn btn-secondary" style={{ width: "100%", justifyContent: "center" }} onClick={() => onLogin({ email: "demo@blameengine.io", name: "Demo User", isPro: false })}>
          Continue as Demo
        </button>
        <div className="auth-footer">
          {mode === "login" ? <>No account? <a onClick={() => setMode("signup")}>Sign up</a></> : <>Already have one? <a onClick={() => setMode("login")}>Sign in</a></>}
        </div>
      </div>
    </div>
  );
}

// ─── APP ─────────────────────────────────────────────────────────────────────
function AppContent() {
  const [page, setPage] = useState("home");
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [toasts, setToasts] = useState([]);
  const { theme, toggleTheme } = useTheme();

  const addToast = useCallback((message, type = "info") => {
    const id = Date.now();
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3500);
  }, []);

  const handleLogin = (u) => {
    setUser(u);
    setShowAuth(false);
    addToast(`Welcome back, ${u.name}!`, "success");
  };

  const navItems = [
    { id: "autopsy", label: "Race Autopsy" },
    { id: "simulator", label: "Strategist" },
    { id: "leaderboard", label: "Leaderboard" },
    { id: "teams", label: "For Teams" },
    {id: "news", label: "WHAT'S NEW" },
  ];

  return (
    <>
      <style>{css}</style>
      <Toast toasts={toasts} />

      <nav className="nav">
        <div className="nav-logo" onClick={() => setPage("home")}>
          <div className="dot" />
          <span>TARZAN</span>
        </div>
        <div className="nav-links">
          {navItems.map(item => (
            <button key={item.id} className={`nav-link ${page === item.id ? "active" : ""}`} onClick={() => setPage(item.id)}>
              {item.label}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center" }}>
          <div className={`theme-toggle ${theme === 'light' ? 'active' : ''}`} onClick={toggleTheme}>
            <span className="theme-toggle-label">{theme === 'dark' ? 'Dark' : 'Light'}</span>
            <div className="theme-toggle-switch">
              <div className="theme-toggle-dot" />
            </div>
          </div>

          {user ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {user.isPro && <span className="pro-badge">⚡ PRO</span>}
              <div style={{ fontFamily: "var(--font-display)", fontSize: 13, fontWeight: 700, cursor: "pointer", padding: "4px 10px", border: "1px solid var(--border)", borderRadius: 4 }} onClick={() => setUser(null)}>
                {user.name}
              </div>
            </div>
          ) : (
            <button className="nav-cta" onClick={() => { setAuthMode("login"); setShowAuth(true); }}>
              Sign In
            </button>
          )}
        </div>
      </nav>

      {page === "home" && <Landing setPage={setPage} />}
      {page === "autopsy" && <RaceAutopsy user={user} addToast={addToast} setShowAuth={() => setShowAuth(true)} />}
      {page === "simulator" && <F1Simulator />}
      {page === "leaderboard" && <Leaderboard />}
      {page === "teams" && <TeamDashboard user={user} setShowAuth={() => setShowAuth(true)} />}
      {page === "news" && <News />}
      {showAuth && (
        <AuthModal
          mode={authMode}
          setMode={setAuthMode}
          onClose={() => setShowAuth(false)}
          onLogin={handleLogin}
        />
      )}
    </>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}


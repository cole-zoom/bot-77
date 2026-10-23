import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./index.css";
import Overview from "./pages/Overview";
import Matches from "./pages/Matches";
import Match from "./pages/Match";
import Moment from "./pages/Moment";
import Training from "./pages/Training";
import Features from "./pages/Features";
import Models from "./pages/Models";

const nav = [
  ["/", "Overview"],
  ["/matches", "Matches"],
  ["/training", "Training data"],
  ["/models", "Models"],
  ["/features", "Features"],
] as const;

function Shell() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
          <span className="font-semibold tracking-tight text-violet-300">bot-77 data</span>
          {nav.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) => `text-sm ${isActive ? "text-white" : "text-zinc-400 hover:text-zinc-200"}`}>
              {label}
            </NavLink>
          ))}
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/matches" element={<Matches />} />
          <Route path="/matches/:id" element={<Match />} />
          <Route path="/moments/:id" element={<Moment />} />
          <Route path="/training" element={<Training />} />
          <Route path="/features" element={<Features />} />
          <Route path="/models" element={<Models />} />
        </Routes>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  </React.StrictMode>,
);

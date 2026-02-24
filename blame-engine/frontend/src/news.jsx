import React from "react";
import "./news.css";

import redbull from "./assets/redbull.png";
import ferrari from "./assets/ferrari.png";
import mercedes from "./assets/mercedes.png";

const newsData = [
  {
    title: "RED BULL DRIVER ROTATION STRATEGY FOR 2026",
    desc: "Red Bull evaluating internal driver swap scenarios ahead of regulation overhaul.",
    category: "TEAM NEWS",
    image: redbull
  },
  {
    title: "FERRARI INTRODUCES NEW FLOOR CONCEPT",
    desc: "Mid-season aerodynamic upgrade focused on high-speed stability.",
    category: "TECH UPDATE",
    image: ferrari
  },
  {
    title: "MERCEDES SIMULATION MODEL BREAKTHROUGH",
    desc: "Wolff confirms advanced race-strategy AI now deployed trackside.",
    category: "STRATEGY",
    image: mercedes
  }
];

export default function News() {
  return (
    <div className="news-page">
      <div className="news-header">
        <h1>WHAT'S NEW</h1>
        <div className="news-underline"></div>
      </div>

      <div className="news-grid">
        {newsData.map((item, index) => (
          <div className="news-card" key={index}>
            <div className="news-image">
              <img src={item.image} alt={item.title} />
            </div>

            <div className="news-content">
              <span className="news-category">{item.category}</span>
              <h2>{item.title}</h2>
              <p>{item.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
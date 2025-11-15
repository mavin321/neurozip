import React, { useState } from "react";
import Header from "./components/Header.jsx";
import TextArea from "./components/TextArea.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import { compressText, decompressText } from "./api.js";

export default function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [stats, setStats] = useState(null);
  const [mode, setMode] = useState("compress");
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    setOutput("");
    setStats(null);

    const start = performance.now();

    try {
      let resp;
      if (mode === "compress") {
        resp = await compressText(input);
        setOutput(resp.data);
        setStats({
          input: input.length,
          output: resp.data.length,
          ratio: (resp.data.length / input.length).toFixed(2),
          time_ms: (performance.now() - start).toFixed(1),
        });
      } else {
        resp = await decompressText(input);
        setOutput(resp.text);
        setStats({
          input: input.length,
          output: resp.text.length,
          ratio: (input.length / resp.text.length).toFixed(2),
          time_ms: (performance.now() - start).toFixed(1),
        });
      }
    } catch (err) {
      alert("Error: " + err);
    }

    setLoading(false);
  }

  return (
    <>
      <Header />

      <div
        style={{
          maxWidth: "900px",
          margin: "30px auto",
          padding: "20px",
        }}
      >
        <div style={{ display: "flex", gap: "12px", marginBottom: "12px" }}>
          <button
            onClick={() => setMode("compress")}
            style={{
              padding: "10px 18px",
              borderRadius: "8px",
              border: "1px solid #d0d7de",
              background: mode === "compress" ? "#0366d6" : "white",
              color: mode === "compress" ? "white" : "#24292e",
              cursor: "pointer",
            }}
          >
            Compress
          </button>

          <button
            onClick={() => setMode("decompress")}
            style={{
              padding: "10px 18px",
              borderRadius: "8px",
              border: "1px solid #d0d7de",
              background: mode === "decompress" ? "#0366d6" : "white",
              color: mode === "decompress" ? "white" : "#24292e",
              cursor: "pointer",
            }}
          >
            Decompress
          </button>

          <button
            onClick={run}
            disabled={loading}
            style={{
              marginLeft: "auto",
              padding: "10px 18px",
              borderRadius: "8px",
              border: "1px solid #d0d7de",
              background: "#2ea44f",
              color: "white",
              cursor: "pointer",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Working..." : "Run"}
          </button>
        </div>

        <TextArea
          value={input}
          onChange={setInput}
          placeholder={
            mode === "compress"
              ? "Enter text to compress..."
              : "Paste Base64 data to decompress..."
          }
        />

        <h3 style={{ marginTop: "20px" }}>Output</h3>
        <TextArea
          value={output}
          onChange={setOutput}
          placeholder="Output will appear here"
        />

        <StatsPanel stats={stats} />
      </div>
    </>
  );
}

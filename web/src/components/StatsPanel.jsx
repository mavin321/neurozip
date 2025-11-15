export default function StatsPanel({ stats }) {
  if (!stats) return null;

  return (
    <div
      style={{
        marginTop: "20px",
        padding: "12px 16px",
        background: "#fff",
        borderRadius: "8px",
        border: "1px solid #e3e6e8",
      }}
    >
      <div>
        <strong>Input size:</strong> {stats.input} bytes
      </div>
      <div>
        <strong>Output size:</strong> {stats.output} bytes
      </div>
      <div>
        <strong>Ratio:</strong> {stats.ratio}
      </div>
      <div>
        <strong>Time:</strong> {stats.time_ms} ms
      </div>
    </div>
  );
}

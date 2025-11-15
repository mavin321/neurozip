const API_URL = "http://localhost:5000";

export async function compressText(text) {
  const res = await fetch(`${API_URL}/compress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function decompressText(base64) {
  const res = await fetch(`${API_URL}/decompress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: base64 }),
  });
  return res.json();
}

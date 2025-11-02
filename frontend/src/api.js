export async function classifyImage(file) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch("/api/predict", { 
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

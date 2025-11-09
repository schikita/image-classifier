export async function classifyImage(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch('/api/predict', { method: 'POST', body: fd });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'Predict failed');

  const label =
    data.predicted_label ?? data.label ?? data.class ?? 'unknown';
  const confidence = Number(data.confidence ?? data.top_prob ?? data.probability ?? 0);

  return {
    ...data,
    label,
    top_prob: Number.isFinite(confidence) ? confidence : 0,
  };
}

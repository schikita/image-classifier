import { useState } from "react";
import { classifyImage } from "../api";

export default function ImageClassifier() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError("Select image...");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await classifyImage(image);
      setResult(data); // <-- сохраняем весь объект ответа
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Classify image</h2>

      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? "Detecting..." : "Recognize"}
        </button>
      </form>

      {preview && (
        <div className="preview">
          <img src={preview} alt="preview" />
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <p className="result">
          Result: <b>{result.label}</b>{" "}
          ({(result.top_prob * 100).toFixed(1)}%)
        </p>
      )}

      {/* опционально: топ-3 классов */}
      {result?.probs && (
        <ul>
          {result.probs
            .map((p, i) => ({ i, p }))
            .sort((a, b) => b.p - a.p)
            .slice(0, 3)
            .map(({ i, p }) => (
              <li key={i}>
                class {i}: {(p * 100).toFixed(1)}%
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}

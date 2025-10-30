import { useState } from "react";
import { classifyImage } from "../api";

export default function ImageClassifier() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];

        if (file) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
            setError(null);
        }
    };

    const handleSubmit = async (e) => {
        e.prevtDefault();
        if (!image) {
            setError("Select image...");
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const data = await classifyImage(image);
            setError(data.result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card">
            <h2>Classify images</h2>

            <form action="" onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleFileChange}/>
                <button type="submit" disable={loading}>
                    {loading ? "Detecting..." : "Recognize"}
                </button>
            </form>

            {
                preview && (
                    <div className="preview">
                        <img src={preview} alt="preview" />
                    </div>
                )
            }

            {error && <p className="error">{error}</p>}
            {result && <p className="result">Result: {result}</p>}
        </div>
    );
}
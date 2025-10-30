export async function classifyImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error("Error load in API...");
    return response.json()
}
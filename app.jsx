import { useState } from "react";
import axios from "axios";

function App() {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState("");

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);

        try {
            const res = await axios.post("http://127.0.0.1:5000/predict", formData);
            setPrediction(res.data.prediction);
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    };

    return (
        <div className="app">
            <h1>AI Medicine Recognition</h1>
            <input type="file" onChange={(e) => setFile(e.target.files[0])} />
            <button onClick={handleUpload}>Upload</button>
            {prediction && <h3>Predicted: {prediction}</h3>}
        </div>
    );
}

export default App;

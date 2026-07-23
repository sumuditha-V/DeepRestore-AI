import React, { useState } from 'react';
import axios from 'axios';
import './DocumentRestore.css';

const API_URL = 'http://localhost:5000';

const DocumentRestore = () => {
    const [file, setFile] = useState(null);
    const [originalImage, setOriginalImage] = useState('');
    const [denoisedImage, setDenoisedImage] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (event) => {
        const selected = event.target.files[0];
        if (!selected) return;

        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(selected.type)) {
            setError('Please select a valid image file (JPEG or PNG)');
            return;
        }
        if (selected.size > 10 * 1024 * 1024) {
            setError('File size too large (max 10MB)');
            return;
        }

        setFile(selected);
        setError('');
        setResult(null);
        setDenoisedImage('');

        const reader = new FileReader();
        reader.onload = (e) => setOriginalImage(e.target.result);
        reader.readAsDataURL(selected);
    };

    const handleRestore = async () => {
        if (!file) {
            setError('Please select a document image first');
            return;
        }
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post(
                `${API_URL}/restore-document`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: 180000, // NLP model load can be slow on first run
                }
            );

            setDenoisedImage(`${API_URL}${response.data.denoised_image_path}`);
            setResult(response.data);
        } catch (err) {
            let msg = 'An error occurred while restoring the document.';
            if (err.code === 'ECONNABORTED') {
                msg = 'Request timed out (the NLP model may still be loading — try again).';
            } else if (err.response) {
                msg = err.response.data.error || `Server error (${err.response.status})`;
            } else if (err.request) {
                msg = 'No response from server. Is the Flask backend running?';
            }
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    const copyText = (text) => navigator.clipboard?.writeText(text || '');

    return (
        <div className="doc-restore-container">
            <h2>Document Restore <span className="badge">CV + NLP</span></h2>
            <p className="doc-intro">
                Upload a noisy or scanned document. DeepRestore AI denoises it (Computer
                Vision), reads the text with OCR, then spell-corrects, summarises and
                extracts keywords (NLP).
            </p>

            <div className="doc-upload-section">
                <label className="doc-dropzone">
                    <input
                        type="file"
                        accept=".jpg,.jpeg,.png"
                        onChange={handleFileChange}
                        hidden
                    />
                    {!file ? (
                        <>
                            <div className="doc-icon">📄</div>
                            <p>Click to choose a document image (JPEG / PNG, max 10MB)</p>
                        </>
                    ) : (
                        <p><strong>{file.name}</strong> selected</p>
                    )}
                </label>

                <button
                    className="doc-button"
                    onClick={handleRestore}
                    disabled={loading || !file}
                >
                    {loading ? 'Restoring… (first run loads the NLP model)' : 'Restore & Read Document'}
                </button>
            </div>

            {error && (
                <div className="doc-error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {(originalImage || denoisedImage) && (
                <div className="doc-images">
                    {originalImage && (
                        <div className="doc-image-card">
                            <h3>Original</h3>
                            <img src={originalImage} alt="Original document" />
                        </div>
                    )}
                    {denoisedImage && (
                        <div className="doc-image-card">
                            <h3>Denoised</h3>
                            <img src={denoisedImage} alt="Denoised document" />
                        </div>
                    )}
                </div>
            )}

            {result && (
                <div className="doc-results">
                    {result.summary && (
                        <div className="doc-panel doc-summary">
                            <div className="doc-panel-head">
                                <h3>NLP Summary</h3>
                                <button onClick={() => copyText(result.summary)}>Copy</button>
                            </div>
                            <p>{result.summary}</p>
                        </div>
                    )}

                    {result.keywords?.length > 0 && (
                        <div className="doc-panel">
                            <h3>Keywords</h3>
                            <div className="doc-keywords">
                                {result.keywords.map((k) => (
                                    <span key={k} className="doc-chip">{k}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="doc-panel">
                        <div className="doc-panel-head">
                            <h3>Extracted Text (spell-corrected)</h3>
                            <button onClick={() => copyText(result.corrected_text)}>Copy</button>
                        </div>
                        <div className="doc-stats">
                            {result.stats?.word_count} words · {result.stats?.char_count} chars
                        </div>
                        <pre className="doc-text">
                            {result.corrected_text || '(no text detected)'}
                        </pre>
                    </div>

                    {result.notes?.length > 0 && (
                        <div className="doc-notes">
                            <strong>Notes:</strong>
                            <ul>
                                {result.notes.map((n, i) => <li key={i}>{n}</li>)}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default DocumentRestore;

import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import DocumentRestore from './components/DocumentRestore';
import './App.css';

function App() {
  const [tab, setTab] = useState('image');

  return (
    <div className="App">
      <header className="app-header">
        <h1>DeepRestore AI</h1>
        <p className="app-subtitle">
          Intelligent Multimedia Noise Removal using Autoencoders
        </p>
      </header>

      <nav className="app-tabs">
        <button
          className={tab === 'image' ? 'active' : ''}
          onClick={() => setTab('image')}
        >
          🖼️ Image Denoiser
        </button>
        <button
          className={tab === 'document' ? 'active' : ''}
          onClick={() => setTab('document')}
        >
          📄 Document Restore (CV + NLP)
        </button>
      </nav>

      {tab === 'image' ? <ImageUploader /> : <DocumentRestore />}
    </div>
  );
}

export default App;

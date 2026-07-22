import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const API_URL = 'http://localhost:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [serverUp, setServerUp] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [activeTab, setActiveTab] = useState('scanner');
  const [history, setHistory] = useState(() => JSON.parse(localStorage.getItem('pixeltruth_history') || '[]'));

  useEffect(() => {
    localStorage.setItem('pixeltruth_history', JSON.stringify(history));
  }, [history]);

  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? setServerUp(true) : setServerUp(false))
      .catch(() => setServerUp(false));
  }, []);

  const pickFile = useCallback((f) => {
    if (!f) return;
    const ok = ['image/jpeg', 'image/png', 'image/webp'];
    if (!ok.includes(f.type)) { setError('Unsupported format. Use JPG, PNG, or WEBP.'); return; }
    if (f.size > 10 * 1024 * 1024) { setError('File exceeds 10 MB limit.'); return; }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Server returned ${res.status}`);
      }
      const newResult = await res.json();
      setResult(newResult);
      setHistory(prev => [{
          id: Date.now(),
          filename: file.name,
          date: new Date().toLocaleString(),
          label: newResult.label,
          confidence: newResult.confidence
      }, ...prev]);
      setServerUp(true);
    } catch (e) {
      if (e.message.includes('Failed to fetch') || e.name === 'TypeError') {
        setError('Cannot reach server. Is the backend running?'); setServerUp(false);
      } else { setError(e.message); }
    } finally { setLoading(false); }
  };

  const reset = () => {
    if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null); setResult(null); setError(null); setLoading(false);
  };

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]); };

  const label = result ? result.label.toUpperCase() : null;
  const isReal = label === 'REAL';
  const conf = result ? (result.confidence <= 1 ? result.confidence * 100 : result.confidence) : 0;
  const confBlocks = Math.round((conf / 100) * 12);

  return (
    <>
      {/* Navbar */}
      <nav className="w-full sticky top-0 z-40 bg-surface border-b-4 border-black flex justify-between items-center h-16 px-6 max-w-full font-mono tracking-tighter shadow-retro">
        <div className="text-xl font-bold text-black flex items-center gap-2 uppercase">
            PixelTruth
        </div>
        <div className="hidden md:flex gap-8 uppercase font-bold">
            <a className={`cursor-pointer ${activeTab === 'scanner' ? 'text-black border-b-4 border-black pb-1' : 'text-black/60 hover:text-black border-b-4 border-transparent pb-1'}`} onClick={() => setActiveTab('scanner')}>Scanner</a>
            <a className={`cursor-pointer ${activeTab === 'history' ? 'text-black border-b-4 border-black pb-1' : 'text-black/60 hover:text-black border-b-4 border-transparent pb-1'}`} onClick={() => setActiveTab('history')}>History</a>
        </div>
        <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1 bg-surface border-2 border-black rounded-none shadow-retro-sm text-xs font-bold`}>
                <div className={`w-3 h-3 border border-black ${serverUp ? 'bg-[#00ff88]' : 'bg-danger'} animate-pulse`}></div>
                <span className="text-black text-xs">{serverUp ? 'SYS_ONLINE' : 'SYS_OFFLINE'}</span>
            </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8 max-w-6xl z-10 relative">
        <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-black mb-4 tracking-tight drop-shadow-sm uppercase">
                &gt; AI-GENERATED IMAGE DETECTOR_
            </h1>
            <p className="text-black/70 max-w-2xl mx-auto font-bold">Upload an image to analyze digital artifacts and determine synthetic origin probability.</p>
        </div>

        {activeTab === 'scanner' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Left Column: IMAGE INPUT */}
              <div className="bg-surface border-4 border-black rounded-none p-6 shadow-retro flex flex-col h-full">
                  <div className="flex items-center gap-3 border-b-4 border-black pb-4 mb-6">
                      <span className="material-symbols-outlined text-black font-bold">upload_file</span>
                      <h2 className="text-lg font-bold text-black tracking-wider uppercase">IMAGE INPUT</h2>
                  </div>

                  <div 
                      className={`flex-grow flex flex-col justify-center items-center border-4 ${dragOver ? 'border-black border-solid bg-black/5' : 'border-dashed border-black'} transition-colors rounded-none bg-background p-8 text-center cursor-pointer mb-6 group relative overflow-hidden`}
                      onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
                      onClick={() => !loading && inputRef.current?.click()}
                  >
                      <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={e => e.target.files[0] && pickFile(e.target.files[0])} />
                      
                      {preview ? (
                          <img src={preview} alt="Preview" className="w-full h-full object-contain max-h-64 border-2 border-black bg-white" />
                      ) : (
                          <>
                              <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                              <span className="material-symbols-outlined text-5xl text-black/40 group-hover:text-black mb-4 transition-colors font-bold">cloud_upload</span>
                              <p className="text-lg text-black font-bold mb-2 uppercase">DROP IMAGE HERE</p>
                              <p className="text-sm text-black/60 font-bold uppercase">or click to browse</p>
                              <div className="mt-6 px-3 py-1 bg-surface border-2 border-black rounded-none text-xs text-black font-bold shadow-retro-sm">
                                  JPG · PNG · WEBP — max 10 MB
                              </div>
                          </>
                      )}
                  </div>

                  {file && (
                      <div className="flex justify-between items-center text-sm font-bold font-mono text-black mb-4 px-2">
                          <span className="truncate max-w-[70%]">{file.name}</span>
                          <span>{(file.size / 1024).toFixed(0)} KB</span>
                      </div>
                  )}
                  
                  {error && (
                      <div className="text-white bg-danger font-bold text-sm font-mono mb-4 px-3 py-2 border-2 border-black shadow-retro-sm">
                          ERROR: {error}
                      </div>
                  )}

                  <div className="grid grid-cols-2 gap-4 mb-4">
                      <button className="px-4 py-3 border-2 border-black bg-white hover:bg-black/5 text-black font-bold uppercase shadow-retro-sm active:translate-y-1 active:translate-x-1 active:shadow-none transition-all" onClick={() => !loading && inputRef.current?.click()}>
                          SELECT FILE
                      </button>
                      <button className="px-4 py-3 border-2 border-black bg-white hover:bg-danger/10 text-danger font-bold uppercase shadow-retro-sm active:translate-y-1 active:translate-x-1 active:shadow-none transition-all" onClick={reset}>
                          RESET
                      </button>
                  </div>

                  <button 
                      className={`w-full px-4 py-4 ${loading || !file ? 'bg-black/10 text-black/40 border-black/20 shadow-none' : 'bg-black text-white hover:bg-black/90 shadow-retro active:translate-y-1 active:translate-x-1 active:shadow-none'} border-4 border-black font-bold text-lg uppercase transition-all`}
                      onClick={analyze}
                      disabled={loading || !file}
                  >
                      {loading ? 'ANALYZING...' : 'ANALYZE IMAGE'}
                  </button>
              </div>

              {/* Right Column: DETECTION RESULT */}
              <div className="bg-surface border-4 border-black rounded-none p-6 shadow-retro flex flex-col h-full">
                  <div className="flex items-center gap-3 border-b-4 border-black pb-4 mb-6">
                      <span className="material-symbols-outlined text-black font-bold">analytics</span>
                      <h2 className="text-lg font-bold text-black tracking-wider uppercase">DETECTION RESULT</h2>
                  </div>

                  <div className={`flex-grow border-4 ${result ? (isReal ? 'border-[#00ff88]' : 'border-danger') : 'border-black'} bg-background flex flex-col items-center justify-center mb-6 relative overflow-hidden h-64 lg:h-auto p-4 shadow-inner`}>
                      <div className="absolute inset-0 flex items-center justify-center bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjMGEwYTBmIj48L3JlY3Q+CjxwYXRoIGQ9Ik0wIDBMOCA4Wk04IDBMMCA4WiIgc3Ryb2tlPSIjMWExYTI1IiBzdHJva2Utd2lkdGg9IjEiPjwvcGF0aD4KPC9zdmc+')] opacity-10"></div>
                      
                      <div className="text-center z-10 w-full bg-white/80 p-4 border-2 border-black shadow-retro-sm max-w-sm mx-auto">
                          {loading ? (
                              <>
                                  <span className="material-symbols-outlined text-5xl text-black mb-2 animate-spin">memory</span>
                                  <p className="text-black tracking-widest font-bold animate-pulse">SCANNING...</p>
                              </>
                          ) : result ? (
                              <div className="flex flex-col items-center space-y-4">
                                  <span className={`text-4xl font-bold tracking-widest uppercase ${isReal ? 'text-black' : 'text-danger'}`}>{label}</span>
                                  <span className={`text-6xl font-mono font-bold ${isReal ? 'text-black' : 'text-danger'}`}>{conf.toFixed(1)}%</span>
                                  <p className="text-black font-bold mt-2">{isReal ? 'This image appears to be a genuine photograph.' : 'This image appears to be AI-generated.'}</p>
                              </div>
                          ) : (
                              <>
                                  <span className="material-symbols-outlined text-5xl text-black/40 mb-2">hourglass_empty</span>
                                  <p className="text-black tracking-widest font-bold uppercase">AWAITING INPUT</p>
                              </>
                          )}
                      </div>
                  </div>

                  <div className="space-y-4">
                      <div className="flex justify-between items-center bg-background p-3 border-2 border-black shadow-retro-sm">
                          <span className="text-sm font-bold text-black uppercase">CONFIDENCE</span>
                          <div className="flex gap-1 border-2 border-black bg-white p-1">
                              {Array.from({ length: 12 }).map((_, i) => (
                                  <div key={i} className={`w-3 h-5 border border-black/20 ${result && i < confBlocks ? (isReal ? 'bg-[#00ff88]' : 'bg-danger') : 'bg-background'}`}></div>
                              ))}
                          </div>
                      </div>

                      <div className="flex justify-between items-center bg-background p-3 border-2 border-black shadow-retro-sm">
                          <span className="text-sm font-bold text-black uppercase">LABEL</span>
                          <span className={`px-2 py-1 text-xs font-bold uppercase border-2 ${result ? (isReal ? 'bg-white text-black border-black' : 'bg-danger text-white border-black') : 'bg-surface text-black/40 border-black'}`}>
                              {result ? label : '--'}
                          </span>
                      </div>

                      <div className="flex justify-between items-center bg-background p-3 border-2 border-black shadow-retro-sm">
                          <span className="text-sm font-bold text-black uppercase">RAW SCORE</span>
                          <span className="font-mono font-bold text-black bg-white px-2 py-1 border-2 border-black">
                              {result ? result.confidence.toFixed(4) : '0.0000'}
                          </span>
                      </div>
                  </div>
              </div>
          </div>
        ) : (
          <div className="bg-surface border-4 border-black p-8 shadow-retro min-h-[500px]">
              <div className="flex items-center gap-3 border-b-4 border-black pb-4 mb-8">
                  <span className="material-symbols-outlined text-black font-bold text-3xl">history</span>
                  <h2 className="text-2xl font-bold text-black tracking-wider uppercase">SCAN HISTORY</h2>
              </div>
              {history.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-64 opacity-50">
                      <span className="material-symbols-outlined text-6xl mb-4">inbox</span>
                      <p className="text-lg font-bold font-mono uppercase tracking-widest">No history yet</p>
                  </div>
              ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {history.map(item => {
                          const isRealHistory = item.label === 'REAL';
                          const confVal = item.confidence <= 1 ? item.confidence * 100 : item.confidence;
                          return (
                          <div key={item.id} className="border-4 border-black bg-background p-4 shadow-retro-sm flex flex-col justify-between hover:bg-white transition-colors">
                              <div className="mb-4 pb-4 border-b-2 border-black border-dashed">
                                  <p className="font-bold font-mono text-black text-lg truncate w-full" title={item.filename}>{item.filename}</p>
                                  <p className="text-xs font-mono text-black/60 mt-1 uppercase font-bold">{item.date}</p>
                              </div>
                              <div className="flex justify-between items-end">
                                  <div className="flex flex-col">
                                      <span className="text-[10px] text-black/60 font-bold uppercase mb-1">Result</span>
                                      <span className={`px-2 py-1 text-xs font-bold uppercase border-2 text-center w-20 ${isRealHistory ? 'bg-white text-black border-black' : 'bg-danger text-white border-black'}`}>
                                          {item.label}
                                      </span>
                                  </div>
                                  <div className="flex flex-col text-right">
                                      <span className="text-[10px] text-black/60 font-bold uppercase mb-1">Confidence</span>
                                      <span className="font-mono font-bold text-black text-xl">
                                          {confVal.toFixed(1)}%
                                      </span>
                                  </div>
                              </div>
                          </div>
                      )})}
                  </div>
              )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full py-8 mt-auto bg-surface border-t-4 border-black flex flex-col md:flex-row justify-between items-center px-8 gap-4 font-mono text-xs uppercase font-bold text-black">
        <div className="text-sm font-bold">
            PixelTruth
        </div>
        <div className="text-center flex-grow">
            PixelTruth v1.0 — CNN-based AI image detector
        </div>
        <div className="flex gap-4">
            <a className="hover:underline cursor-pointer" onClick={() => setModalContent({title: 'Documentation', text: 'PixelTruth uses a Convolutional Neural Network (CNN) architecture trained on millions of real and AI-generated images to detect synthetic artifacts. Our model looks for frequency-domain anomalies common in diffusion models and GANs.'})}>Documentation</a>
            <a className="hover:underline cursor-pointer" onClick={() => setModalContent({title: 'Privacy Policy', text: 'We do not store your images. All processing is done locally or instantly discarded after inference. No data is shared with third parties.'})}>Privacy</a>
            <a className="hover:underline cursor-pointer" onClick={() => setModalContent({title: 'Terms of Service', text: 'PixelTruth is provided "as is" without warranty of any kind. You agree not to use this service for malicious purposes or automated scraping without permission.'})}>Terms</a>
            <a className="hover:underline cursor-pointer" href="https://github.com/anshul1058/Pixel-truth" target="_blank" rel="noreferrer">Github</a>
        </div>
        <div className="">
            SYSTEM_STABLE // PIXELTRUTH v1.0.0
        </div>
      </footer>
      {/* Modal */}
      {modalContent && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 backdrop-blur-sm" onClick={() => setModalContent(null)}>
          <div className="bg-surface border-4 border-black p-6 shadow-retro max-w-lg w-full relative" onClick={e => e.stopPropagation()}>
            <button className="absolute top-4 right-4 material-symbols-outlined text-black hover:text-danger" onClick={() => setModalContent(null)}>close</button>
            <h3 className="text-xl font-bold font-mono uppercase mb-4 text-black border-b-4 border-black pb-2 inline-block">{modalContent.title}</h3>
            <p className="font-mono font-bold text-black/80 leading-relaxed">{modalContent.text}</p>
            <div className="mt-8 flex justify-end">
              <button className="px-6 py-2 border-4 border-black bg-white hover:bg-black/5 text-black font-bold uppercase shadow-retro-sm active:translate-y-1 active:translate-x-1 active:shadow-none transition-all" onClick={() => setModalContent(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

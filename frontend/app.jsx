const { useState, useEffect } = React;

function App() {
    // State for the Controls
    const [illumination, setIllumination] = useState(0.5);
    const [machBands, setMachBands] = useState(false);

    // State for Backend Response
    const [imageData, setImageData] = useState(null);
    const [telemetry, setTelemetry] = useState({});
    const [loading, setLoading] = useState(false);

    // Fetch data from FastAPI backend
    const processImage = async () => {
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('illumination', illumination);
            formData.append('mach_bands', machBands);

            const response = await fetch('http://localhost:8000/api/perception/simulate', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setImageData(data.image);
                setTelemetry(data.telemetry);
            } else {
                console.error("Backend Error");
            }
        } catch (error) {
            console.error("Network Error:", error);
        }
        setLoading(false);
    };

    // Auto-update when sliders change
    useEffect(() => {
        processImage();
    }, [illumination, machBands]);

    return (
        <div className="dashboard-container">
            {/* LEFT PANE: CONTROLS */}
            <div className="pane pane-controls">
                <h1>DIP Engine</h1>

                <h2>Command Center</h2>
                <div className="control-group">
                    <label>Module Selection</label>
                    <select className="cyber-btn" style={{ background: 'var(--bg-panel)', color: 'var(--text-main)', width: '100%', marginBottom: '2rem' }}>
                        <option>1.1 Visual Perception</option>
                        <option disabled>More modules coming soon...</option>
                    </select>
                </div>

                <h2>Perception Parameters</h2>
                <div className="control-group">
                    <label>Illumination Level: <span className="mono-text">{illumination}</span></label>
                    <input
                        type="range"
                        min="0" max="1" step="0.01"
                        value={illumination}
                        onChange={(e) => setIllumination(parseFloat(e.target.value))}
                    />
                </div>

                <div className="control-group" style={{ marginTop: '2rem' }}>
                    <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                        <input
                            type="checkbox"
                            checked={machBands}
                            onChange={(e) => setMachBands(e.target.checked)}
                            style={{ marginRight: '10px', width: '20px', height: '20px' }}
                        />
                        Enable Mach Band Effect
                    </label>
                </div>

                <button className="cyber-btn" onClick={processImage} style={{ marginTop: '3rem' }}>
                    {loading ? "Processing..." : "Force Render"}
                </button>
            </div>

            {/* CENTER PANE: VIEWPORT */}
            <div className="pane pane-viewport">
                <div className="viewer-card" style={{ marginTop: '2rem', flexGrow: 1 }}>
                    <h2 style={{ alignSelf: 'flex-start', borderBottom: 'none' }}>Viewport Alpha</h2>
                    {imageData ? (
                        <img src={imageData} alt="Processed Output" className="render-canvas" style={{ marginTop: 'auto', marginBottom: 'auto' }} />
                    ) : (
                        <div style={{ color: 'var(--neon-cyan)', margin: 'auto', fontFamily: 'Fira Code', textShadow: '0 0 10px var(--neon-cyan)' }}>
                            [ NO SIGNAL RECEIVED ]
                        </div>
                    )}
                </div>
            </div>

            {/* RIGHT PANE: TELEMETRY & STATS */}
            <div className="pane pane-telemetry">
                <h2>System Telemetry</h2>

                <div className="telemetry-card">
                    <label>Render Status</label>
                    <div className="metric-value mono-text" style={{ color: loading ? 'var(--neon-pink)' : 'var(--neon-green)' }}>
                        {loading ? "COMPUTING" : "ONLINE"}
                    </div>
                </div>

                {machBands ? (
                    <>
                        <div className="telemetry-card">
                            <label>Perception Mode</label>
                            <div className="metric-value">Mach Bands</div>
                        </div>
                        <div className="telemetry-card">
                            <label>Intensity Steps</label>
                            <div className="metric-value mono-text">{telemetry.steps || 0}</div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="telemetry-card">
                            <label>Vision Paradigm</label>
                            <div className="metric-value" style={{ fontSize: '1.4rem' }}>{telemetry.vision_mode || "Await Data"}</div>
                        </div>
                        <div className="telemetry-card">
                            <label>Rod Activation</label>
                            <div className="metric-value mono-text" style={{ color: telemetry.scotopic_active ? 'var(--neon-cyan)' : 'var(--text-muted)' }}>
                                {telemetry.scotopic_active ? "HIGH" : "LOW"}
                            </div>
                        </div>
                        <div className="telemetry-card">
                            <label>Bg Intensity (0-255)</label>
                            <div className="metric-value mono-text">{telemetry.background_intensity || 0}</div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

import React from 'react';

const VisualPerceptionControls = ({
    illumination,
    setIllumination,
    showMachBands,
    setShowMachBands,
    visionMode,
    setVisionMode
}) => {

    // Derived calc for display
    const linearIntensity = Math.pow(10, illumination).toFixed(2);

    return (
        <div className="vp-controls-panel">

            {/* Illumination Slider */}
            <div className="control-group">
                <label className="control-label" htmlFor="illum-slider">
                    Illumination (Log Scale)
                    <span className="control-value">{illumination.toFixed(2)}</span>
                </label>
                <input
                    id="illum-slider"
                    type="range"
                    min="-2.0"
                    max="1.0"
                    step="0.01"
                    value={illumination}
                    onChange={(e) => setIllumination(parseFloat(e.target.value))}
                    aria-label="Illumination Logarithmic Slider"
                />
                <div className="control-helper">
                    Simulates adapting from dark (-2) to bright (+1) scenes.
                </div>
            </div>

            {/* Linear Intensity Readout */}
            <div className="control-group">
                <div className="control-label">
                    Linear Intensity
                    <span className="control-value">{linearIntensity} cd/m²</span>
                </div>
                <div className="control-helper">
                    Real-world intensity scaling based on log input.
                </div>
            </div>

            {/* Vision Mode */}
            <div className="control-group">
                <label className="control-label">Vision Mode</label>
                <div className="radio-group">
                    <label className="radio-option">
                        <input
                            type="radio"
                            name="visionMode"
                            value="photopic"
                            checked={visionMode === 'photopic'}
                            onChange={() => setVisionMode('photopic')}
                        />
                        Photopic (Cones)
                    </label>
                    <label className="radio-option">
                        <input
                            type="radio"
                            name="visionMode"
                            value="scotopic"
                            checked={visionMode === 'scotopic'}
                            onChange={() => setVisionMode('scotopic')}
                        />
                        Scotopic (Rods)
                    </label>
                </div>
                <div className="control-helper">
                    {visionMode === 'photopic'
                        ? 'Color vision, high detail, active in bright light.'
                        : 'Grayscale vision, low detail, active in dim light.'}
                </div>
            </div>

            {/* Mach Bands Toggle */}
            <div className="control-group">
                <label className="toggle-label">
                    <input
                        type="checkbox"
                        checked={showMachBands}
                        onChange={(e) => setShowMachBands(e.target.checked)}
                    />
                    <span className="control-label">Show Mach Bands Illusion</span>
                </label>
                <div className="control-helper">
                    Visual system enhances edges by creating "overshoot" bands.
                </div>
            </div>

        </div>
    );
};

export default VisualPerceptionControls;

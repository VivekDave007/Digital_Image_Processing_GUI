import React, { useMemo } from 'react';

const VisualPerceptionSimulation = ({ illumination, showMachBands, visionMode }) => {

    // Calculate simulation styles
    const isScotopic = visionMode === 'scotopic';
    const linearInt = Math.pow(10, illumination);

    // Brightness adaptation simulation
    // Map -2..1 to a visible opacity/brightness range suitable for screen
    // -2 => Very dark (0.2 opacity), 1 => Bright (1.0 opacity)
    const brightness = 0.2 + ((illumination + 2) / 3) * 0.8;

    const simulationStyle = {
        // Basic adjustments
        filter: `brightness(${brightness}) grayscale(${isScotopic ? 1 : 0})`,
        // Mach Bands: Create a gradient stripe pattern
        background: showMachBands
            ? `repeating-linear-gradient(
          90deg, 
          #111 0%, #111 10%, 
          #333 10%, #333 20%,
          #555 20%, #555 30%,
          #777 30%, #777 40%,
          #999 40%, #999 50%,
          #bbb 50%, #bbb 60%,
          #ddd 60%, #ddd 70%,
          #fff 70%, #fff 80%
        )`
            : `radial-gradient(circle at center, ${isScotopic ? '#fff' : '#ffeb3b'} 0%, ${isScotopic ? '#111' : '#f44336'} 70%)`
    };

    return (
        <div className="vp-simulation-panel">
            <div className="sim-canvas-container">

                {/* The Scene Layer */}
                <div
                    className="sim-canvas-content"
                    style={simulationStyle}
                    role="img"
                    aria-label={showMachBands ? "Mach Bands Gradient Pattern" : "Simulated Flower Scene"}
                />

                {/* Dynamic Overlay Info */}
                <div className="sim-overlay-text">
                    State: {showMachBands ? 'Mach Bands' : 'Natural Scene'} |
                    Mode: {isScotopic ? 'Scotopic' : 'Photopic'} |
                    Brightness: {(brightness * 100).toFixed(0)}%
                </div>

            </div>
        </div>
    );
};

export default VisualPerceptionSimulation;

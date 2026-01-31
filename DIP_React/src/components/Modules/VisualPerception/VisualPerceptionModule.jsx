import React, { useState } from 'react';
import VisualPerceptionControls from './VisualPerceptionControls';
import VisualPerceptionSimulation from './VisualPerceptionSimulation';
import KeyIdeasPanel from './KeyIdeasPanel';
import './VisualPerception.css';

const VisualPerceptionModule = () => {
    // State
    const [illumination, setIllumination] = useState(0.0);
    const [showMachBands, setShowMachBands] = useState(false);
    const [visionMode, setVisionMode] = useState('photopic');

    return (
        <div className="vp-container">

            {/* Header */}
            <header className="vp-header">
                <h2 className="vp-title">1. Elements of Visual Perception</h2>
                <p className="vp-description">
                    Understanding how the human eye perceives light, intensity, and contrast
                    is fundamental to image processing. Explore adaptation and visual illusions below.
                </p>
            </header>

            {/* Left Column: Controls */}
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <VisualPerceptionControls
                    illumination={illumination}
                    setIllumination={setIllumination}
                    showMachBands={showMachBands}
                    setShowMachBands={setShowMachBands}
                    visionMode={visionMode}
                    setVisionMode={setVisionMode}
                />
                <KeyIdeasPanel />
            </div>

            {/* Right Column: Simulation */}
            <VisualPerceptionSimulation
                illumination={illumination}
                showMachBands={showMachBands}
                visionMode={visionMode}
            />

        </div>
    );
};

export default VisualPerceptionModule;

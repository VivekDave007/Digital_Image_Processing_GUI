import React from 'react';

const PlaceholderModule = ({ title }) => {
    return (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#a0a0a0' }}>
            <h2>{title}</h2>
            <p style={{ marginTop: '1rem' }}>⚠️ This module is currently under development.</p>
            <p>Check "Visual Perception" for the active content.</p>
        </div>
    );
};

export default PlaceholderModule;

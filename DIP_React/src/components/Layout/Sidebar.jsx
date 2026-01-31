import React from 'react';
import './Layout.css';

const Sidebar = ({ activeModule, onSelectModule }) => {
    const modules = [
        "Visual Perception",
        "EM Spectrum",
        "Acquisition",
        "Sampling",
        "Relationships",
        "Math Tools"
    ];

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <h1 className="sidebar-title">DIP Fundamentals</h1>
                <p className="sidebar-subtitle">Gonzalez & Woods, Ch 2</p>
            </div>

            <nav className="nav-list">
                {modules.map((mod) => (
                    <div
                        key={mod}
                        className={`nav-item ${activeModule === mod ? 'active' : ''}`}
                        onClick={() => onSelectModule(mod)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => { if (e.key === 'Enter') onSelectModule(mod) }}
                    >
                        {mod}
                    </div>
                ))}
            </nav>

            <div className="sidebar-footer">
                Developed by Vivek Dave.<br />
                v1.0 • Deployment Ready
            </div>
        </aside>
    );
};

export default Sidebar;

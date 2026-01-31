import React from 'react';

const KeyIdeasPanel = () => {
    return (
        <div className="key-ideas-panel">
            <h3 className="key-ideas-title">💡 Key Ideas</h3>
            <ul className="key-ideas-list">
                <li>
                    <strong>Adaptation:</strong> The eye cannot see
                    dynamic ranges of 10<sup>10</sup> simultaneously.
                    It adapts (Scotopic vs Photopic) around a mean brightness.
                </li>
                <li>
                    <strong>Mach Bands:</strong> The visual system "undershoots"
                    and "overshoots" at sharp intensity boundaries to enhance edges.
                </li>
                <li>
                    <strong>Weber Ratio:</strong> Brightness discrimination depends
                    on background intensity ($\Delta I / I$).
                </li>
            </ul>
        </div>
    );
};

export default KeyIdeasPanel;

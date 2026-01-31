import React, { useState } from 'react';
import Layout from './components/Layout/Layout';
import PlaceholderModule from './components/Modules/PlaceholderModule';
import VisualPerceptionModule from './components/Modules/VisualPerception/VisualPerceptionModule';

function App() {
    const [activeModule, setActiveModule] = useState("Visual Perception");

    const renderModule = () => {
        switch (activeModule) {
            case "Visual Perception":
                return <VisualPerceptionModule />;
            default:
                return <PlaceholderModule title={activeModule} />;
        }
    };

    return (
        <Layout activeModule={activeModule} onSelectModule={setActiveModule}>
            {renderModule()}
        </Layout>
    );
}

export default App;

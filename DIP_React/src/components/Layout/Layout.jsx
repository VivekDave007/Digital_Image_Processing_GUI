import React from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import './Layout.css';

const Layout = ({ activeModule, onSelectModule, children }) => {
    return (
        <div className="layout-container">
            <Sidebar activeModule={activeModule} onSelectModule={onSelectModule} />
            <div className="main-content-wrapper">
                <Header />
                <main className="content-scroll-area">
                    {children}
                </main>
            </div>
        </div>
    );
};

export default Layout;

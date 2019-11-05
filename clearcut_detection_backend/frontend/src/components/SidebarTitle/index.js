import React from 'react';

import './style.css';

const SidebarTitle = ({ children }) => (
  <div className="sidebar-title">
    <h4 className="title">{children}</h4>
  </div>
);

export default SidebarTitle;

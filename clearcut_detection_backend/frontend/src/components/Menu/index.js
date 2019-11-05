import React from 'react';

import './style.css';

const Menu = ({ links }) => {
  const renderLinks = () => {
    return links.map(({ url, text }) => (
      <a href={url} className="menu-link">
        {text}
      </a>
    ));
  };

  return <nav className="menu">{renderLinks()}</nav>;
};

export default Menu;

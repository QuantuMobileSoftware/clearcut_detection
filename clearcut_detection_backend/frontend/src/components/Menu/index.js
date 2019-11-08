import React from 'react';

import './style.css';

const Menu = ({ links }) => {
  const renderLinks = () => {
    return links.map(({ url, text }) => (
      <a
        key={text}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="menu-link"
      >
        {text}
      </a>
    ));
  };

  return <nav className="menu">{renderLinks()}</nav>;
};

export default Menu;

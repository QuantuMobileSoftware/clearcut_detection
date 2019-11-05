import React from 'react';

import './style.css';

const Switch = ({ state, handleClick }) => {
  const renderIcon = () => (
    <div className="switch-icon">
      <div></div>
      <div></div>
      <div></div>
    </div>
  );

  return (
    <div className={`switch${state ? ' is-open' : ''}`}>
      <button onClick={handleClick} className="switch-button">
        {renderIcon()}
      </button>
    </div>
  );
};

export default Switch;

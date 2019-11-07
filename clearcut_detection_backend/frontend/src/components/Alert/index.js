import React from 'react';
import { toast } from 'react-toastify';

import './style.css';

export const alertMessage = (message, type = 'default') =>
  toast(<Alert message={message} type={type} />, {
    type,
    closeButton: <>✕</>,
    hideProgressBar: true
  });

const Alert = ({ message, type }) => {
  const renderIcon = () => {
    return <div className="alert-icon">{type === 'error' ? '!' : '✓'}</div>;
  };

  return (
    <div className="alert-content">
      {type !== 'default' && renderIcon()}
      <div className="alert-message">
        <p className="message">{message}</p>
      </div>
    </div>
  );
};

export default Alert;

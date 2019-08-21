import React from 'react';

import './LoadingScreen.css';

const LoadingScreen = () => (
  <div className="load-screen">
    <div className="overlay-spinner">
      <div className="overlay-spinner-bound" />
    </div>
  </div>
);

export default LoadingScreen;

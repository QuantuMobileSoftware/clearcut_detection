import React from 'react';

import './CustomLegend.css';

const CustomLegend = ({ data }) => (
  <div className="custom-legend">
    {
      data.map((item, i) => (
        <div key={i} className="custom-legend-item">
          <span className={`square ${item.color}`} />
          <div>{item.label}</div>
        </div>
      ))
    }
  </div>
);

export default CustomLegend;

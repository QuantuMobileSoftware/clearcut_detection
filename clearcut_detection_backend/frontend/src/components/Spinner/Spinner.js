import React  from 'react';
import PropTypes from 'prop-types';

import './Spinner.css';

const Spinner = ({ className }) => (
  <div className={`spinner ${className}`} />
);

Spinner.propTypes = {
  className: PropTypes.string
};

Spinner.defaultProps = {
  className: ''
};

export default Spinner;

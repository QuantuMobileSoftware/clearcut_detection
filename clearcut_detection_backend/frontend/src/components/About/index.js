import React from "react";

import { ReactComponent as Logo } from "../../assets/images/logo-company.svg";

import "./style.css";
import Menu from "../Menu";

const About = ({ links }) => {
  const renderDescription = () => (
    <p className="about-description">
      Soils most likely suffering from erosion
    </p>
  );

  const renderAuthor = () => (
    <div className="about-author">
      <p className="author-text">Powered by</p>
      <div className="author-logo">
        <a
          href="https://www.quantumobile.com/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Logo />
        </a>
      </div>
    </div>
  );

  return (
    <div className="about">
      {renderDescription()}
      <Menu links={links} />
      {renderAuthor()}
    </div>
  );
};

export default About;

import React, { Component } from "react";
import ReactMapGL from "react-map-gl";

export default class Map extends Component {
  state = {
    viewport: {
      width: "100%",
      height: "100%",
      latitude: 49.988358,
      longitude: 36.232845,
      zoom: 9
    }
  };
  render() {
    return (
      <ReactMapGL
        {...this.state.viewport}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v9"
        mapboxApiAccessToken="pk.eyJ1IjoiYXZha2luIiwiYSI6ImNqeXk1cTk5aTAwcmszZnA4MjF0d2Fic3AifQ.-KMWEhdsLQ4dQIiC3p0KoA"
        onViewportChange={viewport => {
          console.log(viewport);
          this.setState({ viewport });
        }}
      />
    );
  }
}

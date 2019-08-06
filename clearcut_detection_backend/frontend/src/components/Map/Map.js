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
    },
    activeItem: null,
    position: {
      x: 0,
      y: 0
    }
  };
  _onLoad = (e, data) => {
    const MAP = this.map.getMap();
    this.initMapData(MAP, this.props.data, "clearcut");
    MAP.addLayer({
      id: "clearcut-polygon",
      type: "fill",
      source: "clearcut",
      paint: {
        "fill-color": "#00bcd4",
        "fill-opacity": 0.6
      },
      filter: ["==", "$type", "Polygon"]
    });
  };
  initMapData(map, data, sourceID) {
    map.addSource(sourceID, {
      type: "geojson",
      data
    });
  }
  _handleClick = event => {
    const {
      features,
      center: { x, y }
    } = event;
    const activeItem =
      features && features.find(f => f.layer.id === "clearcut-polygon");

    this.setState({ activeItem, position: { x, y } });
  };
  _renderTooltip = () => {
    const {
      activeItem,
      position: { x, y }
    } = this.state;
    return (
      activeItem && (
        <div className="tooltip" style={{ left: `${x}px`, top: `${y}px` }}>
          Polygon info:
          <div>Date of Image: {activeItem.properties.img_date}</div>
        </div>
      )
    );
  };
  _getCursor = cursors => {
    switch (true) {
      case cursors.isDragging:
        return "grabbing";
      case cursors.isHovering:
        return "pointer";

      default:
        return "default";
    }
  };
  render() {
    return (
      <ReactMapGL
        {...this.state.viewport}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v9"
        ref={node => (this.map = node)}
        mapboxApiAccessToken="pk.eyJ1IjoiYXZha2luIiwiYSI6ImNqeXk1cTk5aTAwcmszZnA4MjF0d2Fic3AifQ.-KMWEhdsLQ4dQIiC3p0KoA"
        onViewportChange={viewport => {
          this.setState({ viewport, activeItem: null });
        }}
        interactiveLayerIds={["clearcut-polygon"]}
        getCursor={this._getCursor}
        onLoad={this._onLoad}
        onClick={this._handleClick}
      >
        {this._renderTooltip()}
      </ReactMapGL>
    );
  }
}

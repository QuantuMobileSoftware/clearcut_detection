import React, { Component } from "react";
import ReactMapGL, { Popup } from "react-map-gl";

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
    hoveredItem: null,
    position: {
      longitude: 0,
      latitude: 0
    }
  };
  _onLoad = (e, data) => {
    const MAP = this.map.getMap();
    this.initMapData(MAP, this.props.data, "clearcut");
    console.log(this.props.data);
    MAP.addLayer({
      id: "clearcut-polygon",
      type: "fill",
      source: "clearcut",
      state: {
        hover: true
      },
      paint: {
        "fill-color": [
          "case",
          ["==", ["get", "color"], 0],
          "#00bcd4",
          ["==", ["get", "color"], 1],
          "yellow",
          ["==", ["get", "color"], 2],
          "red",
          "#fff"
        ],
        // "fill-opacity": 0.6,
        "fill-opacity": [
          "case",
          ["boolean", ["feature-state", "hover"], false],
          1,
          0.5
        ]
      },
      filter: ["==", "$type", "Polygon"]
    });
    MAP.addLayer({
      id: "clearcut-borders",
      type: "line",
      source: "clearcut",
      layout: {},
      paint: {
        "line-color": [
          "case",
          ["==", ["get", "color"], 0],
          "#00bcd4",
          ["==", ["get", "color"], 1],
          "yellow",
          ["==", ["get", "color"], 2],
          "red",
          "#fff"
        ],
        "line-width": 1.5
      }
    });
  };
  initMapData(map, data, sourceID) {
    map.addSource(sourceID, {
      type: "geojson",
      data
    });
  }

  _renderTooltip = () => {
    const { activeItem, position } = this.state;

    return (
      activeItem && (
        <Popup
          {...position}
          tipSize={10}
          closeOnClick={false}
          onClose={() => this.setState({ activeItem: null })}
        >
          <div className="tooltip">
            Polygon info:
            <div>Date of Image: {activeItem.properties.img_date}</div>
          </div>
        </Popup>
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
  _onClick = event => {
    const { features, lngLat } = event;
    const [longitude, latitude] = lngLat;
    const activeItem =
      features && features.find(f => f.layer.id === "clearcut-polygon");
    console.log(activeItem);
    this.setState({ activeItem, position: { longitude, latitude } });
  };
  _onHover = (event, map) => {
    // console.log(e);
    const { features } = event;
    const hoveredItem =
      features && features.find(f => f.layer.id === "clearcut-polygon");

    // console.log(hoveredItem);
  };
  render() {
    return (
      <ReactMapGL
        {...this.state.viewport}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v9"
        ref={node => (this.map = node)}
        mapboxApiAccessToken="pk.eyJ1IjoiYXZha2luIiwiYSI6ImNqeXk1cTk5aTAwcmszZnA4MjF0d2Fic3AifQ.-KMWEhdsLQ4dQIiC3p0KoA"
        onViewportChange={viewport => {
          this.setState({ viewport });
        }}
        interactiveLayerIds={["clearcut-polygon"]}
        getCursor={this._getCursor}
        onLoad={this._onLoad}
        onClick={this._onClick}
        onHover={this._onHover}
      >
        {this._renderTooltip()}
      </ReactMapGL>
    );
  }
}

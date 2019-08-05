import React from "react";
import {
  Map,
  Marker,
  Popup,
  TileLayer,
  GridLayer,
  WMSTileLayer
} from "react-leaflet";
import "./App.css";
import {} from "leaflet";

function App() {
  return (
    <div className="App">
      <div className="map_holder">
        <UIMap />
      </div>
    </div>
  );
}

export default App;

class UIMap extends React.PureComponent {
  state = {
    lat: 51.505,
    lng: -0.09,
    zoom: 13
  };
  render() {
    const position = [this.state.lat, this.state.lng];
    return (
      <Map center={position} style={{ height: "100%" }} zoom={13}>
        <TileLayer
          opacity={1}
          id="mapbox.satellite"
          url="https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoieGxlYm5hOWFrcm9zaGthIiwiYSI6ImNqeHV0azVxazA0Y3gzZ21udTNoZHAxNjYifQ.FJdUfEfj6P3B9Iuw8zZ57Q"
          attribution='© <a href="https://www.mapbox.com/about/maps/" target="_blank">Mapbox</a> © <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
        />
        {/* <TileLayer
          opacity={0.5}
          zIndex={2}
          id="mapbox.mapbox-streets-v8"
          url="https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoieGxlYm5hOWFrcm9zaGthIiwiYSI6ImNqeHV0azVxazA0Y3gzZ21udTNoZHAxNjYifQ.FJdUfEfj6P3B9Iuw8zZ57Q"
          attribution='© <a href="https://www.mapbox.com/about/maps/" target="_blank">Mapbox</a> © <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
        /> */}
      </Map>
    );
  }
}

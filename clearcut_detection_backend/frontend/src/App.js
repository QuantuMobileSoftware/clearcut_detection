import React from "react";
import "./App.css";
import {} from "leaflet";
import MapWrapper from "./components/Map";
function App() {
  return (
    <div className="App">
      <div className="map_holder">
        <MapWrapper />
      </div>
    </div>
  );
}

export default App;

import React from "react";
import "./App.css";
import {} from "leaflet";
import MapWrapper from "./components/Map";
import DATA from "./data.json";
function App() {
  return (
    <div className="App">
      <div className="map_holder">
        <MapWrapper data={DATA} />
      </div>
    </div>
  );
}

export default App;

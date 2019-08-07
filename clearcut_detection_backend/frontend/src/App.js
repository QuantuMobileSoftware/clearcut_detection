import React from "react";
import "./App.css";
import {} from "leaflet";
import MapWrapper from "./components/Map";
import DATA from "./data.json";
import TEST from "./test.json";
function App() {
  return (
    <div className="App">
      <div className="map_holder">
        <MapWrapper data={TEST} />
      </div>
    </div>
  );
}

export default App;

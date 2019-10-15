import React, { Component } from 'react';
import moment from "moment";

import './App.css';
import MapWrapper from './components/Map';
import Calendar from './components/Calendar';
import LoadingScreen from './components/LoadingScreen/LoadingScreen';
import CustomLegend from './components/CustomLegend/CustomLegend';

import api from './utils/api';
import { URL } from './config/url';
import { DATE_FORMAT, CUSTOM_LEGEND_DATA, CHART_COLORS } from './config';

class App extends Component {
  static fetchData(startDate, endDate) {
    return api
      .get(URL.map.get(startDate, endDate))
      .then(res => res.ok ? res.json() : null);
  }

  static fetchPolygonInfo(id, startDate, endDate) {
    return api
      .get(URL.map.polygon.get(id, startDate, endDate))
      .then(res => res.ok ? res.json() : []);
  }

  static prepareActivePolygonData(data) {
    return data.map((item, i) => {
      const INDEX = i % 3;

      return {
        name: item.image_date,
        y: item.zone_area,
        color: CHART_COLORS[INDEX]
      };
    });
  }

  static parseDate() {
    let search = window.location.search;
    let params = new URLSearchParams(search);
    let startDateParam = parseInt(params.get('startDate'));
    let endDateParam = parseInt(params.get('endDate'));
    return {
      startDate:  moment.unix(startDateParam),
      endDate:  moment.unix(endDateParam),
      isDateSet: !isNaN(startDateParam && endDateParam)
    }
  }

  constructor(props) {
    super(props);

    let {startDate, endDate, isDateSet} = App.parseDate();

    this.state = {
      startDate: isDateSet ? startDate : moment.utc().subtract(120, 'days'),
      endDate: isDateSet ? endDate : moment.utc(),
      data: null,
      activePolygonData: [],
      loading: false,
      activeItem: null,
      focusedFilterInput: null,
      position: {
        longitude: 0,
        latitude: 0
      },
      viewport: {
        width: '100%',
        height: '100%',
        latitude: 49.988358,
        longitude: 36.232845,
        zoom: 9
      },
    };
    this.onDatesChange = this.onDatesChange.bind(this);
    this.onClick = this.onClick.bind(this);
    this.loadData = this.loadData.bind(this);
    this.loadPolygonInfo = this.loadPolygonInfo.bind(this);
  }

  componentDidMount() {
    const { startDate, endDate } = this.state;

    this.loadData(startDate, endDate);
  }

  loadData(startDate, endDate) {
    if (startDate && endDate) {
      this.setState({ loading: true });
      App.fetchData(startDate.format(DATE_FORMAT.default), endDate.format(DATE_FORMAT.default))
         .then(data => this.setState({ data, loading: false, startDate, endDate }))
         .catch(err => console.log(err));
    }
  }

  loadPolygonInfo(id, startDate, endDate) {
    if (id && startDate && endDate) {
      this.setState({ loading: true });
      App.fetchPolygonInfo(id, startDate.format(DATE_FORMAT.default), endDate.format(DATE_FORMAT.default))
         .then(data => this.setState({
           activePolygonData: App.prepareActivePolygonData(data),
           loading: false
         }))
         .catch(err => console.log(err));
    }
  }

  onClick(e) {
    const { startDate, endDate } = this.state;
    const { features, lngLat } = e;
    const [longitude, latitude] = lngLat;
    const activeItem = features && features.find(f => f.layer.id === 'clearcut-polygon');

    if (activeItem) {
      this.loadPolygonInfo(activeItem.properties.pk, startDate, endDate);
    }
    this.setState({ activeItem, position: { longitude, latitude } });
  }

  onDatesChange({ startDate, endDate }) {
    const { activeItem } = this.state;

    if (startDate && endDate) {
      const FORMATTED_START_DATE = startDate.format(DATE_FORMAT.default);
      const FORMATTED_END_DATE = endDate.format(DATE_FORMAT.default);
      const PROMISES = [api.get(URL.map.get(FORMATTED_START_DATE, FORMATTED_END_DATE))];

      if (activeItem) {
        PROMISES.push(api.get(URL.map.polygon.get(activeItem.properties.pk, FORMATTED_START_DATE, FORMATTED_END_DATE)));
      }

      this.setState({ loading: true });

      Promise
        .all(PROMISES)
        .then(([allDataRes, polygonInfoRes]) => Promise.all([
          allDataRes.ok ? allDataRes.json() : null,
          polygonInfoRes && polygonInfoRes.ok ? polygonInfoRes.json() : []
        ]))
        .then(([allData, polygonInfo]) => {
          this.setState({
            data: allData,
            activePolygonData: App.prepareActivePolygonData(polygonInfo),
            loading: false,
            startDate,
            endDate
          });
        });
    } else {
      this.setState({ startDate, endDate });
    }
  }

  render() {
    const { viewport, data, activePolygonData, startDate, endDate, focusedFilterInput, loading, activeItem, position } = this.state;

    return (
      <div className="App">
        <div className="map_holder">
          <MapWrapper
            viewport={viewport}
            data={data}
            startDate={startDate}
            endDate={endDate}
            activePolygonData={activePolygonData}
            activeItem={activeItem}
            position={position}
            onClick={this.onClick}
            onTooltipClose={() => this.setState({ activeItem: null })}
            onViewportChange={viewport => this.setState({ viewport })}
          />
          <Calendar
            startDate={startDate}
            endDate={endDate}
            focusedInput={focusedFilterInput}
            onDatesChange={this.onDatesChange}
            onFocusChange={focusedInput => this.setState({ focusedFilterInput: focusedInput })}
          />
          <CustomLegend data={CUSTOM_LEGEND_DATA} />
        </div>
        {loading && <LoadingScreen />}
      </div>
    );
  }
}

export default App;

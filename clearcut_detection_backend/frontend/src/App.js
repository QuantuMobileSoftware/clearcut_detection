import React, { Component } from 'react';
import ReactGA from 'react-ga';
import moment from 'moment';
import { debounce } from 'lodash';
import { ToastContainer } from 'react-toastify';

import 'react-toastify/dist/ReactToastify.css';

import './App.css';
import Sidebar from './components/Sidebar';
import About from './components/About';
import Calendar from './components/Calendar';
import CustomLegend from './components/CustomLegend/CustomLegend';
import MapWrapper from './components/Map';
import LoadingScreen from './components/LoadingScreen/LoadingScreen';
import links from './constants/links';

import api from './api/api';
import { URL } from './config/url';
import { DATE_FORMAT, CUSTOM_LEGEND_DATA } from './config';
import { fetchData, fetchPolygonInfo, prepareActivePolygonData } from "./api/maps";

const trackingId = process.env.REACT_APP_GOOGLE_ANALYTICS_TRACKING_ID;
ReactGA.initialize(trackingId);
ReactGA.pageview(window.location.pathname + window.location.search);

class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      startDate: moment.utc('2019-04-01'),
      endDate: moment.utc(),
      data: null,
      activePolygonData: [],
      loading: false,
      activeItem: null,
      focusedFilterInput: null,
      isCalendarOpen: false,
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
      }
    };
    this.onDatesChange = this.onDatesChange.bind(this);
    this.onClick = this.onClick.bind(this);
    this.loadData = this.loadData.bind(this);
    this.loadPolygonInfo = this.loadPolygonInfo.bind(this);
  }

  handleResize = () => {
    this.setState(prevState => {
      return {
        viewport: {
          ...prevState.viewport,
          width: window.innerWidth,
          height: window.innerHeight
        }
      };
    });
  };

  componentDidMount() {
    const { startDate, endDate } = this.state;

    this.loadData(startDate, endDate);

    window.addEventListener('resize', debounce(this.handleResize), 300);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.handleResize);
  }

  loadData(startDate, endDate) {
    if (startDate && endDate) {
      this.setState({ loading: true });
      fetchData(
        startDate.format(DATE_FORMAT.default),
        endDate.format(DATE_FORMAT.default)
      )
        .then(data => {
          this.setState({ data, loading: false, startDate, endDate });
        })
        .catch(() => {
          this.setState({ loading: false });
        });
    }
  }

  loadPolygonInfo(id, startDate, endDate) {
    if (id && startDate && endDate) {
      fetchPolygonInfo(
        id,
        startDate.format(DATE_FORMAT.default),
        endDate.format(DATE_FORMAT.default)
      )
        .then(data =>
          this.setState({
            activePolygonData: prepareActivePolygonData(data),
          })
        )
        .catch(err => console.log(err));
    }
  }

  onClick(e) {
    this.setState({ activePolygonData: null, activeItem: null });
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
      const PROMISES = [
        api.get(URL.map.get(FORMATTED_START_DATE, FORMATTED_END_DATE))
      ];

      if (activeItem) {
        PROMISES.push(
          api.get(
            URL.map.polygon.get(
              activeItem.properties.pk,
              FORMATTED_START_DATE,
              FORMATTED_END_DATE
            )
          )
        );
      }

      this.setState({ loading: true });

      Promise.all(PROMISES)
        .then(([allDataRes, polygonInfoRes]) =>
          Promise.all([
            allDataRes.ok ? allDataRes.json() : null,
            polygonInfoRes && polygonInfoRes.ok ? polygonInfoRes.json() : []
          ])
        )
        .then(([allData, polygonInfo]) => {
          this.setState({
            data: allData,
            activePolygonData: prepareActivePolygonData(polygonInfo),
            loading: false,
            startDate,
            endDate
          });
        })
        .catch(() => {
          this.setState({ loading: false, startDate, endDate });
        });
    } else {
      this.setState({ startDate, endDate });
    }
  }

  render() {
    const {
      viewport,
      data,
      activePolygonData,
      startDate,
      endDate,
      focusedFilterInput,
      loading,
      activeItem,
      position,
      isCalendarOpen
    } = this.state;

    return (
      <div className="App">
        <Sidebar handleSidebarClose={() => {}}>
          <Calendar
            startDate={startDate}
            endDate={endDate}
            focusedInput={focusedFilterInput}
            onDatesChange={this.onDatesChange}
            onFocusChange={focusedInput =>
              this.setState({ focusedFilterInput: focusedInput })
            }
            onCalendarIconClick={() => {
              if (!focusedFilterInput && !isCalendarOpen) {
                return this.setState({
                  focusedFilterInput: 'startDate',
                  isCalendarOpen: true
                });
              }
            }}
            onClose={() => {
              setTimeout(() => {
                this.setState({
                  isCalendarOpen: false
                });
              }, 0);
            }}
          />
          <CustomLegend data={CUSTOM_LEGEND_DATA} />
          <div className="spacer"></div>
          <About links={links} />
        </Sidebar>

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
        </div>
        {loading && <LoadingScreen />}
        <ToastContainer />
      </div>
    );
  }
}

export default App;

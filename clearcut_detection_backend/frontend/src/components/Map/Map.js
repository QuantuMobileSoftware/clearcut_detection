import React, { Component } from 'react';
import ReactMapGL, { Popup } from 'react-map-gl';
import Highcharts from 'highcharts';
import NoDataToDisplay from 'highcharts/modules/no-data-to-display';
import HighchartsReact from 'highcharts-react-official';
import darkUnica from 'highcharts/themes/dark-unica';

import './style.css';

NoDataToDisplay(Highcharts);
darkUnica(Highcharts);

export default class Map extends Component {
  state = {
    hoveredItem: null
  };

  componentDidUpdate(prevProps, prevState, snapshot) {
    const { data, startDate, endDate } = this.props;

    if (
      (!prevProps.data && data) ||
      prevProps.startDate !== startDate ||
      prevProps.endDate !== endDate
    ) {
      const MAP = this.map.getMap();

      if (MAP.style.getLayer('clearcut-polygon')) {
        MAP.removeLayer('clearcut-polygon');
      }
      if (MAP.style.getLayer('clearcut-borders')) {
        MAP.removeLayer('clearcut-borders');
      }
      if (MAP.getSource('clearcut')) {
        MAP.removeSource('clearcut');
      }

      this._loadMap();
    }
  }

  _loadMap = () => {
    const MAP = this.map.getMap();

    if (!MAP.getSource('clearcut')) {
      this.initMapData(MAP, this.props.data, 'clearcut');
    }

    if(MAP.style.getLayer('clearcut-polygon')) {
      return;
    }

    MAP.addLayer({
      id: 'clearcut-polygon',
      type: 'fill',
      source: 'clearcut',
      state: {
        hover: true
      },
      paint: {
        'fill-color': [
          'case',
          ['==', ['get', 'color'], 0],
          '#ff394a',
          ['==', ['get', 'color'], 1],
          '#ffed57',
          ['==', ['get', 'color'], 2],
          '#2d4ab9',
          '#fff'
        ],
        'fill-opacity': [
          'case',
          ['boolean', ['feature-state', 'hover'], false],
          1,
          0.5
        ]
      },
      filter: ['==', '$type', 'Polygon']
    });
    MAP.addLayer({
      id: 'clearcut-borders',
      type: 'line',
      source: 'clearcut',
      layout: {},
      paint: {
        'line-color': [
          'case',
          ['==', ['get', 'color'], 0],
          'red',
          ['==', ['get', 'color'], 1], //data is change
          'yellow',
          ['==', ['get', 'color'], 2], //no data
          '#bebebe',
          '#fff'
        ],
        'line-width': 1.5
      }
    });
  };

  initMapData(map, data, sourceID) {
    map.addSource(sourceID, {
      type: 'geojson',
      data
    });
  }

  // Need a refactor
  _renderTooltip = () => {
    const { position, activePolygonData, onTooltipClose } = this.props;
    const options = {
      chart: {
        type: 'column',
        width: 400,
        height: 300,
        backgroundColor: '#343434'
      },
      credits: {
        enabled: false
      },
      title: {
        text: 'Cutting Area Changes',
        style: {
          fontFamily: 'Arial, sans-serif'
        }
      },
      yAxis: {
        title: {
          text: 'Area, ㎡',
          style: {
            fontSize: '16px',
            fontFamily: 'Arial, sans-serif'
          }
        },
        labels: {
          style: {
            fontSize: '12px'
          }
        }
      },
      xAxis: {
        type: 'category',
        labels: {
          rotation: -45,
          style: {
            fontSize: '12px',
            fontFamily: 'Arial, sans-serif'
          }
        },
        width: '90%'
      },
      legend: {
        enabled: false
      },
      lang: {
        noData: 'No Data'
      },
      noData: {
        style: {
          fontFamily: 'Arial, sans-serif',
          fontWeight: 'bold',
          fontSize: '20px'
        }
      },
      tooltip: {
        formatter: function() {
          return `${this.key} <br>Area: <b>${this.y}</b>㎡`;
        },
        style: {
          fontFamily: 'Arial, sans-serif',
          fontSize: '14px'
        }
      },
      series: [
        {
          data: activePolygonData,
          dataLabels: {
            enabled: true,
            format: '{point.y}'
          }
        }
      ]
    };

    return (
      <Popup
        {...position}
        tipSize={10}
        closeOnClick={false}
        onClose={onTooltipClose}
      >
        <div className="tooltip">
          <HighchartsReact highcharts={Highcharts} options={options} />
        </div>
      </Popup>
    );
  };

  _getCursor = cursors => {
    switch (true) {
      case cursors.isDragging:
        return 'grabbing';
      case cursors.isHovering:
        return 'pointer';
      default:
        return 'default';
    }
  };

  render() {
    const { viewport, onClick, onViewportChange, activeItem } = this.props;

    return (
      <ReactMapGL
        {...viewport}
        mapStyle={ process.env.REACT_APP_MAPBOX_STYLE_URL }
        ref={node => (this.map = node)}
        mapboxApiAccessToken={ process.env.REACT_APP_MAPBOX_API_KEY }
        onViewportChange={onViewportChange}
        interactiveLayerIds={['clearcut-polygon']}
        getCursor={this._getCursor}
        onLoad={this._loadMap}
        onClick={onClick}
      >
        {activeItem && this._renderTooltip()}
      </ReactMapGL>
    );
  }
}

import { URL } from "../config/url";
import { CHART_COLORS } from "../config";
import api from "./api"

function fetchData(startDate, endDate) {
    return api
        .get(URL.map.get(startDate, endDate))
        .then(res => (res.ok ? res.json() : null));
}

function fetchPolygonInfo(id, startDate, endDate) {
    return api
        .get(URL.map.polygon.get(id, startDate, endDate))
        .then(res => (res.ok ? res.json() : []));
}

function prepareActivePolygonData(data) {
    return data.map((item, i) => {
        const INDEX = i % 3;

        return {
            name: item.image_date,
            y: item.zone_area,
            color: CHART_COLORS[INDEX]
        };
    });
}

export {
    fetchData,
    fetchPolygonInfo,
    prepareActivePolygonData
}
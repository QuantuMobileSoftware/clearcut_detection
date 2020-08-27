
export const URL_PREFIX = '/api';

export const URL = {
  map: {
    get: (startDate, endDate) => `${URL_PREFIX}/clearcuts/info/${startDate}/${endDate}`,
    polygon: {
      get: (id, startDate, endDate) => `${URL_PREFIX}/clearcuts/area_chart/${id}/${startDate}/${endDate}`
    }
  }
};
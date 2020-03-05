
export const URL_PREFIX = '/api';

export const URL = {
  map: {
    get: (startDate, endDate) => `${URL_PREFIX}/clearcuts_info/${startDate}/${endDate}`,
    polygon: {
      get: (id, startDate, endDate) => `${URL_PREFIX}/clearcut_area_chart/${id}/${startDate}/${endDate}`
    }
  }
};
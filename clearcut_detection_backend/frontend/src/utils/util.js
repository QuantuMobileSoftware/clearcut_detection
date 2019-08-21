export function dataToQueryParams(data) {
  if (typeof data !== 'object' || Array.isArray(data) || data === null) {
    return '';
  }

  return Object
    .keys(data)
    .map(key => {
      if (Array.isArray(data[key]) && data[key].length) {
        return `${key}=${data[key].join(',')}`;
      }
      if (~['object', 'undefined'].indexOf(typeof data[key])) {
        return null;
      }
      return `${key}=${data[key]}`;
    })
    .filter(str => str)
    .join('&');
}
import { dataToQueryParams } from './util';

const DEFAULT_HEADERS = () => ({
  'Content-Type': 'application/json'
});
const REQUIRED_HEADERS = {
  'X-Requested-With': 'XMLHttpRequest'
};

export default {
  get(path, headers = null, body) {
    const HEADERS = { ...DEFAULT_HEADERS(), ...headers, ...REQUIRED_HEADERS };
    const DATA = dataToQueryParams(body);
    const PREFIX = /\?/.test(path) ? '&' : '?';

    return fetch(`${path}${DATA ? `${PREFIX}${DATA}` : ''}`, {
      credentials: 'same-origin',
      headers: HEADERS
    });
  }
};

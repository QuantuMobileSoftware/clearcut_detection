import { dataToQueryParams } from '../utils/util';

const DEFAULT_HEADERS = () => ({
  'Content-Type': 'application/json'
});
const REQUIRED_HEADERS = {
  'X-Requested-With': 'XMLHttpRequest'
};

export default {
  async get(path, headers = null, body) {
    const HEADERS = { ...DEFAULT_HEADERS(), ...headers, ...REQUIRED_HEADERS };
    const DATA = dataToQueryParams(body);
    const PREFIX = /\?/.test(path) ? '&' : '?';

    const response = await fetch(`${path}${DATA ? `${PREFIX}${DATA}` : ''}`, {
      credentials: 'same-origin',
      headers: HEADERS
    });

    if (response.status === 404) throw new Error('404');

    return response;
  }
};

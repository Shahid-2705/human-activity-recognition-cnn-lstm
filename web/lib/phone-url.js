import net from 'node:net';

function isLocalIPv4(ip) {
  const parts = ip.split('.').map(Number);
  if (parts.length !== 4 || parts.some((n) => isNaN(n) || n < 0 || n > 255)) return false;
  const [a, b] = parts;
  if (a === 127) return true; // 127.0.0.0/8
  if (a === 10) return true; // 10.0.0.0/8
  if (a === 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12
  if (a === 192 && b === 168) return true; // 192.168.0.0/16
  if (a === 169 && b === 254) return true; // 169.254.0.0/16
  return false;
}

function isLocalIPv6(ip) {
  const clean = ip.toLowerCase();
  if (clean === '::1') return true;
  if (/^fe[89ab]/i.test(clean)) return true; // link-local fe80::/10
  if (/^f[cd]/i.test(clean)) return true; // unique local fc00::/7
  return false;
}

export function isLocalHost(host) {
  if (!host || typeof host !== 'string') return false;
  let h = host.toLowerCase();
  if (h.startsWith('[') && h.endsWith(']')) {
    h = h.slice(1, -1);
  }
  if (h === 'localhost' || h.endsWith('.local')) return true;
  if (net.isIPv4(h)) return isLocalIPv4(h);
  if (net.isIPv6(h)) return isLocalIPv6(h);
  return false;
}

export function parsePhoneUrl(urlString) {
  if (!urlString || typeof urlString !== 'string') return null;
  let parsed;
  try {
    parsed = new URL(urlString);
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  if (parsed.username || parsed.password) return null;
  if (!isLocalHost(parsed.hostname)) return null;
  return parsed.href;
}

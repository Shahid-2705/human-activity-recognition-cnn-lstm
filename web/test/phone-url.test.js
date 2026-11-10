import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePhoneUrl, isLocalHost } from '../lib/phone-url.js';

test('accepts typical phone-camera URLs on the local network', () => {
  for (const u of [
    'http://192.168.1.100:4747/video',
    'http://10.0.0.5:8080/video',
    'http://172.16.4.9:8080/video',
    'http://172.31.255.1/video',
    'http://localhost:4747/video',
    'http://127.0.0.1:4747/mjpegfeed',
    'http://my-phone.local:8080/video',
    'https://192.168.0.20:8080/video',
  ]) assert.ok(parsePhoneUrl(u), u);
});

test('rejects internet hosts, odd schemes, credentials and garbage', () => {
  for (const u of [
    'http://example.com/video',
    'http://8.8.8.8/video',
    'http://172.32.0.1/video',       // just outside 172.16/12
    'http://192.169.1.1/video',
    'http://169.253.1.1/video',
    'file:///etc/passwd',
    'ftp://192.168.1.2/x',
    'javascript:alert(1)',
    'http://user:pw@192.168.1.5/video',
    'http://192.168.1.5.evil.com/video',
    'http://localhost.evil.com/video',
    'not a url', '', undefined,
  ]) assert.equal(parsePhoneUrl(u), null, String(u));
});

test('IPv4 shorthand is normalised before the check, not after', () => {
  assert.equal(parsePhoneUrl('http://0x08.0x08.0x08.0x08/'), null);   // 8.8.8.8
  assert.ok(parsePhoneUrl('http://0xc0.0xa8.1.5/'));                   // 192.168.1.5
});

test('IPv6 loopback / link-local / unique-local are local, public is not', () => {
  assert.ok(isLocalHost('::1'));
  assert.ok(isLocalHost('fe80::1'));
  assert.ok(isLocalHost('fd12:3456::1'));
  assert.equal(isLocalHost('2001:4860:4860::8888'), false);
});

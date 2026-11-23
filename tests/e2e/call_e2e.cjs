// End-to-end test of the call page in a real Chromium with a fake microphone.
//   node tests/e2e/call_e2e.cjs [scenario]          (needs Node + playwright + a Chromium; dev only)
// The speech recogniser is scripted (the server runs with --script), everything else is real: browser voice-activity
// detection, WAV encoding, HTTP, the call controller, the store.
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs'), os = require('os'), path = require('path'), http = require('http');

const ROOT = path.resolve(__dirname, '..', '..');
const SCENARIOS = {
  happy: {
    script: ['yes', 'my name is anil kumar', 'yes', 'I am twenty five years old', 'yes', 'male', 'yes',
             'nine eight seven six five four three two one zero', 'yes', 'o negative', 'yes', 'yes'],
    expect: { outcome: 'completed', values: { name: 'Anil Kumar', age: '25', gender: 'Male', phone: '9876543210', blood_group: 'O-' } },
  },
  silent: {
    mic: 'silent', script: ['yes'],
    expect: { outcome: 'timed_out', values: { name: null, age: null, gender: null, phone: null, blood_group: null } },
  },
  correction: {
    script: ['yes', 'my name is anil kumar', 'no, say again', 'my name is sunil kumar', 'yes', 'I am twenty five', 'no, it is twenty six', 'yes',
             'female', 'yes', '9876543210', 'yes', 'skip', 'yes'],
    expect: { outcome: 'completed', values: { name: 'Sunil Kumar', age: '26', gender: 'Female', phone: '9876543210', blood_group: null } },
  },
};

function get(url, headers) { return new Promise((res, rej) => http.get(url, { headers }, r => { let b = ''; r.on('data', d => b += d); r.on('end', () => res(JSON.parse(b))); }).on('error', rej)); }

(async () => {
  const name = process.argv[2] || 'happy'; const sc = SCENARIOS[name];
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'svara_e2e_'));
  const wav = path.join(tmp, 'mic.wav'), script = path.join(tmp, 'script.json'), calls = path.join(tmp, 'calls');
  execFileSync('python3', [path.join(__dirname, 'make_fake_mic.py'), wav].concat(sc.mic === 'silent' ? ['--silent'] : []));
  fs.writeFileSync(script, JSON.stringify(sc.script));
  const port = 8700 + Math.floor(Math.random() * 200);
  const srv = spawn('python3', ['-m', 'callpage', '--port', String(port), '--script', script, '--no-tts', '--calls', calls, '--forms', path.join(ROOT, 'data', 'forms')], { cwd: ROOT });
  let out = ''; srv.stdout.on('data', d => out += d); srv.stderr.on('data', d => out += d);
  const fail = async (msg, browser) => { console.error('FAIL:', msg); console.error(out.slice(-1500)); srv.kill(); if (browser) await browser.close(); process.exit(1); };
  for (let i = 0; i < 50 && !out.includes('SVARA call page'); i++) await new Promise(r => setTimeout(r, 100));
  if (!out.includes('SVARA call page')) return fail('server did not start');

  const browser = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`] });
  const ctx = await browser.newContext({ permissions: ['microphone'] });
  const page = await ctx.newPage();
  page.on('pageerror', e => console.error('PAGE ERROR:', e.message));
  page.on('console', m => { if (m.type() === 'error') console.error('CONSOLE ERROR:', m.text()); });
  const url = `http://127.0.0.1:${port}/?test=1`;
  const t0 = Date.now();
  await page.goto(url);
  await page.waitForFunction(() => document.querySelectorAll('#form option').length > 0, null, { timeout: 10000 });
  await page.click('#start');                                   // the ONLY click of the whole call
  try { await page.waitForFunction(() => window.__finished !== null, null, { timeout: 170000, polling: 500 }); }
  catch (e) { const log = await page.evaluate(() => window.__log); console.error('LOG SO FAR:', JSON.stringify(log.slice(-8))); return fail('call did not finish in time', browser); }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  const fin = await page.evaluate(() => window.__finished), spoken = await page.evaluate(() => window.__spoken), log = await page.evaluate(() => window.__log);
  const token = await page.evaluate(() => document.querySelector('meta[name=call-token]').content);
  await page.screenshot({ path: process.env.SHOT || path.join(tmp, 'end.png'), fullPage: true });
  const rec = await get(`http://127.0.0.1:${port}/api/call/${fin.callId}`, { 'X-Call-Token': token });
  await browser.close(); srv.kill();

  const errs = [];
  if (fin.outcome !== sc.expect.outcome) errs.push(`outcome ${fin.outcome} != ${sc.expect.outcome}`);
  for (const [k, v] of Object.entries(sc.expect.values)) { const f = rec.fields.find(x => x.name === k); if (!f || f.value !== v) errs.push(`${k}: ${f && f.value} != ${v}`); }
  if (rec.outcome !== sc.expect.outcome) errs.push(`stored outcome ${rec.outcome}`);
  console.log(`scenario "${name}" finished in ${secs}s: ${spoken.length} sentences spoken, ${log.filter(x => x[0] === 'you').length} user turns heard`);
  console.log('stored values:', JSON.stringify(Object.fromEntries(rec.fields.map(f => [f.name, f.value]))));
  console.log('first spoken:', spoken[0].slice(0, 80)); console.log('last spoken :', spoken[spoken.length - 1]);
  if (errs.length) { console.error('FAIL:', errs.join('; ')); process.exit(1); }
  console.log('PASS');
})().catch(e => { console.error('CRASH:', e); process.exit(1); });

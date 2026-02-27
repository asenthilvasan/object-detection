import http from 'k6/http';
import { check } from 'k6';
import { scenario as execScenario } from 'k6/execution';
import { Trend, Counter, Rate } from 'k6/metrics';

/** ======== CONFIG (kept as requested) ======== */
const RPS_LIST = [1, 5, 10, 20, 30, 40, 50, 60];
const DURATION = '30s';        // per stage
const TIME_UNIT = '1s';
const PREALLOC_VUS = 400;
const MAX_VUS = 400;
const GRACEFUL_STOP = '0s';    // hard stop so stages don't bleed into each other
const START_GAP_SECONDS = 120; // cooldown so the server drains fully

const BASE_URL =
  __ENV.OD_BASE_URL || 'http://object-detection.ml-pipelines.svc.cluster.local:8000';

/** ======== Build sequential scenarios ======== */
const scenarios = {};
RPS_LIST.forEach((rps, i) => {
  scenarios[`rps_${rps}`] = {
    executor: 'constant-arrival-rate',
    rate: rps,
    timeUnit: TIME_UNIT,
    duration: DURATION,
    preAllocatedVUs: PREALLOC_VUS,
    maxVUs: MAX_VUS,
    gracefulStop: GRACEFUL_STOP,
    startTime: `${i * START_GAP_SECONDS}s`,
  };
});
export const options = { scenarios };

/** ======== Per-RPS custom metrics ======== */
const dur = {}, done = {}, ok = {};
for (const rps of RPS_LIST) {
  dur[rps]  = new Trend(`http_duration_rps_${rps}`, true); // ms
  done[rps] = new Counter(`http_done_rps_${rps}`);          // completed requests (any result)
  ok[rps]   = new Rate(`http_ok_rate_rps_${rps}`);          // success rate
}

/** ======== Load test image from disk ======== */
// When running in k8s, the init container downloads the image to /data/test_image.jpg.
// When running locally, point to a local file or override with: k6 run -e IMAGE_PATH=./test_image.jpg ...
const IMAGE_PATH = __ENV.IMAGE_PATH || '/data/test_image.jpg';
const imageBytes = open(IMAGE_PATH, 'b');

/** ======== Test function (one HTTP request per iteration) ======== */
export default () => {
  const t0 = Date.now();
  let success = false;

  try {
    const payload = {
      file: http.file(imageBytes, 'test.jpg', 'image/jpeg'),
    };

    const res = http.post(`${BASE_URL}/detect-upload`, payload, {
      timeout: '30s',
    });

    success = check(res, {
      'status 200':         (r) => r.status === 200,
      'content-type image': (r) => (r.headers['Content-Type'] || '').includes('image'),
    });
  } catch (_) {
    success = false;
  }

  const dt = Date.now() - t0;

  const scen = execScenario.name;      // e.g., "rps_80"
  const rps = Number(scen.split('_')[1]);
  if (!Number.isNaN(rps)) {
    dur[rps].add(dt);
    done[rps].add(1);       // one request completed
    ok[rps].add(success);   // success/fail for this request
  }
};

/** ======== Helpers ======== */
function parseDurationSeconds(str) {
  const re = /(\d+)(ms|s|m|h)/g;
  let total = 0, m;
  while ((m = re.exec(str)) !== null) {
    const v = Number(m[1]);
    total += m[2] === 'ms' ? v / 1000
         :  m[2] === 's'  ? v
         :  m[2] === 'm'  ? v * 60
         :                   v * 3600;
  }
  return Math.round(total);
}
const DURATION_SECS = parseDurationSeconds(DURATION);

/** ======== Summary → CSV ======== */
export function handleSummary(data) {
  let csv = 'rps,p95_ms,avg_ms,iterations,ok_count,fail_count,iter_dropped,ok_rate,scheduled,completion_pct\n';

  for (const rps of RPS_LIST) {
    const mDur  = data.metrics[`http_duration_rps_${rps}`];
    const mDone = data.metrics[`http_done_rps_${rps}`];
    const mOk   = data.metrics[`http_ok_rate_rps_${rps}`];

    const p95 = mDur?.values?.['p(95)'];
    const avg = mDur?.values?.['avg'];

    // completed = successes + failures (only those that finished)
    const iterations = (mDone?.values?.count ?? mDone?.values?.sum ?? 0);
    const okCount    = (mOk?.values?.passes ?? 0);
    const failCount  = (mOk?.values?.fails ?? Math.max(0, iterations - okCount));
    const okRate     = (mOk?.values?.rate ?? (iterations ? okCount / iterations : 0));

    // scheduled = target RPS × stage duration (in seconds)
    const scheduled  = rps * DURATION_SECS;

    // dropped = scheduled - completed (never negative)
    const iterDropped = Math.max(0, scheduled - iterations);

    const completion = scheduled ? (iterations / scheduled) : 0;

    csv += [
      rps,
      p95 !== undefined ? p95.toFixed(2) : '',
      avg !== undefined ? avg.toFixed(2) : '',
      iterations,
      okCount,
      failCount,
      iterDropped,
      okRate.toFixed(4),
      scheduled,
      (completion * 100).toFixed(1)
    ].join(',') + '\n';
  }

  return {
    'p95_by_rps.csv': csv,
    // omit stdout to keep the console clean; run with --no-progress too
  };
}

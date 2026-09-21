const $ = (id) => document.getElementById(id);
const formatTime = (value) => value ? new Date(value).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}) : '--';
let feedRunning = false;

function render(data) {
  const isRunning = Boolean(data.video?.running);

  $('mode').textContent = isRunning ? 'AI ACTIVE' : 'SYSTEM IDLE';
  $('source').textContent = data.source || '--';
  $('updated').textContent = data.updated_at ? `Telemetry ${formatTime(data.updated_at)}` : 'Waiting for telemetry';
  $('camera-id').textContent = data.camera_id;
  const status = data.video?.status || 'IDLE';
  $('camera-status').textContent = isRunning ? 'ANALYZING' : (status === 'IDLE' ? 'CAMERA OFFLINE' : status);
  $('monitoring-banner').textContent = isRunning ? 'AI ACTIVE' : (status === 'CCTV VIDEO FINISHED' ? 'VIDEO FINISHED' : 'MONITORING OFF');
  if (isRunning && !feedRunning) $('live-feed').src = `/video_feed?t=${Date.now()}`;
  if (!isRunning) $('live-feed').removeAttribute('src');
  feedRunning = isRunning;
  $('fps').textContent = data.video?.fps ? data.video.fps.toFixed(1) : '--';
  $('detections').textContent = data.counts?.detections ?? 0;
  $('people').textContent = data.counts?.people ?? 0;
  $('vehicles').textContent = data.counts?.vehicles ?? 0;

  // ============================================================
// TRACKED OBJECTS
// ============================================================

const tracks = Array.isArray(data.tracks)
  ? data.tracks
  : [];

$('tracks').innerHTML = tracks.length
  ? tracks.map((track) => {

      const trackId =
        track.person_id ??
        track.track_id ??
        '--';

      const className =
        track.class_name ??
        track.label ??
        track.class ??
        'object';

      const movement =
        track.movement ??
        track.status ??
        'stationary';

      const confidence =
        Number(track.confidence ?? 0);

      return `
        <div class="track">

          <span>
            <b>#${trackId}</b>
            ${className}
          </span>

          <span>
            ${movement}
            •
            ${Math.round(confidence * 100)}%
          </span>

        </div>
      `;

    }).join('')
  : '<span class="muted">No objects in frame</span>';


// ============================================================
// ALERTS
// ============================================================

// Current events are the live alerts.
// Recent alerts are used as a fallback so the dashboard
// does not immediately show ALERT NONE after an alert is created.

const currentAlerts =
  Array.isArray(data.current_events)
    ? data.current_events
    : [];

const recentAlerts =
  Array.isArray(data.alerts)
    ? data.alerts
    : [];

const alertList =
  currentAlerts.length > 0
    ? currentAlerts
    : recentAlerts;


// ------------------------------------------------------------
// CURRENT ALERT METRIC
// ------------------------------------------------------------

$('alerts').textContent =
  alertList.length > 0
    ? 'ALERT ACTIVE'
    : 'ALERT NONE';


// ------------------------------------------------------------
// ALERT PANEL
// ------------------------------------------------------------

if (alertList.length > 0) {

  $('events').innerHTML = alertList
    .map((event) => {

      const eventType =
        event.event_type ||
        'Virtual Zone Violation';

      const trackingId =
        event.tracking_id ??
        event.track_id ??
        '--';

      const severity =
        event.severity ||
        'MEDIUM';

      const confidence =
        event.confidence != null
          ? `${Math.round(Number(event.confidence) * 100)}%`
          : '--';

      return `
        <div class="event alert-event">

          <strong>
            🚨 ${eventType}
          </strong>

          <span>
            Track #${trackingId}
            • ${severity}
            • ${confidence}
          </span>

          <time>
            ${formatTime(event.timestamp)}
          </time>

        </div>
      `;

    })
    .join('');

} else {

  $('events').innerHTML =
    '<span class="muted">No active threat</span>';

}

}
// ============================================================
// STATUS REFRESH
// ============================================================

async function refresh() {

  try {

    const response =
      await fetch('/api/status');

    if (!response.ok) {
      throw new Error(
        `Status request failed: ${response.status}`
      );
    }

    const data =
      await response.json();

    render(data);

  } catch (error) {

    console.error(
      '[STATUS] Refresh failed:',
      error
    );

    $('mode').textContent =
      'OFFLINE';
  }
}


// ============================================================
// START WEBCAM / CCTV MONITORING
// ============================================================

$('start-camera').addEventListener(
  'click',
  async () => {

    console.log(
      '[UI] Start CCTV monitoring clicked'
    );

    try {

      const response =
        await fetch(
          '/api/monitoring/start',
          {
            method: 'POST'
          }
        );

      const result =
        await response.json();

      console.log(
        '[UI] Start response:',
        result
      );

      $('input-message').textContent =
        response.ok
          ? result.message
          : (
              result.error ||
              'Could not start camera.'
            );

      await refresh();

    } catch (error) {

      console.error(
        '[UI] Camera start failed:',
        error
      );

      $('input-message').textContent =
        `Camera start failed: ${error.message}`;
    }
  }
);


// ============================================================
// STOP MONITORING
// ============================================================

$('stop-camera').addEventListener(
  'click',
  async () => {

    console.log(
      '[UI] Stop monitoring clicked'
    );

    try {

      const response =
        await fetch(
          '/api/monitoring/stop',
          {
            method: 'POST'
          }
        );

      const result =
        await response.json();

      $('input-message').textContent =
        response.ok
          ? result.message
          : (
              result.error ||
              'Could not stop monitoring.'
            );

      await refresh();

    } catch (error) {

      console.error(
        '[UI] Stop failed:',
        error
      );

      $('input-message').textContent =
        `Stop failed: ${error.message}`;
    }
  }
);


// ============================================================
// SELECT CCTV VIDEO
// ============================================================

$('cctv-video-file').addEventListener(
  'change',
  (event) => {

    const file =
      event.target.files[0];

    $('selected-video').textContent =
      file
        ? `Selected file: ${file.name}`
        : 'No file selected';
  }
);


// ============================================================
// UPLOAD + ANALYZE CCTV VIDEO
// ============================================================

$('upload-cctv-video').addEventListener(
  'click',
  async () => {

    const file =
      $('cctv-video-file').files[0];

    if (!file) {

      $('input-message').textContent =
        'Choose a CCTV video first.';

      return;
    }

    try {

      $('input-message').textContent =
        'Uploading CCTV video...';

      console.log(
        '[VIDEO] Uploading:',
        file.name
      );

      const form =
        new FormData();

      form.append(
        'video',
        file
      );

      const uploadResponse =
        await fetch(
          '/api/video/upload',
          {
            method: 'POST',
            body: form
          }
        );

      const uploadResult =
        await uploadResponse.json();

      console.log(
        '[VIDEO] Upload response:',
        uploadResult
      );

      if (!uploadResponse.ok) {

        throw new Error(
          uploadResult.error ||
          'Video upload failed.'
        );
      }

      $('input-message').textContent =
        'Video uploaded. Starting AI analysis...';


      // --------------------------------------------------------
      // START ANALYSIS
      // --------------------------------------------------------

      const startForm =
        new FormData();

      startForm.append(
        'filename',
        uploadResult.filename
      );

      const startResponse =
        await fetch(
          '/api/monitoring/start-video',
          {
            method: 'POST',
            body: startForm
          }
        );

      const startResult =
        await startResponse.json();

      console.log(
        '[VIDEO] Start response:',
        startResult
      );

      $('input-message').textContent =
        startResponse.ok
          ? startResult.message
          : (
              startResult.error ||
              'Could not start CCTV video analysis.'
            );

      await refresh();

    } catch (error) {

      console.error(
        '[VIDEO] CCTV video analysis failed:',
        error
      );

      $('input-message').textContent =
        `Video analysis failed: ${error.message}`;
    }
  }
);


// ============================================================
// RTSP CCTV
// ============================================================

$('rtsp-form').addEventListener(
  'submit',
  async (event) => {

    event.preventDefault();

    console.log(
      '[RTSP] Starting RTSP CCTV'
    );

    try {

      const response =
        await fetch(
          '/api/rtsp/start',
          {
            method: 'POST',
            body: new FormData(event.target)
          }
        );

      const result =
        await response.json();

      console.log(
        '[RTSP] Response:',
        result
      );

      $('input-message').textContent =
        response.ok
          ? result.message
          : (
              result.error ||
              'Could not start RTSP CCTV.'
            );

      await refresh();

    } catch (error) {

      console.error(
        '[RTSP] Start failed:',
        error
      );

      $('input-message').textContent =
        `RTSP start failed: ${error.message}`;
    }
  }
);


// ============================================================
// INITIAL STATUS LOAD
// ============================================================

refresh();


// ============================================================
// AUTOMATIC DASHBOARD REFRESH
// ============================================================

setInterval(
  refresh,
  1500
);


// ============================================================
// VIRTUAL RESTRICTED ZONE
// ============================================================

let restrictedZonePoints = [];
let restrictedZoneDrawing = false;
let restrictedZoneCanvas = null;
let restrictedZoneContext = null;


// ============================================================
// INITIALIZE
// ============================================================

function initializeRestrictedZone() {

  const videoElement = $('live-feed');

  if (!videoElement) {
    console.warn('[ZONE] #live-feed not found.');
    return;
  }

  const parent = videoElement.parentElement;

  if (!parent) {
    console.warn('[ZONE] Video parent not found.');
    return;
  }

  // Make parent suitable for canvas overlay
  if (getComputedStyle(parent).position === 'static') {
    parent.style.position = 'relative';
  }

  restrictedZoneCanvas =
    document.getElementById('restricted-zone-canvas');

  if (!restrictedZoneCanvas) {

    restrictedZoneCanvas =
      document.createElement('canvas');

    restrictedZoneCanvas.id =
      'restricted-zone-canvas';

    restrictedZoneCanvas.style.position =
      'absolute';

    restrictedZoneCanvas.style.left = '0';
    restrictedZoneCanvas.style.top = '0';

    restrictedZoneCanvas.style.width = '100%';
    restrictedZoneCanvas.style.height = '100%';

    restrictedZoneCanvas.style.zIndex = '50';

    restrictedZoneCanvas.style.pointerEvents =
      'none';

    parent.appendChild(
      restrictedZoneCanvas
    );
  }

  restrictedZoneContext =
    restrictedZoneCanvas.getContext('2d');

  resizeRestrictedZoneCanvas();

  window.addEventListener(
    'resize',
    resizeRestrictedZoneCanvas
  );

  // Click on CCTV frame
  restrictedZoneCanvas.addEventListener(
    'click',
    handleRestrictedZoneClick
  );

  loadRestrictedZone();

  console.log(
    '[ZONE] Restricted zone initialized.'
  );
}


// ============================================================
// RESIZE CANVAS
// ============================================================

function resizeRestrictedZoneCanvas() {

  if (!restrictedZoneCanvas) {
    return;
  }

  const rect =
    restrictedZoneCanvas.getBoundingClientRect();

  const dpr =
    window.devicePixelRatio || 1;

  restrictedZoneCanvas.width =
    rect.width * dpr;

  restrictedZoneCanvas.height =
    rect.height * dpr;

  restrictedZoneContext.setTransform(
    dpr,
    0,
    0,
    dpr,
    0,
    0
  );

  drawRestrictedZone();
}


// ============================================================
// START DRAWING
// ============================================================

function startRestrictedZone() {

  if (!restrictedZoneCanvas) {
    initializeRestrictedZone();
  }

  restrictedZonePoints = [];

  restrictedZoneDrawing = true;

  restrictedZoneCanvas.style.pointerEvents =
    'auto';

  drawRestrictedZone();

  $('input-message').textContent =
    'Click points on the CCTV frame to draw the restricted zone.';

  console.log(
    '[ZONE] Drawing started.'
  );
}


// ============================================================
// HANDLE CLICK
// ============================================================

function handleRestrictedZoneClick(event) {

  if (!restrictedZoneDrawing) {
    return;
  }

  const rect =
    restrictedZoneCanvas.getBoundingClientRect();

  const x =
    event.clientX - rect.left;

  const y =
    event.clientY - rect.top;

  restrictedZonePoints.push([
    x,
    y
  ]);

  drawRestrictedZone();

  console.log(
    '[ZONE] Point:',
    x,
    y
  );
}


// ============================================================
// DRAW ZONE
// ============================================================

function drawRestrictedZone() {

  if (
    !restrictedZoneCanvas ||
    !restrictedZoneContext
  ) {
    return;
  }

  const ctx =
    restrictedZoneContext;

  const width =
    restrictedZoneCanvas.clientWidth;

  const height =
    restrictedZoneCanvas.clientHeight;

  ctx.clearRect(
    0,
    0,
    width,
    height
  );

  if (
    restrictedZonePoints.length === 0
  ) {
    return;
  }

  // ----------------------------------------------------------
  // POLYGON
  // ----------------------------------------------------------

  if (
    restrictedZonePoints.length >= 2
  ) {

    ctx.beginPath();

    ctx.moveTo(
      restrictedZonePoints[0][0],
      restrictedZonePoints[0][1]
    );

    for (
      let i = 1;
      i < restrictedZonePoints.length;
      i++
    ) {

      ctx.lineTo(
        restrictedZonePoints[i][0],
        restrictedZonePoints[i][1]
      );
    }

    if (
      restrictedZonePoints.length >= 3
    ) {
      ctx.closePath();
    }

    // Restricted area
    ctx.fillStyle =
      'rgba(255, 0, 0, 0.20)';

    if (
      restrictedZonePoints.length >= 3
    ) {
      ctx.fill();
    }

    // Border
    ctx.strokeStyle =
      '#ff0000';

    ctx.lineWidth = 3;

    ctx.stroke();
  }


  // ----------------------------------------------------------
  // POINTS
  // ----------------------------------------------------------

  restrictedZonePoints.forEach(
    (point, index) => {

      ctx.beginPath();

      ctx.arc(
        point[0],
        point[1],
        6,
        0,
        Math.PI * 2
      );

      ctx.fillStyle =
        '#ffffff';

      ctx.fill();

      ctx.strokeStyle =
        '#ff0000';

      ctx.lineWidth = 3;

      ctx.stroke();

      // Point number
      ctx.fillStyle =
        '#ffffff';

      ctx.font =
        'bold 12px Arial';

      ctx.fillText(
        String(index + 1),
        point[0] + 9,
        point[1] - 9
      );
    }
  );
}


// ============================================================
// SAVE ZONE
// ============================================================

async function saveRestrictedZone() {

  if (
    restrictedZonePoints.length < 3
  ) {

    $('input-message').textContent =
      'Select at least 3 points for the restricted zone.';

    return;
  }

  try {

    const response =
      await fetch(
        '/api/zone',
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json'
          },

          body: JSON.stringify({
            points:
              restrictedZonePoints
          })
        }
      );

    const result =
      await response.json();

    if (!response.ok) {

      throw new Error(
        result.error ||
        'Failed to save restricted zone.'
      );
    }

    restrictedZoneDrawing = false;

    restrictedZoneCanvas.style.pointerEvents =
      'none';

    $('input-message').textContent =
      'Restricted zone saved successfully.';

    drawRestrictedZone();

    refresh();

    console.log(
      '[ZONE] Saved:',
      result
    );

  } catch (error) {

    console.error(
      '[ZONE] Save failed:',
      error
    );

    $('input-message').textContent =
      `Zone save failed: ${error.message}`;
  }
}


// ============================================================
// LOAD ZONE
// ============================================================

async function loadRestrictedZone() {

  try {

    const response =
      await fetch('/api/zone');

    const result =
      await response.json();

    if (
      result.configured &&
      Array.isArray(result.points)
    ) {

      restrictedZonePoints =
        result.points;

      drawRestrictedZone();

      console.log(
        '[ZONE] Existing zone loaded.'
      );
    }

  } catch (error) {

    console.error(
      '[ZONE] Could not load zone:',
      error
    );
  }
}


// ============================================================
// CLEAR ZONE
// ============================================================

async function clearRestrictedZone() {

  try {

    const response =
      await fetch(
        '/api/zone',
        {
          method: 'DELETE'
        }
      );

    const result =
      await response.json();

    if (!response.ok) {

      throw new Error(
        result.error ||
        'Failed to clear zone.'
      );
    }

    restrictedZonePoints = [];

    restrictedZoneDrawing = false;

    if (restrictedZoneCanvas) {

      restrictedZoneCanvas.style.pointerEvents =
        'none';
    }

    drawRestrictedZone();

    $('input-message').textContent =
      'Restricted zone cleared.';

    refresh();

  } catch (error) {

    console.error(
      '[ZONE] Clear failed:',
      error
    );

    $('input-message').textContent =
      `Could not clear zone: ${error.message}`;
  }
}


// ============================================================
// CANCEL DRAWING
// ============================================================

function cancelRestrictedZone() {

  restrictedZonePoints = [];

  restrictedZoneDrawing = false;

  if (restrictedZoneCanvas) {

    restrictedZoneCanvas.style.pointerEvents =
      'none';
  }

  drawRestrictedZone();

  $('input-message').textContent =
    'Restricted zone drawing cancelled.';
}


// ============================================================
// ESC = CANCEL
// ============================================================

document.addEventListener(
  'keydown',
  (event) => {

    if (
      event.key === 'Escape' &&
      restrictedZoneDrawing
    ) {

      cancelRestrictedZone();
    }
  }
);


// ============================================================
// INITIALIZE AFTER PAGE LOAD
// ============================================================

window.addEventListener(
  'load',
  () => {

    setTimeout(
      initializeRestrictedZone,
      500
    );

  }
);
// ============================================================
// RESTRICTED ZONE BUTTONS
// ============================================================

$('draw-zone').addEventListener(
  'click',
  () => {

    startRestrictedZone();

    $('zone-message').textContent =
      'Click 3 or more points on the CCTV feed.';
  }
);


$('save-zone').addEventListener(
  'click',
  async () => {

    await saveRestrictedZone();

    $('zone-message').textContent =
      'Restricted zone saved.';
  }
);


$('clear-zone').addEventListener(
  'click',
  async () => {

    await clearRestrictedZone();

    $('zone-message').textContent =
      'Restricted zone cleared.';
  }
);

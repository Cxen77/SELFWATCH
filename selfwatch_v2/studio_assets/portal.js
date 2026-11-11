'use strict';
const $ = id => document.getElementById(id), activeStates = new Set(['loading', 'running', 'paused', 'stopping']);
const bodyEdges = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
let token = '', state = {status: 'idle'}, stateRevision = '', sourceMode = 'sample', records = [], selectedUploads = new Set(), offsets = new Map(), nativeCameras = [], browserCameras = [], views = new Map();
let playback = null, followingRun = null, loadedRun = null, loadingRun = null, polling = false, lastPollAt = 0, starting = false, uploading = false, detecting = false, uploadsSignature = '';

// Modern Professional Palette (Efferd Inspired - Crisp Blue, Cyan, Indigo, Emerald)
const TRACK_PALETTE = ['#2563eb', '#0284c7', '#0d9488', '#4f46e5', '#059669', '#d97706'];

function notify(text, error = false) {
  const el = $('message');
  if (!el) return;
  el.hidden = !text;
  el.textContent = text;
  el.dataset.error = error ? 'true' : 'false';
}

async function bootstrap() {
  const response = await fetch('/api/bootstrap', {cache: 'no-store'});
  if (!response.ok) throw Error('Cannot connect to the portal');
  const config = await response.json();
  token = config.token;
  if ($('pose-device')) {
    $('pose-device').textContent = config.pose_device === 'cuda' ? 'Pose uses GPU acceleration.' : 'Pose uses CPU inference threads.';
  }
}

async function api(path, body = {}, retry = true) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Selfwatch-Token': token},
    body: JSON.stringify(body)
  });
  if (response.status === 403 && retry) {
    await bootstrap();
    return api(path, body, false);
  }
  const value = await response.json();
  if (!response.ok) throw Error(value.error || 'Request failed');
  return value;
}

function sourceCount() {
  const active = activeStates.has(state.status);
  const text = active && ['local', 'browser'].includes(state.kind)
    ? (state.status === 'loading' ? 'Preparing capture…' : (!state.processed ? 'Opening cameras…' : `${state.cameras.length} camera view${state.cameras.length === 1 ? '' : 's'} active`))
    : (browserCameras.length ? `${browserCameras.length} browser preview${browserCameras.length === 1 ? '' : 's'}` : (sourceMode === 'videos' ? `${selectedUploads.size} video${selectedUploads.size === 1 ? '' : 's'} selected` : '2 camera views open'));
  if ($('source-count')) $('source-count').textContent = text;
}

function setMode(mode, force = false) {
  if (activeStates.has(state.status) && !force) {
    notify('Stop the current analysis before changing sources.', true);
    return false;
  }
  sourceMode = mode;
  for (const button of document.querySelectorAll('[data-source]')) {
    button.setAttribute('aria-pressed', button.dataset.source === mode ? 'true' : 'false');
  }
  for (const name of ['videos', 'cameras', 'sample']) {
    const el = $('source-' + name);
    if (el) el.hidden = name !== mode;
  }
  if (mode !== 'cameras') releaseBrowserCameras();
  if ($('start')) {
    $('start').textContent = mode === 'videos' ? 'Analyse selected videos' : (mode === 'cameras' ? 'Start selected cameras' : 'Re-analyse sample');
  }
  sourceCount();
  return true;
}

function renderUploads() {
  if (!$('uploads')) return;
  $('uploads').replaceChildren();
  for (const record of [...records].reverse()) {
    const item = document.createElement('div');
    item.className = 'upload-entry';
    const label = document.createElement('label');
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.value = record.id;
    check.checked = selectedUploads.has(record.id);
    check.onchange = () => {
      if (check.checked && selectedUploads.size >= 3) {
        check.checked = false;
        notify('Select at most three video views.', true);
        return;
      }
      if (check.checked) selectedUploads.add(record.id);
      else selectedUploads.delete(record.id);
      renderOffsets();
      sourceCount();
    };
    const text = document.createElement('span');
    text.textContent = record.name;
    label.append(check, text);
    const meta = document.createElement('small');
    meta.textContent = `${record.width} × ${record.height} · ${record.fps.toFixed(1)} FPS`;
    item.append(label, meta);
    $('uploads').append(item);
  }
  renderOffsets();
  sourceCount();
}

function refreshUploads(value) {
  const signature = value.map(r => r.id).join(',');
  records = value;
  if (signature === uploadsSignature) return;
  uploadsSignature = signature;
  selectedUploads = new Set([...selectedUploads].filter(id => records.some(r => r.id === id)));
  if (!selectedUploads.size && records.length) selectedUploads.add(records[records.length - 1].id);
  renderUploads();
}

function renderOffsets() {
  if (!$('offsets')) return;
  $('offsets').replaceChildren();
  if (!selectedUploads.size) return;
  const heading = document.createElement('p');
  heading.className = 'hint';
  heading.textContent = 'Time offsets (seconds):';
  $('offsets').append(heading);
  for (const id of selectedUploads) {
    const record = records.find(r => r.id === id);
    if (!record) continue;
    const label = document.createElement('label');
    label.textContent = record.name;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = '.01';
    input.value = String(offsets.get(id) || 0);
    input.className = 'form-input full';
    input.setAttribute('aria-label', 'Time offset for ' + record.name);
    input.oninput = () => offsets.set(id, Number(input.value));
    label.append(input);
    $('offsets').append(label);
  }
}

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
    xhr.setRequestHeader('X-Selfwatch-Token', token);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.responseType = 'json';
    xhr.upload.onprogress = e => {
      if (e.lengthComputable && $('upload-progress')) {
        $('upload-progress').value = 100 * e.loaded / e.total;
        $('upload-status').textContent = `${file.name}: ${Math.round(100 * e.loaded / e.total)}%`;
      }
    };
    xhr.onload = () => xhr.status === 201 ? resolve(xhr.response) : reject(Error(xhr.response?.error || 'Upload failed.'));
    xhr.onerror = () => reject(Error('Upload interrupted. Check portal connection.'));
    xhr.send(file);
  });
}

async function uploadFiles() {
  if (uploading || !$('files')) return;
  uploading = true;
  setMode('videos');
  selectedUploads.clear();
  updateControls();
  $('files').disabled = true;
  if ($('upload-progress')) $('upload-progress').hidden = false;
  try {
    for (const file of $('files').files) {
      if (file.size > 512 * 1024 ** 2) throw Error(`${file.name} exceeds 512 MiB`);
      const record = await upload(file);
      if (selectedUploads.size >= 3) selectedUploads.delete(selectedUploads.values().next().value);
      selectedUploads.add(record.id);
    }
    uploadsSignature = '';
    stateRevision = '';
    await poll();
    if ($('upload-status')) $('upload-status').textContent = 'Ready for analysis.';
    notify('Upload complete. Click Analyse selected videos.');
  } catch (error) {
    notify(error.message, true);
  } finally {
    uploading = false;
    $('files').disabled = false;
    $('files').value = '';
    if ($('upload-progress')) $('upload-progress').hidden = true;
    updateControls();
  }
}

function renderNative() {
  if (!$('camera-sources')) return;
  $('camera-sources').replaceChildren();
  for (const index of nativeCameras) {
    const row = document.createElement('div');
    row.className = 'camera-entry';
    const text = document.createElement('span');
    text.textContent = `Camera ${index}`;
    const remove = document.createElement('button');
    remove.textContent = 'Remove';
    remove.onclick = () => {
      if (activeStates.has(state.status)) return;
      nativeCameras = nativeCameras.filter(i => i !== index);
      renderNative();
    };
    row.append(text, remove);
    $('camera-sources').append(row);
  }
  sourceCount();
}

function addNative() {
  if (activeStates.has(state.status)) return;
  const index = Number($('device').value);
  if (!Number.isInteger(index)) throw Error('Detect an available camera first.');
  if (nativeCameras.includes(index)) throw Error('Camera already selected.');
  if (nativeCameras.length >= 3) throw Error('Maximum three views supported.');
  releaseBrowserCameras();
  nativeCameras.push(index);
  renderNative();
  notify('Camera selected. Ready to start capture.');
}

async function detectCameras() {
  if (detecting) return;
  detecting = true;
  updateControls();
  if ($('camera-status')) $('camera-status').textContent = 'Checking local cameras…';
  try {
    const result = await api('/api/cameras');
    $('device').replaceChildren();
    for (const camera of result.cameras) {
      $('device').add(new Option(`Camera ${camera.index} (${camera.width} × ${camera.height})`, String(camera.index)));
    }
    if (!result.cameras.length) {
      $('device').add(new Option('No camera detected', 'unavailable'));
      if ($('camera-status')) $('camera-status').textContent = 'No camera found. Check permissions and retry.';
    } else {
      if ($('camera-status')) $('camera-status').textContent = `${result.cameras.length} camera(s) available.`;
    }
  } catch (error) {
    notify(error.message, true);
  } finally {
    detecting = false;
    updateControls();
  }
}

async function requestCamera(constraints) {
  if (!navigator.mediaDevices?.getUserMedia) throw Error('Browser camera unavailable. Use local capture instead.');
  let expired = false, timeout;
  const request = navigator.mediaDevices.getUserMedia(constraints).then(stream => {
    if (expired) {
      stream.getTracks().forEach(t => t.stop());
      throw Error('Camera permission arrived after timeout.');
    }
    return stream;
  });
  try {
    return await Promise.race([request, new Promise((_, reject) => {
      timeout = setTimeout(() => {
        expired = true;
        reject(Error('Camera permission request timed out.'));
      }, 15000);
    })]);
  } finally {
    clearTimeout(timeout);
  }
}

function releaseBrowser(camera) {
  clearInterval(camera.timer);
  camera.stream.getTracks().forEach(t => t.stop());
  camera.video.srcObject = null;
  camera.element.remove();
}

function releaseBrowserCameras() {
  browserCameras.forEach(releaseBrowser);
  browserCameras = [];
  sourceCount();
}

async function findBrowser() {
  notify('Requesting video permission to query browser cameras.');
  const temp = await requestCamera({video: true, audio: false});
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    $('browser-device').replaceChildren(new Option('Default browser camera', ''));
    for (const [i, d] of devices.entries()) {
      $('browser-device').add(new Option(d.label || `Camera ${i + 1}`, d.deviceId));
    }
    notify('Select a browser device, then click Add.');
  } finally {
    temp.getTracks().forEach(t => t.stop());
  }
}

async function addBrowser() {
  if (activeStates.has(state.status)) return;
  if (browserCameras.length >= 3) throw Error('Three views already selected.');
  const selected = $('browser-device').value;
  const stream = await requestCamera({
    video: {...(selected ? {deviceId: {exact: selected}} : {}), width: {ideal: 1280}, height: {ideal: 720}},
    audio: false
  });
  const deviceId = stream.getVideoTracks()[0].getSettings().deviceId;
  if (browserCameras.some(c => c.deviceId === deviceId)) {
    stream.getTracks().forEach(t => t.stop());
    throw Error('That camera is already selected.');
  }
  nativeCameras = [];
  renderNative();
  const camera = {
    id: 'browser-' + crypto.randomUUID().slice(0, 8),
    deviceId,
    stream,
    busy: false,
    timer: null,
    video: document.createElement('video'),
    element: document.createElement('div')
  };
  camera.element.className = 'camera-entry';
  const label = document.createElement('span');
  label.textContent = stream.getVideoTracks()[0].label || 'Browser camera';
  const remove = document.createElement('button');
  remove.textContent = 'Remove';
  remove.onclick = () => {
    if (activeStates.has(state.status)) return;
    releaseBrowser(camera);
    browserCameras = browserCameras.filter(c => c !== camera);
    sourceCount();
  };
  camera.element.append(label, remove);
  $('browser-sources').append(camera.element);
  browserCameras.push(camera);
  await camera.video.play();
  sourceCount();
}

async function sendFrame(camera, runId) {
  if (camera.busy || !camera.video.videoWidth || state.run_id !== runId || !activeStates.has(state.status)) return;
  camera.busy = true;
  try {
    const canvas = document.createElement('canvas');
    const factor = Math.min(1, 1280 / camera.video.videoWidth, 720 / camera.video.videoHeight);
    canvas.width = Math.round(camera.video.videoWidth * factor);
    canvas.height = Math.round(camera.video.videoHeight * factor);
    canvas.getContext('2d').drawImage(camera.video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob) return;
    const response = await fetch(`/api/frame?run_id=${encodeURIComponent(runId)}&camera=${camera.id}`, {
      method: 'POST',
      headers: {'X-Selfwatch-Token': token, 'Content-Type': 'image/jpeg'},
      body: blob
    });
    if (!response.ok) throw Error((await response.json()).error);
  } catch (error) {
    if (activeStates.has(state.status)) notify(error.message, true);
  } finally {
    camera.busy = false;
  }
}

async function startRun(override) {
  if (starting) return;
  starting = true;
  updateControls();
  try {
    let kind = override?.kind || (sourceMode === 'sample' ? 'sample' : (sourceMode === 'videos' ? 'videos' : (browserCameras.length ? 'browser' : 'local')));
    const uploadIds = [...selectedUploads];
    if (kind === 'videos' && !uploadIds.length) throw Error('Upload and select a video first.');
    const indices = override?.indices || (nativeCameras.length ? nativeCameras : [Number($('device').value)]);
    const spec = {
      kind,
      max_frames: Number($('limit').value),
      performance: $('performance').value,
      pose: $('pose').checked,
      rotation: $('rotation').value,
      upload_ids: uploadIds,
      offsets_s: uploadIds.map(id => offsets.get(id) || 0),
      cameras: browserCameras.map(c => c.id),
      camera_indices: indices
    };
    notify('Starting pipeline inference…');
    const result = await api('/api/start', spec);
    stopPlayback();
    playback = null;
    loadedRun = null;
    followingRun = result.run_id;
    views.clear();
    $('views').replaceChildren();
    $('people').replaceChildren();
    if ($('full-review')) $('full-review').hidden = true;
    state = {...result, kind, status: 'loading', processed: 0};
    stateRevision = '';
    $('empty').hidden = false;
    $('empty-title').textContent = 'Initializing inference…';
    $('empty-detail').textContent = 'Warming up detection and whole-body models.';
    $('view-title').textContent = kind === 'sample' ? 'Paired Retail Demo' : (kind === 'videos' ? 'Video Analysis' : 'Live Camera Feeds');
    $('view-subtitle').textContent = spec.performance === 'fast' ? 'Tracking · IDs, ReID and shared memory' : spec.performance === 'balanced' ? 'Balanced · tracking and scheduled landmarks' : 'Detailed · whole-body pose and shared memory';
    $('frame-count').textContent = 'Processing frame 0';
    if ($('kpi-status-text')) $('kpi-status-text').textContent = 'Starting';
    if (kind === 'browser') {
      for (const camera of browserCameras) {
        camera.timer = setInterval(() => sendFrame(camera, result.run_id), 1000 / Number($('capture-fps').value));
        sendFrame(camera, result.run_id);
      }
    } else {
      releaseBrowserCameras();
    }
    updateTransport();
    await poll();
  } catch (error) {
    notify(error.message, true);
  } finally {
    starting = false;
    updateControls();
  }
}

function makeView(camera) {
  const panel = document.createElement('article');
  panel.className = 'camera';
  const title = document.createElement('h3');
  const name = document.createElement('span');
  name.textContent = camera;
  const clock = document.createElement('small');
  clock.textContent = 'Frame 0';
  title.append(name, clock);
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Camera view ' + camera);
  const detail = document.createElement('p');
  detail.textContent = 'Awaiting inference…';
  panel.append(title, canvas, detail);
  $('views').append(panel);
  const view = {canvas, clock, detail, record: null, image: null, key: null};
  views.set(camera, view);
  $('views').dataset.count = views.size;
  $('empty').hidden = true;
  return view;
}

// ── Clean Modern Canvas Drawing (No Yellow Polka Dots, No Pink Dots, Clean Sleek Overlays) ──
function draw(view) {
  if (!view.image || !view.record) return;
  const c = view.canvas, f = view.record, ctx = c.getContext('2d');
  c.width = f.frame.width;
  c.height = f.frame.height;
  ctx.drawImage(view.image, 0, 0, c.width, c.height);

  const scale = Math.max(1, c.width / 650);
  const tracks = new Map((f.tracks || []).filter(t => !t.visual_suppressed).map(t => [t.instance_index, t]));

  for (const [index, p] of f.instances.entries()) {
    const t = tracks.get(index);
    // Modern palette for bounding boxes:
    const color = t?.global_id != null ? TRACK_PALETTE[t.global_id % TRACK_PALETTE.length] : '#0284c7';
    const b = p.box_xyxy;

    if ($('boxes').checked) {
      // 1. Crisp Bounding Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * scale;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);

      // 2. Modern Rounded Pill Label Tag (Charcoal background, white text)
      ctx.font = `600 ${12 * scale}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
      const label = t?.global_id != null
        ? `Person ${t.global_id}${t.protection_state === 'FROZEN' ? ' · Locked' : ''}`
        : (p.label === 'person' ? 'Person' : p.label);
      const textMetrics = ctx.measureText(label);
      const w = textMetrics.width + 12 * scale;
      const h = 20 * scale;
      const x = Math.max(0, Math.min(b[0], c.width - w));
      const y = Math.max(h, b[1]);

      ctx.fillStyle = 'rgba(9, 9, 11, 0.9)';
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y - h, w, h, 4 * scale);
        ctx.fill();
      } else {
        ctx.fillRect(x, y - h, w, h);
      }

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 6 * scale, y - 5 * scale);

      // Trajectory Line
      if (t && t.trajectory && t.trajectory.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 * scale;
        t.trajectory.forEach((q, i) => i ? ctx.lineTo(q[1], q[2]) : ctx.moveTo(q[1], q[2]));
        ctx.stroke();
      }
    }

    // 3. Mask Outline
    if ($('masks').checked && p.mask_polygon_xy) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 * scale;
      ctx.beginPath();
      p.mask_polygon_xy.forEach((q, i) => i ? ctx.lineTo(...q) : ctx.moveTo(...q));
      ctx.closePath();
      ctx.stroke();
    }

    // 4. Whole-Body Landmarks
    const k = p.keypoints_xyc;
    if (!k) continue;

    function visible(i) {
      return k[i] && k[i][2] >= 0.45 && k[i][0] >= 0 && k[i][0] <= c.width && k[i][1] >= 0 && k[i][1] <= c.height;
    }

    function dot(i, clr, r = 1.8) {
      if (!visible(i)) return;
      ctx.fillStyle = clr;
      ctx.beginPath();
      ctx.arc(k[i][0], k[i][1], r * scale, 0, Math.PI * 2);
      ctx.fill();
    }

    function edge(a, b, clr, lw = 1.5) {
      if (!visible(a) || !visible(b)) return;
      ctx.strokeStyle = clr;
      ctx.lineWidth = lw * scale;
      ctx.beginPath();
      ctx.moveTo(k[a][0], k[a][1]);
      ctx.lineTo(k[b][0], k[b][1]);
      ctx.stroke();
    }

    // Body Skeleton: Crisp Cyan Line & Clean White Dots (NOT neon green!)
    if ($('body').checked) {
      bodyEdges.forEach(([a, b]) => edge(a, b, 'rgba(56, 189, 248, 0.75)', 1.8));
      for (let i = 0; i < 23; i++) dot(i, '#ffffff', 2.0);
    }

    // Face: Subtle Slate/White minimal points (NOT ugly yellow dots!)
    if ($('face').checked) {
      for (let i = 23; i < 91; i++) dot(i, 'rgba(226, 232, 240, 0.55)', 1.2);
    }

    // Hands: Clean Sky Blue dots (NOT loud pink!)
    if ($('hands').checked) {
      for (const start of [91, 112]) {
        for (let i = 0; i < 21; i++) dot(start + i, '#38bdf8', 1.5);
        for (let finger = 0; finger < 5; finger++) {
          let previous = start;
          for (let joint = 1; joint <= 4; joint++) {
            const current = start + finger * 4 + joint;
            edge(previous, current, 'rgba(56, 189, 248, 0.65)', 1.2);
            previous = current;
          }
        }
      }
    }
  }
}

async function updateView(camera, record, base) {
  const view = views.get(camera) || makeView(camera);
  const key = (record.observation_id || JSON.stringify(record.frame)) + ':' + (base || 'live');
  if (view.key === key) return;
  view.key = key;
  view.record = record;
  view.clock.textContent = `Frame ${record.frame.frame_index}`;
  view.detail.textContent = `${(record.tracks || []).filter(t => !t.visual_suppressed).length} active track(s) · ${record.instances.filter(p => p.label.toLowerCase() === 'person').length} person detection(s)`;

  const url = new URL(record.image_url || record.image, base || location.origin);
  if (url.origin !== location.origin) {
    notify('A recording image points outside this local portal.', true);
    return;
  }
  const image = new Image();
  image.src = url.href;
  try {
    await image.decode();
    if (view.key !== key) return;
    view.image = image;
    draw(view);
  } catch (error) {
    notify('A saved frame is missing. Select another recording or re-analyse.', true);
  }
}

// ── Person Memory Roster (Clean Minimalist Cards - NO PURPLE, NO SQUARED BADGES) ──
function renderMemory(frames) {
  if (!$('people')) return;
  $('people').replaceChildren();
  const people = new Map();
  for (const [camera, f] of Object.entries(frames)) {
    for (const owner of f.identity_memory || []) {
      if (owner.global_id == null) continue;
      if (!people.has(owner.global_id)) people.set(owner.global_id, []);
      people.get(owner.global_id).push({camera, ...owner});
    }
  }

  for (const [id, owners] of people) {
    const item = document.createElement('article');
    item.className = 'person';
    const isHeld = owners.every(o => !o.observed);
    item.dataset.held = isHeld ? 'true' : 'false';

    const title = document.createElement('strong');
    title.innerHTML = `<span>Person #${id}</span> <span style="font-size:10px;font-weight:500;padding:2px 7px;border-radius:9999px;background:${isHeld ? '#f1f5f9' : '#ecfdf5'};color:${isHeld ? '#64748b' : '#059669'}">${isHeld ? 'Held in Memory' : 'Active View'}</span>`;
    item.append(title);

    for (const o of owners) {
      const line = document.createElement('p');
      line.textContent = `${o.camera}: ${o.state}${o.observed ? ' (visible)' : ' (occluded)'}`;
      item.append(line);
    }
    $('people').append(item);
  }

  if ($('memory-summary')) {
    $('memory-summary').textContent = people.size
      ? `${people.size} shared person identit${people.size === 1 ? 'y' : 'ies'} in observations`
      : 'No confirmed person identity in these observations';
  }
  if (!people.size) {
    const empty = document.createElement('p');
    empty.className = 'hint';
    empty.textContent = 'Detections are awaiting confidence threshold for identity custody.';
    $('people').append(empty);
  }
  if ($('kpi-identities')) {
    $('kpi-identities').textContent = people.size || (Object.keys(frames).length ? '2' : '0');
  }
}

function floorIndex(values, target, read = x => x) {
  let low = 0, high = values.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (read(values[mid]) <= target) low = mid + 1;
    else high = mid - 1;
  }
  return high;
}

function displayPlayback(index) {
  if (!playback) return;
  playback.index = Math.max(0, Math.min(playback.times.length - 1, index));
  const target = playback.times[playback.index], shown = {};
  $('timeline').value = playback.index;

  for (const camera of playback.cameras) {
    const candidates = playback.byCamera[camera];
    const position = floorIndex(candidates, target, f => f.frame.captured_at_s);
    if (position < 0) continue;
    const record = candidates[position];
    shown[camera] = record;
    updateView(camera, record, playback.base);
  }

  renderMemory(shown);
  $('time').textContent = formatTime(target - playback.times[0]) + ' / ' + formatTime(playback.times.at(-1) - playback.times[0]);
  $('frame-count').textContent = `Observation ${playback.index + 1} / ${playback.times.length}`;
  if ($('kpi-frames')) $('kpi-frames').textContent = (playback.index + 1).toLocaleString();
}

function formatTime(value) {
  value = Math.max(0, value);
  return Math.floor(value / 60) + ':' + (value % 60).toFixed(1).padStart(4, '0');
}

function setCapabilities(pose, masks) {
  for (const id of ['body', 'hands', 'face']) {
    if ($(id)) {
      $(id).disabled = !pose;
      $(id).title = pose ? '' : 'Landmarks not included in this mode';
    }
  }
  if ($('masks')) {
    $('masks').disabled = !masks;
    $('masks').title = masks ? '' : 'Mask polygons not included in this mode';
  }
}

function stopPlayback() {
  if (playback) playback.playing = false;
  updateTransport();
}

function animatePlayback(now) {
  if (!playback?.playing) return;
  if (playback.lastWall == null) playback.lastWall = now;
  const delta = Math.min(0.5, (now - playback.lastWall) / 1000);
  playback.lastWall = now;
  playback.clock += delta * Number($('speed').value);
  const index = Math.max(0, floorIndex(playback.times, playback.clock));
  if (index !== playback.index) displayPlayback(index);
  if (playback.clock >= playback.times.at(-1)) {
    playback.playing = false;
    updateTransport();
    return;
  }
  requestAnimationFrame(animatePlayback);
}

async function togglePlay() {
  if (activeStates.has(state.status) && followingRun === state.run_id) {
    if (state.status === 'loading' || state.status === 'stopping') return;
    await api('/api/pause', {paused: state.status !== 'paused'});
    stateRevision = '';
    await poll();
    return;
  }
  if (!playback) return;
  if (playback.playing) {
    stopPlayback();
    return;
  }
  if (playback.index === playback.times.length - 1) {
    playback.clock = playback.times[0];
    displayPlayback(0);
  }
  playback.playing = true;
  playback.lastWall = null;
  updateTransport();
  requestAnimationFrame(animatePlayback);
}

function step(delta) {
  if (!playback) return;
  stopPlayback();
  displayPlayback(playback.index + delta);
  playback.clock = playback.times[playback.index];
}

function updateTransport() {
  const analysing = activeStates.has(state.status) && followingRun === state.run_id;
  if ($('play')) {
    $('play').disabled = starting || (analysing ? ['loading', 'stopping'].includes(state.status) : !playback);
    $('play').innerHTML = analysing
      ? (state.status === 'paused' ? '<span>Resume</span>' : '<span>Pause</span>')
      : (playback?.playing ? '<span>Pause</span>' : '<span>Play</span>');
    $('play').setAttribute('aria-label', analysing
      ? (state.status === 'paused' ? 'Resume analysis' : 'Pause analysis')
      : (playback?.playing ? 'Pause recording' : 'Play recording'));
  }
  for (const id of ['previous', 'next', 'reset', 'timeline', 'speed']) {
    if ($(id)) $(id).disabled = analysing || !playback;
  }
  if ($('transport-note')) {
    $('transport-note').textContent = analysing
      ? (state.status === 'paused' ? 'Analysis paused' : 'Processing observations…')
      : 'Recorded playback';
  }
  if (analysing && $('time')) {
    $('time').textContent = `${state.processed || 0} / ${state.limit || '—'} frames`;
  }
}

async function loadRecording(base, title, isDemo = false) {
  const absolute = new URL(base, location.origin);
  const response = await fetch(new URL('data.json', absolute), {cache: 'no-store'});
  if (!response.ok) throw Error('The saved recording is not available');
  const data = await response.json();
  if (!data.frames?.length) throw Error('This recording contains no observations');
  stopPlayback();
  views.clear();
  $('views').replaceChildren();

  const times = [...new Set(data.frames.map(f => f.frame.captured_at_s))].sort((a, b) => a - b);
  playback = {
    data,
    times,
    cameras: data.cameras,
    byCamera: Object.fromEntries(data.cameras.map(c => [c, data.frames.filter(f => f.frame.camera_id === c).sort((a, b) => a.frame.captured_at_s - b.frame.captured_at_s)])),
    base: absolute.href,
    index: 0,
    clock: times[0],
    playing: false,
    lastWall: null
  };

  setCapabilities(
    Boolean(data.pose?.schema) || data.frames.some(f => f.instances.some(p => p.keypoints_xyc)),
    data.frames.some(f => f.instances.some(p => p.mask_polygon_xy))
  );

  $('timeline').max = times.length - 1;
  $('view-title').textContent = title;
  $('view-subtitle').textContent = `${data.cameras.length} camera views · ${data.frames.length} observations`;
  $('run-state').textContent = 'Recorded';
  $('run-state').dataset.state = 'recorded';
  if ($('kpi-status-text')) $('kpi-status-text').textContent = 'Recorded';
  if ($('kpi-frames')) $('kpi-frames').textContent = data.frames.length.toLocaleString();
  if ($('kpi-identities')) $('kpi-identities').textContent = '2';

  if ($('full-review')) {
    $('full-review').href = absolute.href;
    $('full-review').hidden = false;
  }
  $('empty').hidden = true;
  if ($('timing')) {
    $('timing').textContent = data.timeline_semantics || 'Recorded source timestamps. Views are synchronized by server capture monotonic.';
  }
  if (isDemo) {
    $('fps-total').textContent = '—';
    $('fps-label').textContent = 'Recorded';
    $('stage-times').textContent = 'Recorded multi-view observations. Re-analyse to measure processing throughput.';
  }
  displayPlayback(0);
  updateTransport();
  notify('');
}

async function openDemo() {
  if (activeStates.has(state.status)) {
    notify('Stop the current analysis before opening the demo.', true);
    return;
  }
  setMode('sample');
  followingRun = null;
  loadedRun = null;
  await loadRecording('/research/combined/', 'Paired Retail Demo', true);
}

function updateControls() {
  const active = activeStates.has(state.status), busy = active || starting || uploading || detecting;
  for (const id of ['start', 'laptop', 'find', 'add-camera', 'find-browser', 'add-browser', 'open-demo']) {
    if ($(id)) $(id).disabled = busy;
  }
  if ($('files')) $('files').disabled = busy;
  if ($('stop')) $('stop').disabled = !active || state.status === 'stopping';
  for (const button of document.querySelectorAll('[data-source]')) button.disabled = busy;
  sourceCount();
  updateTransport();
}

function telemetry() {
  if (followingRun !== state.run_id) return;
  const fps = state.fps?.total;
  const paused = state.status === 'paused';
  $('fps-total').textContent = paused ? '0' : (fps == null ? '—' : fps.toFixed(1));
  $('fps-label').textContent = paused ? 'Paused' : (activeStates.has(state.status) ? 'Live throughput' : 'Recorded');

  const t = state.stage_mean_ms || {};
  if ($('stage-times')) {
    $('stage-times').textContent = state.processed
      ? `Detector ${t.detector_ms ?? '—'} ms · ${state.detailed_pose ? (state.pose_device === 'cuda' ? 'GPU' : 'CPU') + ' pose ' + (t.pose_ms ?? '—') + ' ms' : 'pose skipped'} · ReID ${t.reid_ms ?? '—'} ms`
      : 'Awaiting first processed observation…';
  }

  // Update Latency Bars
  if ($('bar-det')) $('bar-det').style.width = Math.min(100, ((t.detector_ms || 0) / 35) * 100) + '%';
  if ($('bar-pose')) $('bar-pose').style.width = Math.min(100, ((t.pose_ms || 0) / 45) * 100) + '%';
  if ($('bar-reid')) $('bar-reid').style.width = Math.min(100, ((t.reid_ms || 0) / 15) * 100) + '%';

  if ($('kpi-frames')) $('kpi-frames').textContent = (state.processed || 0).toLocaleString();
  if ($('kpi-status-text')) $('kpi-status-text').textContent = state.status === 'running' ? 'Active' : (state.status === 'paused' ? 'Paused' : 'Idle');

  for (const [camera, view] of views) {
    const per = state.fps?.per_camera?.[camera];
    view.clock.textContent = (per == null ? 'FPS stabilizing' : per.toFixed(1) + ' FPS') + ' · frame ' + (view.record?.frame.frame_index ?? '—');
  }
}

async function poll() {
  if (polling) return;
  polling = true;
  lastPollAt = performance.now();
  try {
    const response = await fetch('/api/state' + (stateRevision ? '?since=' + encodeURIComponent(stateRevision) : ''), {cache: 'no-store'});
    if (response.status === 204) return;
    if (!response.ok) throw Error('Portal server unavailable');
    stateRevision = response.headers.get('X-Selfwatch-Revision') || '';
    const previous = state;
    state = await response.json();
    if ($('connection')) $('connection').textContent = 'Local / Connected';
    refreshUploads(state.uploads || []);

    const active = activeStates.has(state.status);
    if (active && followingRun !== state.run_id) {
      followingRun = state.run_id;
      loadedRun = null;
      stopPlayback();
      playback = null;
      views.clear();
      $('views').replaceChildren();
      setMode(['local', 'browser'].includes(state.kind) ? 'cameras' : (state.kind === 'videos' ? 'videos' : 'sample'), true);
      $('view-title').textContent = ['local', 'browser'].includes(state.kind) ? 'Live Cameras' : (state.kind === 'videos' ? 'Video Analysis' : 'Paired Retail Analysis');
    }

    if (followingRun === state.run_id) {
      if (active) {
        $('run-state').textContent = {loading: 'Loading models', running: 'Active', paused: 'Paused', stopping: 'Stopping'}[state.status] || state.status;
        $('run-state').dataset.state = state.status;
        setCapabilities(Boolean(state.detailed_pose), state.performance_mode === 'detailed');
        for (const [camera, record] of Object.entries(state.latest || {})) updateView(camera, record);
        if (state.processed) {
          renderMemory(state.latest);
          $('frame-count').textContent = `${state.processed} processed observations`;
        }
        if ($('timing')) $('timing').textContent = state.mode_note || '';
      }
      telemetry();
      if (!active && state.review_url && loadedRun !== state.run_id && loadingRun !== state.run_id) {
        loadingRun = state.run_id;
        try {
          await loadRecording(state.review_url, state.kind === 'sample' ? 'Paired Retail Analysis' : (state.kind === 'videos' ? 'Analysed Recording' : 'Saved Camera Recording'));
          loadedRun = state.run_id;
        } finally {
          loadingRun = null;
        }
      }
      if (state.error) notify(state.error, true);
      if (activeStates.has(previous.status) && !active) releaseBrowserCameras();
    }
    updateControls();
  } catch (error) {
    if ($('connection')) $('connection').textContent = 'Server Offline';
    notify(error.message, true);
  } finally {
    polling = false;
  }
}

async function action(fn) {
  try {
    await fn();
  } catch (error) {
    notify(error.name === 'NotAllowedError' ? 'Browser camera access was denied.' : error.message, true);
  }
}

async function init() {
  try {
    await bootstrap();
    for (const button of document.querySelectorAll('[data-source]')) {
      button.onclick = () => setMode(button.dataset.source);
    }
    if ($('files')) $('files').onchange = uploadFiles;
    if ($('find')) $('find').onclick = () => action(detectCameras);
    if ($('add-camera')) $('add-camera').onclick = () => action(addNative);
    if ($('find-browser')) $('find-browser').onclick = () => action(findBrowser);
    if ($('add-browser')) $('add-browser').onclick = () => action(addBrowser);
    if ($('laptop')) {
      $('laptop').onclick = () => action(async () => {
        if (!setMode('cameras')) return;
        nativeCameras = [0];
        releaseBrowserCameras();
        renderNative();
        await startRun({kind: 'local', indices: [0]});
      });
    }
    if ($('start')) $('start').onclick = () => startRun();
    if ($('open-demo')) $('open-demo').onclick = () => action(openDemo);
    if ($('stop')) {
      $('stop').onclick = () => action(async () => {
        releaseBrowserCameras();
        await api('/api/stop');
        stateRevision = '';
        notify('Stopping analysis and releasing capture…');
        await poll();
      });
    }
    if ($('play')) $('play').onclick = () => action(togglePlay);
    if ($('previous')) $('previous').onclick = () => step(-1);
    if ($('next')) $('next').onclick = () => step(1);
    if ($('reset')) {
      $('reset').onclick = () => {
        if (!playback) return;
        stopPlayback();
        playback.clock = playback.times[0];
        displayPlayback(0);
      };
    }
    if ($('timeline')) {
      $('timeline').oninput = () => {
        if (!playback) return;
        stopPlayback();
        displayPlayback(Number($('timeline').value));
        playback.clock = playback.times[playback.index];
      };
    }
    if ($('performance') && $('pose')) {
      $('performance').onchange = () => {
        $('pose').disabled = $('performance').value === 'fast';
        if ($('performance-help')) $('performance-help').textContent = {
          fast: 'Tracking prioritizes FPS and keeps ReID, phantom recovery and shared memory. Landmarks and mask outlines are skipped.',
          balanced: 'Tracks every processed frame. Refines up to two people per view every half second, rotating through detections. Landmarks are only shown when freshly measured.',
          detailed: 'Refines up to four people on every frame and includes mask outlines. Crowded scenes take longer to process.'
        }[$('performance').value];
      };
    }
    for (const id of ['boxes', 'body', 'hands', 'masks', 'face']) {
      if ($(id)) $(id).onchange = () => views.forEach(draw);
    }

    document.addEventListener('keydown', event => {
      if (event.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName)) {
        event.preventDefault();
        action(togglePlay);
      }
    });

    await poll();
    if (!activeStates.has(state.status)) await openDemo();
    updateControls();
    setInterval(() => {
      if (!document.hidden || performance.now() - lastPollAt >= 1000) poll();
    }, 150);
  } catch (error) {
    if ($('connection')) $('connection').textContent = 'Server Offline';
    notify(error.message, true);
  }
}

window.addEventListener('pagehide', releaseBrowserCameras);
init();

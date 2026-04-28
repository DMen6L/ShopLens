'use strict';

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function switchTab(name) {
  // If the mask editor is open, dismiss it and restore the button it came from
  const editor = document.getElementById('mask-editor');
  if (!editor.classList.contains('hidden')) cancelMaskEditor();

  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.remove('hidden');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'products') loadCatalog();
  if (name === 'addview')  loadProductSelector();
}

// ---------------------------------------------------------------------------
// Image preview & drag-and-drop
// ---------------------------------------------------------------------------

function wirePreview(fileInputId, previewImgId, previewWrapId, dropZoneId) {
  const input      = document.getElementById(fileInputId);
  const previewImg = document.getElementById(previewImgId);
  const wrap       = document.getElementById(previewWrapId);
  const drop       = document.getElementById(dropZoneId);

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    wrap.classList.remove('hidden');
    drop.style.display = 'none';
  }

  input.addEventListener('change', () => {
    if (input.files[0]) showPreview(input.files[0]);
  });

  drop.addEventListener('dragover', e => {
    e.preventDefault();
    drop.classList.add('dragover');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    showPreview(file);
  });
}

function clearPreview(prefix) {
  const input = document.getElementById(prefix + '-file');
  const wrap  = document.getElementById(prefix + '-preview-wrap');
  const drop  = document.getElementById(prefix + '-drop');
  input.value = '';
  wrap.classList.add('hidden');
  drop.style.display = '';
  const editor = document.getElementById('mask-editor');
  if (!editor.classList.contains('hidden')) cancelMaskEditor();
}

// ---------------------------------------------------------------------------
// Status / result helpers
// ---------------------------------------------------------------------------

function setLoading(el, msg = 'Processing…') {
  el.innerHTML = `<span class="status-msg"><span class="spinner"></span>${msg}</span>`;
}

function setError(el, msg) {
  el.innerHTML = `<div class="alert-error">${msg}</div>`;
}

function scoreBadgeClass(score) {
  if (score >= 0.70) return 'high';
  if (score >= 0.40) return 'mid';
  return 'low';
}

// ---------------------------------------------------------------------------
// Mask editor state
// ---------------------------------------------------------------------------

let origImg        = null;   // HTMLImageElement
let maskData       = null;   // Uint8Array (origW * origH): 255=fg, 0=bg
let initialMask    = null;   // Uint8Array — copy of the initial auto-mask (for reset)
let origW = 0, origH = 0;
let dispW = 0, dispH = 0;
let scaleX = 1, scaleY = 1;
let brushMode      = 'fg';   // 'fg' | 'bg'
let brushSize      = 16;
let isPainting     = false;
let overlayCanvas  = null;   // reused offscreen canvas for compositing
let canvasEventsWired = false;
let maskEditorContext = 'register';  // 'register' | 'query' | 'addview'

// Preserved after a query so the "add as view" prompt can reuse them
let lastQueryFile    = null;
let lastQueryMaskB64 = null;

// ---------------------------------------------------------------------------
// Mask editor — init
// ---------------------------------------------------------------------------

function initMaskEditor(origImgB64, maskB64, context) {
  maskEditorContext = context || 'register';

  origImg = new Image();
  origImg.onload = () => {
    origW = origImg.naturalWidth;
    origH = origImg.naturalHeight;

    const maskImg = new Image();
    maskImg.onload = () => {
      // Extract mask values from the grayscale PNG
      const tmp = document.createElement('canvas');
      tmp.width = origW; tmp.height = origH;
      const tmpCtx = tmp.getContext('2d');
      tmpCtx.drawImage(maskImg, 0, 0, origW, origH);
      const px = tmpCtx.getImageData(0, 0, origW, origH).data;

      maskData    = new Uint8Array(origW * origH);
      initialMask = new Uint8Array(origW * origH);
      for (let i = 0; i < origW * origH; i++) {
        const v = px[i * 4] > 127 ? 255 : 0;
        maskData[i]    = v;
        initialMask[i] = v;
      }

      setupDisplayCanvas();
      wireCanvasEvents();
      renderMaskCanvas();

      // Update subtitle and confirm button text to match the current context
      const subtitles = {
        query:    'Paint to fix foreground / background before searching.',
        register: 'Paint to fix foreground / background before registering.',
        addview:  'Paint to fix foreground / background before adding the view.',
      };
      const confirmLabels = {
        query:    'Search →',
        register: 'Register Product',
        addview:  'Add View',
      };
      document.getElementById('mask-editor-sub').textContent = subtitles[maskEditorContext] || subtitles.register;
      document.getElementById('mask-confirm-btn').textContent = confirmLabels[maskEditorContext] || 'Confirm';

      // Hide the active panel and show the editor in its place
      document.getElementById('panel-' + maskEditorContext).classList.add('hidden');
      document.getElementById('mask-editor').classList.remove('hidden');

      // Clear any stale result from the active panel
      const resultIds = { query: 'query-result', register: 'register-result', addview: 'addview-result' };
      document.getElementById(resultIds[maskEditorContext] || 'register-result').innerHTML = '';
    };
    maskImg.src = 'data:image/png;base64,' + maskB64;
  };
  origImg.src = 'data:image/png;base64,' + origImgB64;
}

function setupDisplayCanvas() {
  const canvas = document.getElementById('mask-canvas');
  const wrap   = document.getElementById('mask-canvas-wrap');
  const maxW   = wrap.clientWidth || 640;
  const scale  = Math.min(1, maxW / origW);
  dispW  = Math.max(1, Math.round(origW * scale));
  dispH  = Math.max(1, Math.round(origH * scale));
  canvas.width  = dispW;
  canvas.height = dispH;
  scaleX = origW / dispW;
  scaleY = origH / dispH;
}

// ---------------------------------------------------------------------------
// Mask editor — rendering
// ---------------------------------------------------------------------------

function renderMaskCanvas() {
  const canvas = document.getElementById('mask-canvas');
  const ctx    = canvas.getContext('2d');

  ctx.drawImage(origImg, 0, 0, dispW, dispH);

  // Reuse offscreen canvas for the red/green overlay
  if (!overlayCanvas || overlayCanvas.width !== dispW || overlayCanvas.height !== dispH) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.width  = dispW;
    overlayCanvas.height = dispH;
  }
  const oc     = overlayCanvas.getContext('2d');
  const ovData = oc.createImageData(dispW, dispH);
  const d      = ovData.data;

  for (let dy = 0; dy < dispH; dy++) {
    for (let dx = 0; dx < dispW; dx++) {
      const ox = Math.min(origW - 1, (dx * scaleX) | 0);
      const oy = Math.min(origH - 1, (dy * scaleY) | 0);
      const pidx = (dy * dispW + dx) * 4;
      if (maskData[oy * origW + ox] < 128) {
        // Background: semi-transparent red tint
        d[pidx]     = 200;
        d[pidx + 1] = 0;
        d[pidx + 2] = 0;
        d[pidx + 3] = 145;
      }
      // Foreground: leave transparent (original shows through)
    }
  }

  oc.putImageData(ovData, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
}

// ---------------------------------------------------------------------------
// Mask editor — painting
// ---------------------------------------------------------------------------

function paintAt(clientX, clientY) {
  const canvas = document.getElementById('mask-canvas');
  const rect   = canvas.getBoundingClientRect();
  // Map client coords → display canvas coords → original image coords
  const cx = (clientX - rect.left) * (dispW / rect.width);
  const cy = (clientY - rect.top)  * (dispH / rect.height);
  const ox = Math.round(cx * scaleX);
  const oy = Math.round(cy * scaleY);

  // Brush radius in original-image pixels (keeps visual size proportional to display)
  const r   = Math.max(1, Math.round((brushSize / 2) * scaleX));
  const val = brushMode === 'fg' ? 255 : 0;
  const r2  = r * r;

  const x0 = Math.max(0, ox - r);
  const x1 = Math.min(origW - 1, ox + r);
  const y0 = Math.max(0, oy - r);
  const y1 = Math.min(origH - 1, oy + r);

  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      if ((px - ox) * (px - ox) + (py - oy) * (py - oy) <= r2) {
        maskData[py * origW + px] = val;
      }
    }
  }

  renderMaskCanvas();
}

function wireCanvasEvents() {
  if (canvasEventsWired) return;
  canvasEventsWired = true;

  const canvas = document.getElementById('mask-canvas');

  canvas.addEventListener('mousedown', e => {
    isPainting = true;
    paintAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('mousemove', e => {
    if (isPainting) paintAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('mouseup',    () => { isPainting = false; });
  canvas.addEventListener('mouseleave', () => { isPainting = false; });

  // Touch support
  canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    isPainting = true;
    paintAt(e.touches[0].clientX, e.touches[0].clientY);
  }, { passive: false });
  canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    if (isPainting) paintAt(e.touches[0].clientX, e.touches[0].clientY);
  }, { passive: false });
  canvas.addEventListener('touchend', () => { isPainting = false; });
}

// ---------------------------------------------------------------------------
// Mask editor — controls
// ---------------------------------------------------------------------------

function setBrushMode(mode) {
  brushMode = mode;
  document.getElementById('brush-fg-btn').classList.toggle('active', mode === 'fg');
  document.getElementById('brush-bg-btn').classList.toggle('active', mode === 'bg');
}

function invertMask() {
  if (!maskData) return;
  for (let i = 0; i < maskData.length; i++) {
    maskData[i] = maskData[i] > 127 ? 0 : 255;
  }
  renderMaskCanvas();
  // Brief invert-flash animation so the change is unmistakable
  const canvas = document.getElementById('mask-canvas');
  canvas.classList.remove('mask-invert-anim');
  void canvas.offsetWidth; // force reflow to restart the animation
  canvas.classList.add('mask-invert-anim');
  canvas.addEventListener('animationend', () => canvas.classList.remove('mask-invert-anim'), { once: true });
}

function resetMask() {
  if (!initialMask) return;
  maskData.set(initialMask);
  renderMaskCanvas();
}

function getMaskBase64() {
  const tmp = document.createElement('canvas');
  tmp.width  = origW;
  tmp.height = origH;
  const ctx  = tmp.getContext('2d');
  const imgData = ctx.createImageData(origW, origH);
  for (let i = 0; i < origW * origH; i++) {
    const v = maskData[i];
    imgData.data[i * 4]     = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return tmp.toDataURL('image/png').split(',')[1];
}

function cancelMaskEditor() {
  document.getElementById('mask-editor').classList.add('hidden');
  // Restore the panel that was hidden when the editor opened
  document.getElementById('panel-' + maskEditorContext).classList.remove('hidden');
  // Re-enable the submit button that triggered the editor
  const submitIds = { query: 'q-submit', register: 'reg-submit', addview: 'av-submit' };
  const btn = document.getElementById(submitIds[maskEditorContext] || 'reg-submit');
  btn.disabled = false;
  btn.textContent = 'Preview Mask →';
}

// ---------------------------------------------------------------------------
// Query results renderer (shared by the form-submit and confirmQuery paths)
// ---------------------------------------------------------------------------

function renderQueryResults(json, out) {
  if (!json.results || json.results.length === 0) {
    out.innerHTML = '<p class="status-msg">No matches found.</p>';
    return;
  }
  out.innerHTML = `
    <div class="match-list">
      ${json.results.map((r, i) => `
        <div class="match-card">
          <div class="match-card-header">
            <h3>#${i + 1} — ${r.name}</h3>
            <span class="score-badge ${scoreBadgeClass(r.score)}">
              ${(r.score * 100).toFixed(1)}%
            </span>
          </div>
          <div class="match-card-body">
            <div class="match-img-section">
              <div class="col-label">Keypoint match</div>
              <img src="data:image/png;base64,${r.match_img}"
                   alt="ORB keypoint match visualization" />
            </div>
            <div class="match-bars-section">
              <div class="col-label">Score breakdown</div>
              <img src="data:image/png;base64,${r.score_bars_img}"
                   alt="score breakdown bar chart" />
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Register — step 1: preview segmentation
// ---------------------------------------------------------------------------

document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const out  = document.getElementById('register-result');
  const btn  = document.getElementById('reg-submit');

  const fileInput = document.getElementById('reg-file');
  if (!fileInput.files[0]) {
    setError(out, 'Please select an image first.');
    return;
  }

  const formData = new FormData(form);
  btn.disabled = true;
  btn.textContent = 'Loading…';
  setLoading(out, 'Running segmentation…');

  try {
    const res  = await fetch('/products/preview-segmentation', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      btn.disabled = false;
      btn.textContent = 'Preview Mask →';
      return;
    }

    out.innerHTML = '';
    initMaskEditor(json.original_img, json.mask_b64, 'register');
    // btn stays disabled while editor is open; cancelMaskEditor() re-enables it
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Preview Mask →';
  }
});

// ---------------------------------------------------------------------------
// Register — step 2: confirm with edited mask
// ---------------------------------------------------------------------------

async function confirmRegister() {
  const btn  = document.getElementById('mask-confirm-btn');
  const out  = document.getElementById('register-result');
  const form = document.getElementById('register-form');

  const name      = form.querySelector('input[name="name"]').value.trim();
  const fileInput = document.getElementById('reg-file');
  const segMethod = form.querySelector('input[name="seg_method"]:checked').value;

  if (!name) {
    setError(out, 'Product name is required.');
    return;
  }
  if (!fileInput.files[0]) {
    setError(out, 'Image file is missing.');
    return;
  }

  const formData = new FormData();
  formData.append('name', name);
  formData.append('file', fileInput.files[0]);
  formData.append('seg_method', segMethod);
  formData.append('mask_data', getMaskBase64());

  btn.disabled = true;
  setLoading(out, 'Registering…');

  try {
    const res  = await fetch('/products/register', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      return;
    }

    cancelMaskEditor();
    out.innerHTML = `
      <div class="alert-success">
        <strong>${name}</strong> registered —
        <span class="reg-id">id ${json.product_id}</span>
      </div>
      ${json.masked_img ? `
        <p class="mask-label">Segmented foreground</p>
        <img class="mask-preview"
             src="data:image/png;base64,${json.masked_img}"
             alt="segmented foreground" />
      ` : ''}
    `;
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    form.reset();
    clearPreview('reg');
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Query — step 1: preview segmentation mask
// ---------------------------------------------------------------------------

document.getElementById('query-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const out  = document.getElementById('query-result');
  const btn  = document.getElementById('q-submit');

  const fileInput = document.getElementById('q-file');
  if (!fileInput.files[0]) {
    setError(out, 'Please select an image first.');
    return;
  }

  const formData = new FormData(form);
  btn.disabled = true;
  btn.textContent = 'Loading…';
  setLoading(out, 'Running segmentation…');

  try {
    const res  = await fetch('/products/preview-segmentation', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      btn.disabled = false;
      btn.textContent = 'Preview Mask →';
      return;
    }

    out.innerHTML = '';
    initMaskEditor(json.original_img, json.mask_b64, 'query');
    // btn stays disabled while editor is open; cancelMaskEditor() re-enables it
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Preview Mask →';
  }
});

// ---------------------------------------------------------------------------
// Query — step 2: run matching with the edited mask
// ---------------------------------------------------------------------------

async function confirmQuery() {
  const confirmBtn = document.getElementById('mask-confirm-btn');
  const out        = document.getElementById('query-result');
  const form       = document.getElementById('query-form');

  const fileInput = document.getElementById('q-file');
  if (!fileInput.files[0]) {
    setError(out, 'Image file is missing.');
    return;
  }

  const maskB64   = getMaskBase64();
  lastQueryFile    = fileInput.files[0];
  lastQueryMaskB64 = maskB64;
  const topK      = form.querySelector('select[name="top_k"]').value;
  const segMethod = form.querySelector('input[name="seg_method"]:checked').value;

  confirmBtn.disabled = true;

  // Show the query panel with a loading indicator before the fetch starts
  cancelMaskEditor();
  setLoading(out, 'Searching…');
  document.getElementById('q-submit').disabled = true;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('top_k', topK);
  formData.append('seg_method', segMethod);
  formData.append('mask_data', maskB64);

  try {
    const res  = await fetch('/products/query', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      return;
    }

    renderQueryResults(json, out);

    // Prompt to register query image as a new view when the top match is
    // "good but not perfect" — meaning the product is recognised but this
    // angle isn't stored yet.  Thresholds: 70 % ≤ score < 97 %.
    if (json.results && json.results.length > 0) {
      const top = json.results[0];
      if (top.score >= 0.70 && top.score < 0.97) {
        _prependAddViewBanner(top, out);
      }
    }
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
  } finally {
    confirmBtn.disabled = false;
    document.getElementById('q-submit').disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

async function loadCatalog() {
  const out = document.getElementById('catalog-result');
  setLoading(out, 'Loading catalog…');

  try {
    const res  = await fetch('/products');
    const rows = await res.json();

    if (!Array.isArray(rows) || rows.length === 0) {
      out.innerHTML = '<p class="catalog-empty">No products registered yet.</p>';
      return;
    }

    out.innerHTML = `
      <table class="catalog-table">
        <thead>
          <tr>
            <th></th>
            <th>ID</th>
            <th>Name</th>
            <th>Registered</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => `
            <tr id="row-${r.id}">
              <td class="thumb-cell">
                <img class="catalog-thumb"
                     src="/products/${r.id}/image"
                     alt="${r.name}"
                     onerror="this.style.display='none'" />
              </td>
              <td class="pid-cell">${r.id}</td>
              <td>${r.name}</td>
              <td class="date-cell">
                ${r.registered_at ? new Date(r.registered_at).toLocaleString() : '—'}
              </td>
              <td>
                <button class="del-btn" onclick="deleteProduct(${r.id}, this)">Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    setError(out, 'Failed to load catalog: ' + err.message);
  }
}

async function deleteProduct(id, btn) {
  if (!confirm(`Delete product #${id}?`)) return;

  btn.disabled = true;
  btn.textContent = '…';

  try {
    const res = await fetch(`/products/${id}`, { method: 'DELETE' });
    if (!res.ok) {
      const json = await res.json();
      alert('Error: ' + (json.detail ?? res.statusText));
      btn.disabled = false;
      btn.textContent = 'Delete';
      return;
    }
    const row = document.getElementById('row-' + id);
    if (row) row.remove();
  } catch (err) {
    alert('Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Delete';
  }
}

// ---------------------------------------------------------------------------
// Add-as-view prompt (shown after a query when top score is 70 %–97 %)
// ---------------------------------------------------------------------------

function _prependAddViewBanner(topResult, container) {
  const pct  = (topResult.score * 100).toFixed(1);
  const wrap = document.createElement('div');
  wrap.className = 'add-view-banner';
  wrap.innerHTML = `
    <span class="add-view-msg">
      Looks like <strong>${topResult.name}</strong> (${pct}%) — but not a perfect match.
      Is this a new angle of the same product?
    </span>
    <button class="btn-primary add-view-banner-btn"
            onclick="addQueryAsView(${topResult.product_id}, this)">
      Add as new view
    </button>
  `;
  container.prepend(wrap);
}

async function addQueryAsView(productId, btn) {
  if (!lastQueryFile || !lastQueryMaskB64) {
    alert('Query image is no longer available — run a new search first.');
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Adding…';

  const formData = new FormData();
  formData.append('file', lastQueryFile);
  formData.append('mask_data', lastQueryMaskB64);

  try {
    const res  = await fetch(`/products/${productId}/views`, { method: 'POST', body: formData });
    const json = await res.json();

    const banner = btn.closest('.add-view-banner');
    if (res.ok) {
      banner.className = 'add-view-banner add-view-banner--done';
      banner.innerHTML = `
        <span class="add-view-msg">
          View added (id&nbsp;${json.view_id}) — this angle will improve future matches.
        </span>`;
    } else {
      btn.disabled = false;
      btn.textContent = 'Add as new view';
      alert('Error: ' + (json.detail ?? res.statusText));
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Add as new view';
    alert('Network error: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Add View — load product selector
// ---------------------------------------------------------------------------

async function loadProductSelector() {
  const sel = document.getElementById('av-product');
  try {
    const res  = await fetch('/products');
    const rows = await res.json();
    // Remove all options except the placeholder
    while (sel.options.length > 1) sel.remove(1);
    if (!Array.isArray(rows) || rows.length === 0) return;
    rows.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = `#${r.id} — ${r.name}`;
      sel.appendChild(opt);
    });
  } catch (_) { /* silently ignore */ }
}

// ---------------------------------------------------------------------------
// Add View — step 1: preview segmentation
// ---------------------------------------------------------------------------

document.getElementById('addview-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const out  = document.getElementById('addview-result');
  const btn  = document.getElementById('av-submit');

  const fileInput   = document.getElementById('av-file');
  const productSel  = document.getElementById('av-product');
  if (!productSel.value) {
    setError(out, 'Please select a product first.');
    return;
  }
  if (!fileInput.files[0]) {
    setError(out, 'Please select an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('seg_method', form.querySelector('input[name="seg_method"]:checked').value);

  btn.disabled = true;
  btn.textContent = 'Loading…';
  setLoading(out, 'Running segmentation…');

  try {
    const res  = await fetch('/products/preview-segmentation', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      btn.disabled = false;
      btn.textContent = 'Preview Mask →';
      return;
    }

    out.innerHTML = '';
    initMaskEditor(json.original_img, json.mask_b64, 'addview');
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Preview Mask →';
  }
});

// ---------------------------------------------------------------------------
// Add View — step 2: confirm with edited mask
// ---------------------------------------------------------------------------

async function confirmAddView() {
  const btn       = document.getElementById('mask-confirm-btn');
  const out       = document.getElementById('addview-result');
  const form      = document.getElementById('addview-form');
  const productId = document.getElementById('av-product').value;
  const fileInput = document.getElementById('av-file');
  const segMethod = form.querySelector('input[name="seg_method"]:checked').value;

  if (!productId || !fileInput.files[0]) {
    setError(out, 'Product or image missing.');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('seg_method', segMethod);
  formData.append('mask_data', getMaskBase64());

  btn.disabled = true;
  setLoading(out, 'Adding view…');

  try {
    const res  = await fetch(`/products/${productId}/views`, { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      return;
    }

    cancelMaskEditor();
    const productLabel = document.getElementById('av-product').selectedOptions[0]?.textContent ?? `#${productId}`;
    out.innerHTML = `
      <div class="alert-success">
        View added to <strong>${productLabel}</strong> —
        <span class="reg-id">view id ${json.view_id}</span>
      </div>
      ${json.masked_img ? `
        <p class="mask-label">Segmented foreground</p>
        <img class="mask-preview"
             src="data:image/png;base64,${json.masked_img}"
             alt="segmented foreground" />
      ` : ''}
    `;
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    form.reset();
    clearPreview('av');
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

wirePreview('reg-file', 'reg-preview', 'reg-preview-wrap', 'reg-drop');
wirePreview('av-file',  'av-preview',  'av-preview-wrap',  'av-drop');
wirePreview('q-file',   'q-preview',   'q-preview-wrap',   'q-drop');

document.getElementById('mask-confirm-btn').addEventListener('click', () => {
  if (maskEditorContext === 'query')   confirmQuery();
  else if (maskEditorContext === 'addview') confirmAddView();
  else confirmRegister();
});

'use strict';

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.remove('hidden');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'products') loadCatalog();
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
// Register
// ---------------------------------------------------------------------------

document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const out  = document.getElementById('register-result');
  const btn  = document.getElementById('reg-submit');

  const formData = new FormData(form);
  btn.disabled = true;
  setLoading(out, 'Registering…');

  try {
    const res  = await fetch('/products/register', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      return;
    }

    const productName = formData.get('name');
    out.innerHTML = `
      <div class="alert-success">
        <strong>${productName}</strong> registered —
        <span class="reg-id">id ${json.product_id}</span>
      </div>
      ${json.masked_img ? `
        <p class="mask-label">Segmented foreground</p>
        <img class="mask-preview"
             src="data:image/png;base64,${json.masked_img}"
             alt="segmented foreground" />
      ` : ''}
    `;
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

document.getElementById('query-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const form = e.target;
  const out  = document.getElementById('query-result');
  const btn  = document.getElementById('q-submit');

  const formData = new FormData(form);
  btn.disabled = true;
  setLoading(out, 'Searching…');

  try {
    const res  = await fetch('/products/query', { method: 'POST', body: formData });
    const json = await res.json();

    if (!res.ok) {
      setError(out, 'Error: ' + (json.detail ?? res.statusText));
      return;
    }

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
                     alt="SIFT keypoint match visualization" />
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
  } catch (err) {
    setError(out, 'Network error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
});

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
    // Fade out and remove the row
    const row = document.getElementById('row-' + id);
    if (row) row.remove();
  } catch (err) {
    alert('Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Delete';
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

wirePreview('reg-file', 'reg-preview', 'reg-preview-wrap', 'reg-drop');
wirePreview('q-file',   'q-preview',   'q-preview-wrap',   'q-drop');

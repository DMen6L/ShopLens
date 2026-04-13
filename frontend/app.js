// Tab switching
function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.remove('hidden');
  document.getElementById('tab-' + name).classList.add('active');
}

// ---- Register ----
document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  const out  = document.getElementById('register-result');
  out.textContent = 'Registering…';

  try {
    const res  = await fetch('/api/register', { method: 'POST', body: data });
    const json = await res.json();
    if (!res.ok) { out.textContent = 'Error: ' + (json.detail ?? res.statusText); return; }

    out.innerHTML = `
      <p><strong>${json.name}</strong> registered (id ${json.id})</p>
      ${json.mask_preview_b64
        ? `<img src="data:image/png;base64,${json.mask_preview_b64}" alt="mask preview" />`
        : ''}
    `;
  } catch (err) {
    out.textContent = 'Network error: ' + err.message;
  }
});

// ---- Query ----
document.getElementById('query-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  const out  = document.getElementById('query-result');
  out.textContent = 'Searching…';

  try {
    const res  = await fetch('/api/query', { method: 'POST', body: data });
    const json = await res.json();
    if (!res.ok) { out.textContent = 'Error: ' + (json.detail ?? res.statusText); return; }

    if (!json.results || json.results.length === 0) {
      out.textContent = 'No matches found.';
      return;
    }

    out.innerHTML = json.results.map(r => `
      <div class="match-card">
        <h3>${r.name}</h3>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar" style="width:${(r.score * 100).toFixed(1)}%"></div>
        </div>
        <small>${(r.score * 100).toFixed(1)}% confidence</small>
        ${r.match_img_b64
          ? `<br/><img src="data:image/png;base64,${r.match_img_b64}" alt="match visualization" />`
          : ''}
      </div>
    `).join('');
  } catch (err) {
    out.textContent = 'Network error: ' + err.message;
  }
});

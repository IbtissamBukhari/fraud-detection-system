/**
 * FRAUD DETECTION AI — app.js
 * Connects to FastAPI backend at http://localhost:8000
 */

const API_BASE = 'http://localhost:8000';

// ─── Init ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadHistory();
  loadModelInfo();
});

// ─── Health Check ─────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      dot.classList.add('connected');
      text.textContent = 'API connected';
    } else { throw new Error(); }
  } catch {
    dot.classList.add('error');
    text.textContent = 'API offline';
  }
}

// ─── Run Analysis ─────────────────────────────────────────
async function runAnalysis() {
  const btn     = document.getElementById('analyzeBtn');
  const btnText = btn.querySelector('.btn-text');
  const spinner = document.getElementById('btnSpinner');

  const TransactionAmt   = parseFloat(document.getElementById('txnAmt').value);
  const ProductCD        = document.getElementById('productCD').value;
  const card4            = document.getElementById('card4').value;
  const card6            = document.getElementById('card6').value;
  const P_emaildomain    = document.getElementById('emailDomain').value.trim() || 'gmail.com';
  const transaction_hour = parseInt(document.getElementById('txnHour').value) || 0;
  const transaction_day  = parseInt(document.getElementById('txnDay').value)  || 1;
  const transaction_week = parseInt(document.getElementById('txnWeek').value) || 1;

  if (isNaN(TransactionAmt) || TransactionAmt <= 0) {
    showToast('Please enter a valid transaction amount.', 'error');
    return;
  }

  btn.disabled = true;
  btnText.textContent = 'Analyzing…';
  spinner.style.display = 'inline-block';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        TransactionAmt, ProductCD, card4, card6,
        P_emaildomain, transaction_hour, transaction_day, transaction_week,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();

    // Render result directly — NO scrolling, result sits right below the form
    renderRiskOutput(data);
    loadHistory();
    showToast('Analysis complete.', 'success');

  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Analyze transaction';
    spinner.style.display = 'none';
  }
}

// ─── Render Risk Output ───────────────────────────────────
function renderRiskOutput(data) {
  // Swap placeholder for result — both sit in the DOM below the form
  document.getElementById('riskPlaceholder').style.display = 'none';
  document.getElementById('riskResult').style.display = 'block';

  const prob       = data.fraud_probability;
  const pct        = (prob * 100).toFixed(2) + '%';
  const level      = (data.risk_level || 'Low').toLowerCase();
  const confidence = data.confidence || '—';

  document.getElementById('riskPct').textContent = pct;

  // Donut — reset then animate
  const circumference = 2 * Math.PI * 48;
  const fill = document.getElementById('donutFill');
  fill.style.transition = 'none';
  fill.style.strokeDashoffset = circumference;
  fill.getBoundingClientRect(); // force reflow
  fill.style.transition = 'stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke .4s';
  fill.style.strokeDashoffset = circumference * (1 - prob);
  fill.style.stroke = level === 'high' ? '#B91C1C' : level === 'medium' ? '#B45309' : '#2E7D52';

  const badge = document.getElementById('riskBadge');
  badge.textContent = data.risk_level + ' Risk';
  badge.className = `risk-badge ${level}`;

  const descs = {
    low:    'Low fraud risk. Transaction appears legitimate.',
    medium: 'Medium fraud risk. Manual review recommended.',
    high:   'High fraud risk. Flag for immediate investigation.',
  };
  document.getElementById('riskDesc').textContent = descs[level] || '—';
  document.getElementById('confidenceVal').textContent = confidence;

  // Hero mini bar
  document.getElementById('heroRiskBar').style.width = (prob * 100) + '%';

  // Feature bars
  const list = document.getElementById('featureList');
  list.innerHTML = '';
  const features = (data.top_features || []).slice(0, 8);
  if (!features.length) return;

  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.0001);
  features.forEach((f, i) => {
    const barPct = Math.round((Math.abs(f.shap_value) / maxAbs) * 100);
    const isPos  = f.shap_value >= 0;
    const item   = document.createElement('div');
    item.className = 'feature-item';
    item.style.animationDelay = `${i * 0.06}s`;
    item.innerHTML = `
      <span class="f-name">${escHtml(f.feature)}</span>
      <div class="f-bar-wrap">
        <div class="f-bar ${isPos ? 'pos' : 'neg'}" style="width:${barPct}%"></div>
      </div>
      <span class="f-val">${f.shap_value >= 0 ? '+' : ''}${f.shap_value.toFixed(4)}</span>
    `;
    list.appendChild(item);
  });
}

// ─── History ──────────────────────────────────────────────
async function loadHistory() {
  try {
    const res  = await fetch(`${API_BASE}/history`);
    if (!res.ok) return;
    renderHistory(await res.json());
  } catch { /* silent */ }
}

function renderHistory(rows) {
  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = '';
  if (!rows || !rows.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No predictions yet.</td></tr>`;
    return;
  }
  [...rows].reverse().forEach(row => {
    const level = (row.risk_level || 'low').toLowerCase();
    const pct   = ((row.fraud_probability || 0) * 100).toFixed(2) + '%';
    const tr    = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono">${escHtml(row.timestamp || '—')}</td>
      <td>$${Number(row.transaction_amount || 0).toFixed(2)}</td>
      <td class="mono">${pct}</td>
      <td><span class="tbl-badge ${level}">${escHtml(row.risk_level || '—')}</span></td>
      <td>${escHtml(row.confidence || '—')}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function exportCSV() {
  try {
    const res = await fetch(`${API_BASE}/history/export`);
    if (!res.ok) { showToast('No history to export.', 'error'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'predictions_history.csv'; a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported.', 'success');
  } catch { showToast('Could not reach API.', 'error'); }
}

// ─── Model Info ───────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) return;
    const info = await res.json();
    setText('modelType',      info.model || '—');
    setText('modelAuc',       info.auc_roc != null ? info.auc_roc.toFixed(3) : '—');
    setText('modelRows',      info.training_rows != null ? Number(info.training_rows).toLocaleString() : '—');
    setText('modelFeatures',  info.features_used != null ? info.features_used : '—');
    setText('modelFraudRate', info.fraud_rate_in_training || '—');
    setText('modelDataset',   info.training_data || '—');
  } catch { /* silent */ }
}

// ─── Toast ────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = 'toast'; }, 3500);
}

// ─── Helpers ─────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.target.matches('.form-input, .form-select')) runAnalysis();
});
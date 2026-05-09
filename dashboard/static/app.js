'use strict';

const DATASET_INFO = {
  pcos: {
    name: "PCOS Dataset — Prasoon Kottarathil + Shreyas Vedpathak",
    meta: "Kaggle · CC0 · ~541 rows merged",
    note: "LH/AMH ratios, follicle count, cycle length — primary PCOS markers"
  },
  breast_cancer: {
    name: "Breast Cancer Wisconsin (Diagnostic) — UCI",
    meta: "Kaggle · CC0 · 569 rows",
    note: "Cell nucleus measurements (radius, texture, perimeter). Top 10 means selected."
  },
  cervical: {
    name: "Cervical Cancer Risk Factors — Ranzeet013",
    meta: "Kaggle · CC BY 4.0 · 858 rows",
    note: "Biopsy as ground truth. Top 10 features selected via SelectKBest (f_classif)."
  },
  thyroid: {
    name: "Thyroid Disease Dataset — Yasser Hessein",
    meta: "Kaggle · Public · 3,771 rows",
    note: "TSH, T3, T4, FTI levels. Binary: hypothyroid vs negative."
  },
  diabetes: {
    name: "Pima Indians Diabetes Database — UCI",
    meta: "Kaggle · CC0 · 768 rows",
    note: "Female patients ≥21 years. Glucose, insulin, BMI, diabetes pedigree function."
  },
  cardiovascular: {
    name: "Cardiovascular Disease Dataset — Sulianova",
    meta: "Kaggle · CC BY 4.0 · 70,000 rows",
    note: "Systolic/diastolic BP, cholesterol, glucose, BMI. Largest dataset in FEM-NET."
  },
  heart: {
    name: "Heart Disease UCI — Redwan Karim Sony",
    meta: "Kaggle · CC BY 4.0 · 920 rows",
    note: "Cleveland + Hungary + VA + Switzerland combined. Target column binarised (num > 0)."
  }
};

function updateProvenance() {
  const condition = document.querySelector('select[name="condition"]').value;
  const info = DATASET_INFO[condition];
  const card = document.getElementById('provenance-card');
  if (!info) { card.style.display = 'none'; return; }
  document.getElementById('provenance-name').textContent = info.name;
  document.getElementById('provenance-meta').textContent = info.meta;
  document.getElementById('provenance-note').textContent = info.note;
  card.style.display = 'block';
}

const API = '';

// ── Gauge ──────────────────────────────────────────────────────────────────
function drawGauge(score) {
  const svg = document.getElementById('gauge-svg');
  const R = 90, CX = 110, CY = 110;
  const startAngle = Math.PI, endAngle = 2 * Math.PI;
  const angle = startAngle + score * Math.PI;

  const color = score < 0.35 ? '#059669' : score < 0.65 ? '#d97706' : '#dc2626';

  const x1 = CX + R * Math.cos(startAngle), y1 = CY + R * Math.sin(startAngle);
  const x2 = CX + R * Math.cos(endAngle), y2 = CY + R * Math.sin(endAngle);
  const fx = CX + R * Math.cos(angle), fy = CY + R * Math.sin(angle);
  const largeArc = score > 0.5 ? 1 : 0;

  svg.innerHTML = `
    <path d="M ${x1} ${y1} A ${R} ${R} 0 1 1 ${x2} ${y2}" fill="none" stroke="#e5e7eb" stroke-width="18" stroke-linecap="round"/>
    <path d="M ${x1} ${y1} A ${R} ${R} 0 ${largeArc} 1 ${fx} ${fy}" fill="none" stroke="${color}" stroke-width="18" stroke-linecap="round"/>
    <circle cx="${fx}" cy="${fy}" r="8" fill="${color}"/>
    <circle cx="${CX}" cy="${CY}" r="4" fill="#7c3aed"/>
  `;
}

// ── Form helpers ───────────────────────────────────────────────────────────
function getFormData() {
  const form = document.getElementById('patient-form');
  const data = {};
  const inputs = form.querySelectorAll('input, select');
  inputs.forEach(el => {
    data[el.name] = el.type === 'text' || el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
  });
  return data;
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Running…';
  } else {
    btn.disabled = false;
    btn.textContent = btn.dataset.label;
  }
}

// ── Predict ────────────────────────────────────────────────────────────────
async function runPrediction(event) {
  event.preventDefault();
  setLoading('predict-btn', true);
  const payload = getFormData();

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    // update gauge
    document.getElementById('gauge-score').textContent = (data.risk_score * 100).toFixed(1) + '%';
    const levelEl = document.getElementById('gauge-level');
    levelEl.textContent = data.risk_level + ' Risk';
    levelEl.className = `gauge-level level-${data.risk_level}`;
    document.getElementById('model-version').textContent = `Model: ${data.model_version}`;
    drawGauge(data.risk_score);
    document.getElementById('gauge-placeholder').style.display = 'none';
    document.getElementById('gauge-result').style.display = 'flex';

    // auto-fetch insights
    await runInsights(payload, data.risk_score);
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    setLoading('predict-btn', false);
  }
}

// ── Insights ───────────────────────────────────────────────────────────────
async function runInsights(patientData, riskScore) {
  const insightWrap = document.getElementById('insight-wrap');
  const insightPlaceholder = document.getElementById('insight-placeholder');
  insightPlaceholder.style.display = 'flex';
  insightWrap.style.display = 'none';

  try {
    const res = await fetch(`${API}/api/insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient: patientData, risk_score: riskScore }),
    });

    if (!res.ok) throw new Error(await res.text());
    const card = await res.json();

    // key indicators
    const indList = document.getElementById('indicators-list');
    indList.innerHTML = card.key_indicators.map(i => `<li>${escHtml(i)}</li>`).join('');

    // evidence
    document.getElementById('evidence-text').textContent = card.evidence_summary;

    // next steps
    const stepsList = document.getElementById('steps-list');
    stepsList.innerHTML = card.recommended_next_steps.map(s => `<li>${escHtml(s)}</li>`).join('');

    insightPlaceholder.style.display = 'none';
    insightWrap.style.display = 'block';
  } catch (err) {
    insightPlaceholder.innerHTML = `<div class="icon">⚠️</div><div>Insights unavailable: ${escHtml(err.message)}</div>`;
  }
}

// ── Federated Status ───────────────────────────────────────────────────────
async function loadFLStatus() {
  const condition = getFormData().condition || 'pcos';
  try {
    const res = await fetch(`${API}/api/federated/status?condition=${condition}`);
    const data = await res.json();
    document.getElementById('fl-rounds').textContent = data.num_rounds_completed;
    document.getElementById('fl-hospitals').textContent = data.participating_hospitals;
    document.getElementById('fl-auc').textContent = data.last_global_auc > 0
      ? (data.last_global_auc * 100).toFixed(1) + '%'
      : '—';
  } catch {
    // silent
  }
}

async function runFederated() {
  const condition = getFormData().condition || 'pcos';
  setLoading('fl-btn', true);
  document.getElementById('fl-status-msg').textContent = 'Starting federated round…';
  try {
    const res = await fetch(`${API}/api/federated/run?condition=${condition}&rounds=5`, { method: 'POST' });
    const data = await res.json();
    document.getElementById('fl-status-msg').textContent =
      `Round started for ${data.condition}. Refresh status in ~30s.`;
  } catch (err) {
    document.getElementById('fl-status-msg').textContent = 'Error: ' + err.message;
  } finally {
    setLoading('fl-btn', false);
  }
}

function escHtml(str) {
  return str.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('patient-form').addEventListener('submit', runPrediction);
  document.getElementById('fl-btn').addEventListener('click', runFederated);
  document.getElementById('fl-refresh-btn').addEventListener('click', loadFLStatus);
  document.querySelector('select[name="condition"]').addEventListener('change', updateProvenance);
  drawGauge(0);
  loadFLStatus();
  updateProvenance();
});

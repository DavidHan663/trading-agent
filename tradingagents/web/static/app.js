const $ = (selector) => document.querySelector(selector);
const labels = {queued: '排队中', running: '分析中', completed: '已完成', failed: '未完成'};
let refreshing = false;
function alertMessage(message) { $('#alert').textContent = message; $('#alert').hidden = !message; }
async function api(path, options = {}) {
  const response = await fetch(new URL(path, window.location.origin), {credentials: 'same-origin', ...options});
  if (!response.ok) {
    let message = '请求失败，请稍后重试。';
    try { const body = await response.json(); if (typeof body.detail === 'string') message = body.detail; else if (response.status === 422) message = '请检查标的代码、日期及分析师选择。'; } catch { /* use default */ }
    if (response.status === 401) message = '登录已失效，请刷新页面重新登录。';
    throw new Error(message);
  }
  return response;
}
function cell(row, value) { const td = document.createElement('td'); td.textContent = value; row.append(td); return td; }
function textLine(parent, label, value) {
  const line = document.createElement('p'); line.textContent = `${label}：${value || '未知'}`; parent.append(line);
}
function safeSource(value) {
  try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; }
  catch { return null; }
}
function renderEvidence(data) {
  const container = $('#evidence-claims'); container.replaceChildren();
  const claims = Array.isArray(data.claims) ? data.claims : [];
  $('#evidence-status').textContent = claims.length ? '复核结果仅供参考，来源和推断仍需独立判断。' : '此报告暂无逐条复核数据。';
  const records = new Map((Array.isArray(data.records) ? data.records : []).filter(r => r && typeof r === 'object').map(r => [r.id, r]));
  for (const claim of claims) {
    if (!claim || typeof claim !== 'object') continue;
    const details = document.createElement('details'); details.className = 'evidence-claim';
    const heading = document.createElement('summary');
    const state = claim.decision === 'allow' ? '通过' : '需复核';
    heading.textContent = `${state} · ${claim.claim_id || '未知'} · ${claim.claim || '无结论文字'}`;
    details.append(heading);
    textLine(details, '原因', Array.isArray(claim.reason_flags) && claim.reason_flags.length ? claim.reason_flags.join('、') : '无');
    textLine(details, '解释', claim.reasoning);
    if (claim.failure_reason) textLine(details, '复核状态', claim.failure_reason);
    for (const missing of Array.isArray(claim.missing_information) ? claim.missing_information : []) textLine(details, '缺失信息', missing);
    for (const quote of Array.isArray(claim.evidence_quotes) ? claim.evidence_quotes : []) {
      if (!quote || typeof quote !== 'object') continue;
      const record = records.get(quote.evidence_id) || {};
      textLine(details, `引用 ${quote.evidence_id || '未知'}`, quote.quote);
      textLine(details, '标的', `请求 ${record.requested_ticker || '未知'} / 目标 ${record.target_ticker || '未知'} / 来源确认 ${record.source_ticker || '未知'}`);
      textLine(details, '时间', `可用 ${record.available_at || '未知'} / 所属期间 ${record.period_end || '未知'} / 获取 ${record.fetched_at || '未知'}`);
      const source = safeSource(record.source_url);
      if (source) { const link = document.createElement('a'); link.href = source; link.textContent = '查看来源 ↗'; link.target = '_blank'; link.rel = 'noopener noreferrer'; details.append(link); }
    }
    container.append(details);
  }
}
async function showReport(job) {
  try {
    const path = `/api/jobs/${job.id}/report`;
    const report = await (await api(path)).text();
    $('#report-title').textContent = `${job.request.ticker} · ${job.request.trade_date}`;
    $('#report-content').textContent = report;
    $('#download').href = path;
    $('#evidence-claims').replaceChildren();
    $('#evidence-status').textContent = '正在读取复核数据…';
    $('#report-dialog').showModal();
    try { renderEvidence(await (await api(`/api/jobs/${job.id}/evidence`)).json()); }
    catch { $('#evidence-status').textContent = '复核数据暂时无法读取，完整报告仍可查看和下载。'; }
  } catch (error) { alertMessage(error.message); }
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const jobs = await (await api('/api/jobs')).json();
    $('#jobs').replaceChildren();
    $('#empty').hidden = jobs.length > 0;
    $('#job-count').textContent = jobs.length;
    for (const job of jobs) {
      const row = document.createElement('tr');
      cell(row, job.request.ticker); cell(row, job.request.trade_date);
      const status = document.createElement('span');
      status.className = `status ${job.status}`; status.textContent = labels[job.status] || job.status;
      cell(row, '').append(status); cell(row, job.signal || '—');
      cell(row, new Date(job.created_at).toLocaleString('zh-CN', {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'}));
      const action = cell(row, '—');
      if (job.status === 'completed' || job.status === 'failed') {
        action.textContent = '';
        const button = document.createElement('button'); button.type = 'button'; button.className = 'ghost';
        button.textContent = job.status === 'completed' ? '查看报告 ↗' : '查看原因';
        button.onclick = () => job.status === 'completed' ? showReport(job) : alertMessage(job.error);
        action.append(button);
      }
      $('#jobs').append(row);
    }
  } finally { refreshing = false; }
}
$('#analysis-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = new FormData(event.target);
  const analysts = data.getAll('analysts');
  if (!analysts.length) { alertMessage('请至少选择一位分析师。'); return; }
  $('#submit').disabled = true; alertMessage('');
  try {
    await api('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ticker:data.get('ticker').trim().toUpperCase(), trade_date:data.get('trade_date'), asset_type:data.get('asset_type'), rounds:Number(data.get('rounds')), analysts})});
    await refresh(); $('#history').scrollIntoView({behavior:'smooth'});
  } catch (error) { alertMessage(error.message); }
  finally { $('#submit').disabled = false; }
});
$('#refresh').onclick = () => refresh().catch(error => alertMessage(error.message));
$('#close-report').onclick = () => $('#report-dialog').close();
$('#asset-type').onchange = () => {
  const crypto = $('#asset-type').value === 'crypto';
  const fundamental = document.querySelector('input[value="fundamentals"]');
  fundamental.disabled = crypto; fundamental.checked = !crypto;
  $('#ticker').placeholder = crypto ? '例如 BTC-USD / ETH-USD' : '例如 AAPL / 0700.HK';
};
const today = new Date();
const dateString = `${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`;
$('#trade-date').value = dateString; $('#trade-date').max = dateString;
async function initialize() {
  try {
    const config = await (await api('/api/config')).json();
    $('#provider').textContent = config.provider;
    $('#deep-model').textContent = config.deep_model;
    $('#quick-model').textContent = config.quick_model;
    $('#engine-status').textContent = config.configured ? '● 已配置 · 等待研究任务' : '○ 尚未配置模型密钥';
    if (!config.configured) alertMessage('请在服务器环境变量中配置模型 API 密钥，再开始研究。');
    await refresh();
  } catch (error) { alertMessage(error.message); }
}
initialize();
setInterval(() => { if (!document.hidden) refresh().catch(error => alertMessage(error.message)); }, 5000);

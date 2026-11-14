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
async function showReport(job) {
  try {
    const path = `/api/jobs/${job.id}/report`;
    const report = await (await api(path)).text();
    $('#report-title').textContent = `${job.request.ticker} · ${job.request.trade_date}`;
    $('#report-content').textContent = report;
    $('#download').href = path;
    $('#report-dialog').showModal();
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

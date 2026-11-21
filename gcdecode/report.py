"""解码过程可视化:生成自包含 HTML 报告。

页面分四块(对应需求):
1. 文法定义(原文 + 脱糖后规则);
2. 每一步的可选词与被屏蔽词(含屏蔽原因、原始/调整后分数);
3. 解码结果(完整文本、词项序列、是否被文法接受);
4. 各种约束方式的耗时对比(掩码计算/打分/总耗时,含缓存命中)。
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

from .decoding import DecodeTrace

__all__ = ["render_html_report", "ReportData"]


@dataclass
class ReportData:
    grammar_text: str
    desugared_text: str
    traces: dict
    stats: dict


def _esc(s: str) -> str:
    return html.escape(str(s))


def _fmt_ns(ns: int) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f} µs"
    return f"{ns} ns"


def render_html_report(data: ReportData) -> str:
    """生成单文件 HTML(内联 CSS/JS,无外部依赖)。"""
    grammar_json = json.dumps(data.grammar_text, ensure_ascii=False)
    desugared_json = json.dumps(data.desugared_text, ensure_ascii=False)
    traces_json = json.dumps(data.traces, ensure_ascii=False)
    stats_json = json.dumps(data.stats, ensure_ascii=False)

    return _TEMPLATE.replace("__GRAMMAR__", grammar_json) \
        .replace("__DESUGARED__", desugared_json) \
        .replace("__TRACES__", traces_json) \
        .replace("__STATS__", stats_json)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>文法约束解码过程报告</title>
<style>
  :root {
    --bg: #0f1419; --panel: #1a2027; --border: #2d3742;
    --text: #d7dde3; --muted: #8a96a3; --accent: #4ea1ff;
    --ok: #3fb950; --bad: #f85149; --warn: #d29922; --chip: #26303b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei",
          "Segoe UI", monospace;
  }
  header {
    padding: 18px 28px; border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #16202c, #0f1419);
  }
  header h1 { margin: 0 0 4px; font-size: 20px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { max-width: 1280px; margin: 0 auto; padding: 24px; }
  section {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 22px; margin-bottom: 22px;
  }
  section h2 {
    margin: 0 0 14px; font-size: 16px; color: var(--accent);
    display: flex; align-items: center; gap: 8px;
  }
  section h2 .num {
    background: var(--accent); color: #08111c; width: 22px; height: 22px;
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; font-size: 12px; font-weight: 700;
  }
  pre {
    background: #0c1116; border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; overflow-x: auto;
    font-size: 12.5px; color: #b8c4d0; white-space: pre-wrap;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td {
    text-align: left; padding: 8px 10px;
    border-bottom: 1px solid var(--border); vertical-align: top;
  }
  th { color: var(--muted); font-weight: 600; }
  .chip {
    display: inline-block; background: var(--chip);
    border: 1px solid var(--border); border-radius: 5px;
    padding: 1px 7px; margin: 2px 3px 2px 0; font-size: 12px;
  }
  .chip.allowed { border-color: #2f6b3a; color: #7ee78c; }
  .chip.blocked { border-color: #6b2f2f; color: #f08a85;
    text-decoration: line-through; opacity: .75; }
  .chip.eos { border-color: #2f556b; color: #7cc4ff; }
  .badge {
    display: inline-block; padding: 2px 9px; border-radius: 11px;
    font-size: 12px; font-weight: 600;
  }
  .badge.ok { background: rgba(63,185,80,.15); color: var(--ok); }
  .badge.bad { background: rgba(248,81,73,.15); color: var(--bad); }
  .result-box {
    background: #0c1116; border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; word-break: break-all;
  }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
  .bar-label { width: 150px; color: var(--muted); font-size: 12.5px; }
  .bar-track { flex: 1; background: #0c1116; border-radius: 6px;
    height: 22px; overflow: hidden; border: 1px solid var(--border); }
  .bar-fill { height: 100%; background: linear-gradient(90deg,#2f6df6,#4ea1ff);
    transition: width .3s; }
  .bar-val { width: 110px; text-align: right; font-size: 12.5px; }
  .step-nav { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; }
  button {
    background: var(--chip); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px;
  }
  button:hover { border-color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
  .muted { color: var(--muted); }
  .score-pos { color: var(--ok); } .score-neg { color: var(--bad); }
  details summary { cursor: pointer; color: var(--muted); margin: 6px 0; }
</style>
</head>
<body>
<header>
  <h1>文法约束解码过程报告</h1>
  <p>贪心 / 温度采样 / 束搜索三种解码方式 · 逐步掩码与耗时对比</p>
</header>
<main>
  <section>
    <h2><span class="num">1</span>文法定义</h2>
    <div class="grid2">
      <div>
        <div class="muted" style="margin-bottom:6px">DSL 原文</div>
        <pre id="grammarSrc"></pre>
      </div>
      <div>
        <div class="muted" style="margin-bottom:6px">脱糖后核心结构</div>
        <pre id="desugaredSrc"></pre>
      </div>
    </div>
  </section>

  <section>
    <h2><span class="num">2</span>每一步的可选词与被屏蔽词</h2>
    <div class="step-nav">
      <button id="prevBtn">← 上一步</button>
      <span id="stepInfo" class="muted"></span>
      <button id="nextBtn">下一步 →</button>
      <select id="modeSel" style="margin-left:auto;background:var(--chip);
        color:var(--text);border:1px solid var(--border);border-radius:6px;
        padding:6px 10px;"></select>
    </div>
    <div id="stepDetail"></div>
  </section>

  <section>
    <h2><span class="num">3</span>解码结果</h2>
    <div id="results"></div>
  </section>

  <section>
    <h2><span class="num">4</span>约束方式耗时对比</h2>
    <div id="timing"></div>
    <details><summary>缓存与机器状态统计</summary><div id="cacheStats"></div></details>
  </section>
</main>

<script>
const GRAMMAR = __GRAMMAR__;
const DESUGARED = __DESUGARED__;
const TRACES = __TRACES__;
const STATS = __STATS__;

document.getElementById('grammarSrc').textContent = GRAMMAR;
document.getElementById('desugaredSrc').textContent = DESUGARED;

const modeSel = document.getElementById('modeSel');
const modeNames = Object.keys(TRACES);
modeNames.forEach(m => {
  const o = document.createElement('option');
  o.value = m; o.textContent = m; modeSel.appendChild(o);
});

let curMode = modeNames[0];
let curStep = 0;

function esc(s){return String(s).replace(/[&<>"]/g,c=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmtNs(ns){
  if(ns>=1e6) return (ns/1e6).toFixed(2)+' ms';
  if(ns>=1e3) return (ns/1e3).toFixed(1)+' µs';
  return ns+' ns';
}
function chip(piece, cls, title){
  return `<span class="chip ${cls}" title="${esc(title||'')}">${esc(piece)}</span>`;
}

function renderStep(){
  const tr = TRACES[curMode];
  const steps = tr.steps || [];
  if(!steps.length){
    document.getElementById('stepDetail').innerHTML =
      '<p class="muted">该模式不保留逐步明细(束搜索见快照)。</p>';
    document.getElementById('stepInfo').textContent = '';
    return;
  }
  curStep = Math.max(0, Math.min(curStep, steps.length-1));
  const st = steps[curStep];
  document.getElementById('stepInfo').textContent =
    `第 ${curStep+1} / ${steps.length} 步 · 机器状态 #${st.state_id}`;
  const allowed = (st.allowed_ids||[]).map(id=>
    chip(st.pieces[id],'allowed','可用词')).join('');
  const blocked = (st.blocked_ids||[]).map(id=>
    chip(st.pieces[id],'blocked','被屏蔽,分数置为 -∞')).join('');
  const eosCls = st.eos_allowed ? 'eos' : 'blocked';
  let html = `<div style="margin-bottom:10px">
      <span class="muted">当前前缀:</span>
      <code>${esc(st.prefix||'(空)')}</code>
    </div>
    <div style="margin-bottom:6px"><b style="color:#7ee78c">可选词
      (${(st.allowed_ids||[]).length})</b>
      <span class="muted">· EOS:
      <span class="chip ${eosCls}">${esc(st.eos_piece)}</span></span></div>
    <div style="margin-bottom:14px">${allowed ||
      '<span class="muted">无(仅 EOS)</span>'}</div>
    <div style="margin-bottom:6px"><b style="color:#f08a85">被屏蔽词
      (${(st.blocked_ids||[]).length})</b>
      <span class="muted">· 分数置为 -∞</span></div>
    <div>${blocked || '<span class="muted">无</span>'}</div>
    <div style="margin-top:12px" class="muted">
      本步选用: <b style="color:var(--text)">${esc(st.chosen_piece)}</b>
      · 掩码计算 ${fmtNs(st.mask_ns)} · 打分 ${fmtNs(st.score_ns)}
      · 整步 ${fmtNs(st.total_ns)}
    </div>`;
  if(st.top_blocked){
    const [id,piece,logit] = st.top_blocked;
    html += `<div class="muted" style="margin-top:6px">最高分被屏蔽词:
      <span class="chip blocked">${esc(piece)}</span>
      原始分数 ${logit.toFixed(3)}</div>`;
  }
  document.getElementById('stepDetail').innerHTML = html;
  document.getElementById('prevBtn').disabled = curStep===0;
  document.getElementById('nextBtn').disabled = curStep===steps.length-1;
}

document.getElementById('prevBtn').onclick = ()=>{curStep--;renderStep();};
document.getElementById('nextBtn').onclick = ()=>{curStep++;renderStep();};
modeSel.onchange = e=>{curMode=e.target.value;curStep=0;renderStep();};

function renderResults(){
  const box = document.getElementById('results');
  box.innerHTML = modeNames.map(m=>{
    const tr = TRACES[m];
    const cls = tr.accepted ? 'ok' : 'bad';
    const txt = tr.accepted ? '被文法完整接受' : '未被接受';
    return `<div style="margin-bottom:14px">
      <div class="muted" style="margin-bottom:4px">${esc(m)}</div>
      <div class="result-box">
        <span class="badge ${cls}">${txt}</span>
        <code style="margin-left:8px">${esc(tr.final_text||'(空)')}</code>
      </div>
      <div class="muted" style="margin-top:4px;font-size:12px">
        词项数 ${(tr.token_ids||[]).length} · 总耗时 ${fmtNs(tr.total_ns||0)}</div>
    </div>`;
  }).join('');
}

function renderTiming(){
  const rows = modeNames.map(m=>{
    const tr = TRACES[m];
    return {name:m, mask:tr.total_mask_ns||0, score:tr.total_score_ns||0,
            total:tr.total_ns||0,
            hits:tr.mask_cache_hits||0, miss:tr.mask_cache_misses||0,
            flat:tr.flat_mask_uses||0};
  });
  const maxTotal = Math.max(...rows.map(r=>r.total),1);
  document.getElementById('timing').innerHTML = rows.map(r=>`
    <div style="margin-bottom:16px">
      <div class="muted" style="margin-bottom:4px">${esc(r.name)}</div>
      ${bar('总耗时', r.total, maxTotal, fmtNs(r.total))}
      ${bar('约束掩码计算', r.mask, maxTotal, fmtNs(r.mask))}
      ${bar('模型打分', r.score, maxTotal, fmtNs(r.score))}
    </div>`).join('') + `
    <div style="margin-top:8px">
      <div class="muted">无约束基线总耗时: ${fmtNs(STATS.unconstrained_total_ns||0)}</div>
      <div class="muted">约束相对开销: ${STATS.overhead_pct ?? '—'}%</div>
    </div>`;
  document.getElementById('cacheStats').innerHTML = rows.map(r=>`
    <div class="bar-row">
      <span class="bar-label">${esc(r.name)}</span>
      <span class="muted">掩码缓存命中 ${r.hits} / 未命中 ${r.miss}
        · 平坦预计算命中 ${r.flat}</span>
    </div>`).join('') +
    `<div class="muted">机器状态总数:${STATS.num_machine_states??'—'} ·
      平坦规则掩码:${STATS.num_flat_masks??'—'}</div>`;
}
function bar(label, val, max, text){
  const pct = max>0 ? (val/max*100) : 0;
  return `<div class="bar-row">
    <span class="bar-label">${esc(label)}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    <span class="bar-val">${esc(text)}</span></div>`;
}

renderStep(); renderResults(); renderTiming();
</script>
</body>
</html>
"""

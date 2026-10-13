"""Offline report using the inspected activation-lab table/control layout."""
import hashlib
import json


def render_report(report):
    payload = json.dumps(report, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    revision = hashlib.sha256(payload.encode()).hexdigest()
    return TEMPLATE.replace('__DATA__', payload).replace('__REVISION__', revision)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="artifact-revision" content="__REVISION__">
<title>Recurrent state lab</title><style>
:root{font:17px system-ui,sans-serif;color:#172b3a;background:#f4f6f8;color-scheme:light}
body{margin:0}main{max-width:1100px;margin:auto;padding:32px 24px 60px}h1{font-size:2rem;line-height:1.2;margin:.3em 0}
h2{font-size:1.35rem;margin:1.8em 0 .6em}p{max-width:88ch;line-height:1.6}.eyebrow{color:#456170;font-weight:600}
.notice{border-left:4px solid #9c6a00;background:#fff7e5;padding:14px 18px}.controls{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}
label{display:grid;gap:6px;font-weight:600}select{font:inherit;padding:8px;background:white;border:1px solid #8193a1;border-radius:4px}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0}th,td{padding:10px 12px;border:1px solid #d5dfe4;text-align:left}
th{background:#eaf0f4}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}caption{text-align:left;font-weight:600;padding:8px 0}
select:focus-visible,summary:focus-visible{outline:3px solid #006faa;outline-offset:3px}details{margin-top:18px;background:white;padding:14px;border:1px solid #d5dfe4}
summary{font-weight:600;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9rem;line-height:1.5}code{overflow-wrap:anywhere}a{color:#006090}
footer{margin-top:30px;line-height:1.5;color:#456170}@media(max-width:640px){main{padding:20px 14px 40px}h1{font-size:1.7rem}th,td{padding:8px}.controls{display:block}label{margin-bottom:16px}}
@media print{body{background:white}main{max-width:none;padding:0}.scroll{overflow:visible}details{break-inside:avoid}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><main>
<div class="eyebrow">Gated DeltaNet CPU reference · Learned synthetic recall · Local report</div>
<h1>Recall accuracy as inputs grow</h1>
<p>Train on 16-token sequences with two key/value pairs. Evaluate held-out 16, 32 and 64-token sequences. A shared block can be applied once or twice; each application keeps its own state.</p>
<div class="notice">This small synthetic experiment does not establish language ability or reproduce a full-scale Gated DeltaNet model. Two loops keep the same parameter count but use more computation and state.</div>
<div class="controls"><label>Model seed<select id="seed"></select></label><label>Shared block loops<select id="loops"><option value="1">1 loop</option><option value="2">2 loops</option></select></label></div>
<h2>Held-out recall</h2><p>Accuracy counts only query positions; filler and key/value context are excluded. The constant64 predictor uses the same held-out labels. Validation never selects a model or an epoch.</p><div class="scroll" id="quality"></div>
<h2>Prefill, supplied token processing and state</h2><p>Every measured repetition is retained after a warmup. Prefill reads the full input. The token measurement feeds a supplied token 16 times; it does not generate language. State bytes include recurrent and convolution tensors for one sequence and exclude model weights and activations. Before/after process RSS can miss peaks.</p><div class="scroll" id="timing"></div>
<h2>Full, split, one-token and reset controls</h2><p>Each control uses two held-out sequences. Errors compare logits with full processing. Reset means starting with no state; batch reordering checks that sequences do not share state.</p><div class="scroll" id="checks"></div>
<h2>Training history</h2><p id="training-note"></p><div class="scroll" id="curve"></div>
<details><summary>Configuration, identity and limits</summary><pre id="evidence"></pre></details>
<footer>Methods: <a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks, ICLR 2025</a> and <a href="https://arxiv.org/abs/2312.04927">Zoology, ICLR 2024</a>. Code: FLA (MIT), Zoology (Apache 2.0). No network requests. Report revision <code>__REVISION__</code>.</footer>
</main><script type="application/json" id="data">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent),get=id=>document.getElementById(id);
const num=v=>Number(v).toPrecision(6),percent=v=>(v*100).toFixed(2)+'%';
function table(id,headers,rows){const t=document.createElement('table'),thead=document.createElement('thead'),head=document.createElement('tr');headers.forEach(v=>{const th=document.createElement('th');th.scope='col';th.textContent=v;head.append(th)});thead.append(head);t.append(thead);const body=document.createElement('tbody');rows.forEach(values=>{const row=document.createElement('tr');values.forEach(v=>{const cell=document.createElement('td');cell.textContent=String(v);row.append(cell)});body.append(row)});t.append(body);get(id).replaceChildren(t)}
data.protocol.seeds.forEach(seed=>{const o=document.createElement('option');o.value=seed;o.textContent=seed;get('seed').append(o)});
function render(){const run=data.runs.find(r=>r.seed===Number(get('seed').value)&&r.loops===Number(get('loops').value));
table('quality',['Input tokens','Correct/queries','Accuracy','Constant 64 baseline'],run.evaluations.map(e=>[e.length,e.quality.correct+'/'+e.quality.queries,percent(e.quality.accuracy),percent(e.constant_value64_baseline.accuracy)]));
table('timing',['Input tokens','Run','Prefill(ms)','Supplied token(ms)','State(bytes)','RSS after(MiB)'],run.evaluations.flatMap(e=>e.timing.map(t=>[e.length,t.repetition,num(t.prefill_ms),num(t.supplied_one_token_ms),t.state_bytes,num(t.rss_after_bytes/1048576)])));
table('checks',['Input tokens','Full/stream error','Split error','One-token error','Reset exact','Batch isolated'],run.evaluations.map(e=>[e.length,num(e.controls.full_vs_stream_max_abs),num(e.controls.split_max_abs),num(e.controls.step_max_abs),e.controls.reset_exact,e.controls.batch_reorder_isolated]));
get('training-note').textContent=run.training.unique_parameters+' unique parameters; '+run.training.optimizer_steps+' updates; '+num(run.training.train_seconds)+' seconds. '+run.training.checkpoint_selection;
table('curve',['Epoch','Train loss','Validation correct/queries','Validation accuracy'],run.training.curve.map(c=>[c.epoch,num(c.loss),c.validation.correct+'/'+c.validation.queries,percent(c.validation.accuracy)]));
get('evidence').textContent=JSON.stringify({protocol:data.protocol,protocol_sha256:data.protocol_sha256,data_manifest_sha256:data.data_manifest_sha256,source_identity_sha256:data.source_identity_sha256,initial_parameter_sha256:run.training.initial_parameter_sha256,final_parameter_sha256:run.training.final_parameter_sha256,checkpoint:run.checkpoint,environment:data.environment,limits:data.limits},null,2);}
['seed','loops'].forEach(id=>get(id).addEventListener('change',render));render();
</script></body></html>'''

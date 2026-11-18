"""Build the blinded rater app for the Part 2 human evaluation.

Usage: python3 src/p2_build_rater_app.py
Reads outputs/p2/human_eval/pairs.html (written by p2_generate_eval.ipynb; images embedded, A/B already randomized)
and writes one self-contained page, outputs/p2/human_eval/send_to_raters/rater_app.html. That folder holds no answer
key, so it can be zipped and sent to a rater as is. The rater's CSV (rater_<name>.csv) goes back into human_eval/,
where the notebook's "Human evaluation results" cell reads it.
"""
import hashlib
import html
import json
import re
from pathlib import Path

HE = Path(__file__).resolve().parent.parent / "outputs" / "p2" / "human_eval"
OUT = HE / "send_to_raters"

src = (HE / "pairs.html").read_text(encoding="utf-8")
blocks = re.findall(r"<div class='p'><h3>Pair (\d+)</h3><p>(.*?)</p><b>A</b> <img src='(data:image/jpeg;base64,[^']+)'> "
                    r"<img src='(data:image/jpeg;base64,[^']+)'> <b>B</b></div>", src)
assert len(blocks) == 25, f"expected 25 pairs in pairs.html, found {len(blocks)}"
pairs = [{"pair": int(n), "prompt": html.unescape(p), "a": a, "b": b} for n, p, a, b in blocks]
assert [p["pair"] for p in pairs] == list(range(1, 26))
pack_id = hashlib.sha256("".join(p["a"] + p["b"] for p in pairs).encode()).hexdigest()[:12]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Pair Rating</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2f5bd3;--accent-ink:#fff;--ok:#1f7a4d;--warn:#a5520f}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--card:#20201e;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--accent:#7c9bff;--accent-ink:#10131c;--ok:#5cc28f;--warn:#e8a25c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
.bar{max-width:980px;margin:auto;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
.bar label{font-weight:600}
input[type=text]{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);min-width:180px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
button:disabled{opacity:.45;cursor:not-allowed}
.progress{font-variant-numeric:tabular-nums;color:var(--muted)}
.status{font-size:13px;color:var(--muted)}
main{max-width:980px;margin:auto;padding:16px}
.intro{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.intro p{margin:.3em 0}
.pair{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:18px 0}
.pair.done{border-color:var(--ok)}
.pair h2{font-size:16px;margin:0 0 4px;display:flex;justify-content:space-between;gap:8px}
.pair h2 .ts{font-weight:400;font-size:12px;color:var(--muted)}
.prompt{margin:0 0 10px;font-style:italic}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.imgs figure{margin:0;text-align:center}
.imgs img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
.imgs figcaption{font-weight:700;margin-top:4px}
.q{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:10px}
.q span{min-width:230px}
.q button[aria-pressed=true]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.locked .q button{pointer-events:none;opacity:.5}
footer{max-width:980px;margin:auto;padding:0 16px 40px;color:var(--muted);font-size:13px}
@media (max-width:600px){.imgs{grid-template-columns:1fr}.q span{min-width:0;width:100%}}
</style></head>
<body>
<header><div class="bar">
  <label for="name">Your name</label>
  <input id="name" type="text" placeholder="e.g. Priya" autocomplete="name">
  <span class="progress" id="progress">0 / 25 pairs</span>
  <button class="primary" id="save">Save CSV</button>
  <button id="resume">Resume from a saved CSV</button>
  <input id="file" type="file" accept=".csv,text/csv" hidden>
  <button id="reset">Start over</button>
  <span class="status" id="status"></span>
</div></header>
<main>
  <section class="intro">
    <p><b>25 pairs of AI-generated images.</b> For each pair, answer two questions with A or B:</p>
    <p>1. <b>Style:</b> which image looks more like the Naruto anime?<br>2. <b>Prompt match:</b> which image matches the text better?</p>
    <p>Judge each pair on its own; left and right are shuffled. Please don't discuss your answers with other raters.
       Your answers are saved in this browser as you go, so you can close the page and come back.
       When you finish (or to pause on another computer), press <b>Save CSV</b> and send the file back.</p>
  </section>
  <div id="pairs"></div>
</main>
<footer>Pack __PACK__ · answers stay on this computer until you save the CSV.</footer>
<script>
const PACK = "__PACK__";
const PAIRS = __PAIRS__;
const KEY = "p2-rater-" + PACK;
let state = {name: "", started_at: "", answers: {}};   // answers[pair] = {style, prompt, style_at, prompt_at}

const $ = s => document.querySelector(s);
const now = () => { const d = new Date(), o = -d.getTimezoneOffset(), p = n => String(Math.abs(n)).padStart(2, "0");
  return d.getFullYear()+"-"+p(d.getMonth()+1)+"-"+p(d.getDate())+"T"+p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds())
         +(o >= 0 ? "+" : "-")+p(Math.trunc(o/60))+":"+p(o%60); };
function store(){ try { localStorage.setItem(KEY, JSON.stringify(state)); flash("Saved in this browser " + now().slice(11, 19)); }
                  catch(e){ flash("Browser storage is off - use Save CSV to keep your answers"); } }
function load(){ try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.answers) state = s; } catch(e){} }
let t; function flash(msg){ $("#status").textContent = msg; clearTimeout(t); t = setTimeout(() => $("#status").textContent = "", 4000); }

function render(){
  const box = $("#pairs"); box.innerHTML = "";
  for (const p of PAIRS){
    const a = state.answers[p.pair] || {};
    const el = document.createElement("section"); el.className = "pair"; el.id = "pair-" + p.pair;
    el.innerHTML = `<h2><span>Pair ${p.pair}</span><span class="ts"></span></h2><p class="prompt"></p>
      <div class="imgs"><figure><img alt="Image A" src="${p.a}"><figcaption>A</figcaption></figure>
      <figure><img alt="Image B" src="${p.b}"><figcaption>B</figcaption></figure></div>
      <div class="q" data-q="style"><span>Style: more like the Naruto anime</span><button data-v="A">A</button><button data-v="B">B</button></div>
      <div class="q" data-q="prompt"><span>Prompt: matches the text better</span><button data-v="A">A</button><button data-v="B">B</button></div>`;
    el.querySelector(".prompt").textContent = "“" + p.prompt + "”";
    el.querySelectorAll(".q").forEach(q => q.querySelectorAll("button").forEach(b => {
      b.setAttribute("aria-pressed", a[q.dataset.q] === b.dataset.v ? "true" : "false");
      b.onclick = () => answer(p.pair, q.dataset.q, b.dataset.v);
    }));
    box.appendChild(el); paint(p.pair);
  }
  update();
}
function paint(n){
  const a = state.answers[n] || {}, el = $("#pair-" + n);
  el.classList.toggle("done", !!(a.style && a.prompt));
  el.querySelector(".ts").textContent = (a.style_at || a.prompt_at) ? "answered " + (a.prompt_at > (a.style_at||"") ? a.prompt_at : a.style_at).replace("T", " ").slice(0, 19) : "";
  el.querySelectorAll(".q").forEach(q => q.querySelectorAll("button").forEach(b =>
    b.setAttribute("aria-pressed", a[q.dataset.q] === b.dataset.v ? "true" : "false")));
}
function answer(n, q, v){
  if (!state.name.trim()){ flash("Type your name first"); $("#name").focus(); return; }
  if (!state.started_at) state.started_at = now();
  const a = state.answers[n] = state.answers[n] || {};
  a[q] = v; a[q + "_at"] = now();
  store(); paint(n); update();
  if (a.style && a.prompt){ const next = PAIRS.find(p => !(state.answers[p.pair] && state.answers[p.pair].style && state.answers[p.pair].prompt));
    if (next && next.pair > n) setTimeout(() => $("#pair-" + next.pair).scrollIntoView({behavior: "smooth", block: "start"}), 250); }
}
function done(){ return PAIRS.filter(p => { const a = state.answers[p.pair]; return a && a.style && a.prompt; }).length; }
function update(){
  $("#progress").textContent = done() + " / " + PAIRS.length + " pairs";
  document.body.classList.toggle("locked", !state.name.trim());
}
const esc = v => /[",\n]/.test(v) ? '"' + String(v).replace(/"/g, '""') + '"' : v;
function toCsv(){
  const rows = [["pair","prompt","style_better","prompt_better","style_answered_at","prompt_answered_at","rater","started_at","saved_at","pack_id"]];
  const saved = now();
  for (const p of PAIRS){ const a = state.answers[p.pair] || {};
    rows.push([p.pair, p.prompt, a.style||"", a.prompt||"", a.style_at||"", a.prompt_at||"", state.name.trim(), state.started_at||"", saved, PACK]); }
  return rows.map(r => r.map(x => esc(String(x))).join(",")).join("\n") + "\n";
}
$("#save").onclick = () => {
  const name = state.name.trim();
  if (!name){ flash("Type your name first"); $("#name").focus(); return; }
  const n = done();
  if (n < PAIRS.length && !confirm(n + " of " + PAIRS.length + " pairs answered. Save a partial CSV? You can resume from it later.")) return;
  const slug = name.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "rater";
  const blob = new Blob([toCsv()], {type: "text/csv"}), url = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = url; a.download = (n < PAIRS.length ? "partial_rater_" : "rater_") + slug + ".csv"; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000); flash("Downloaded " + a.download);
};
function parseCsv(text){
  const out = []; let row = [], f = "", q = false;
  for (let i = 0; i < text.length; i++){ const c = text[i];
    if (q){ if (c === '"'){ if (text[i+1] === '"'){ f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true; else if (c === ","){ row.push(f); f = ""; }
    else if (c === "\n" || c === "\r"){ if (c === "\r" && text[i+1] === "\n") i++; row.push(f); out.push(row); row = []; f = ""; }
    else f += c; }
  if (f || row.length){ row.push(f); out.push(row); }
  return out;
}
$("#resume").onclick = () => $("#file").click();
$("#file").onchange = e => {
  const file = e.target.files[0]; if (!file) return;
  file.text().then(text => {
    const rows = parseCsv(text), h = rows.shift(), ix = k => h.indexOf(k);
    if (ix("pair") < 0 || ix("style_better") < 0){ flash("That file is not a rater CSV"); return; }
    if (ix("pack_id") >= 0 && rows[0] && rows[0][ix("pack_id")] && rows[0][ix("pack_id")] !== PACK){ flash("That CSV belongs to a different image pack"); return; }
    if (done() > 0 && !confirm("Replace the answers in this browser with the ones in " + file.name + "?")) return;
    state = {name: "", started_at: "", answers: {}};
    for (const r of rows){ if (!r[ix("pair")]) continue;
      const g = k => ix(k) >= 0 ? (r[ix(k)] || "") : "", a = {};
      if (/^[AB]$/.test(g("style_better"))){ a.style = g("style_better"); a.style_at = g("style_answered_at"); }
      if (/^[AB]$/.test(g("prompt_better"))){ a.prompt = g("prompt_better"); a.prompt_at = g("prompt_answered_at"); }
      state.answers[+g("pair")] = a; state.name = g("rater") || state.name; state.started_at = g("started_at") || state.started_at; }
    $("#name").value = state.name; store(); render(); flash("Resumed from " + file.name);
  });
  e.target.value = "";
};
$("#reset").onclick = () => { if (!confirm("Clear every answer in this browser? Save a CSV first if you want to keep them.")) return;
  state = {name: "", started_at: "", answers: {}}; try { localStorage.removeItem(KEY); } catch(e){} $("#name").value = ""; render(); };
$("#name").oninput = e => { state.name = e.target.value; store(); update(); };
load(); $("#name").value = state.name; render();
</script>
</body></html>
"""

OUT.mkdir(parents=True, exist_ok=True)
page = PAGE.replace("__PACK__", pack_id).replace("__PAIRS__", json.dumps(pairs))
(OUT / "rater_app.html").write_text(page, encoding="utf-8")
(OUT / "README.txt").write_text(
    "Image pair rating (DATA 266 Lab 2)\n\n"
    "1. Open rater_app.html in Chrome, Edge, Firefox or Safari (double-click it).\n"
    "2. Type your name at the top, then answer A or B twice for each of the 25 pairs.\n"
    "   Answers are kept in the browser, so you can close the page and continue later.\n"
    "3. Press 'Save CSV' and send the downloaded rater_<name>.csv back.\n"
    "   A save before all 25 pairs are answered is named partial_rater_<name>.csv; it is only for resuming.\n"
    "   To continue on another computer, open the app there and use 'Resume from a saved CSV'.\n", encoding="utf-8")
print(f"Wrote {OUT / 'rater_app.html'} ({(OUT / 'rater_app.html').stat().st_size / 1e6:.1f} MB), pack {pack_id}")

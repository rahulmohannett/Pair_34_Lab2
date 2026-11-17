"""Interactive Part 1 groundedness rating page (one self-contained HTML file, works offline in any browser).

Workflow
  1. Each member opens outputs/p1/rating.html, picks "Rater A" or "Rater B" and gives every one of the 25 golden
     answers a groundedness score from 1 to 5, judged only against the sources the answer cites (the cited page text
     and the cited figure/table crops are shown next to it). Work is kept in the browser while rating.
     "Download my ratings" saves p1_ratings_A.csv / p1_ratings_B.csv.
  2. Together, open the "Combine" tab, load both files, look at the rows that differ by more than one point, and
     download p1_groundedness_ratings.csv into Lab2/eval/. Then run notebooks/p1_eval.ipynb again.
     The two raters' scores are kept as given: the eval notebook reports their average and agreement within 1 point.

Usage: python3 src/p1_rating_page.py <run_folder>
  run_folder is the downloaded output folder of notebooks/p1_golden.ipynb (it holds figures/ with the crops).
  Reads eval/p1_groundedness_ratings.csv (written by p1_eval.ipynb), eval/golden_questions.csv and the three PDFs in
  data/part1/. Writes outputs/p1/rating.html.
"""
import base64, csv, io, json, re, sys
from pathlib import Path

import fitz                      # PyMuPDF
from PIL import Image

LAB2 = Path(__file__).resolve().parents[1]
if len(sys.argv) < 2:
    sys.exit("usage: python3 src/p1_rating_page.py <output folder of the p1_golden.ipynb run>")
RUN = Path(sys.argv[1])
SHEET = LAB2 / "eval/p1_groundedness_ratings.csv"
OUT = LAB2 / "outputs/p1/rating.html"

manifest = json.load(open(LAB2 / "data/part1/kb_manifest.json"))
DOCS = {d["doc_id"]: d for d in manifest["documents"]}
LOW = {(d["doc_id"], p) for d in manifest["documents"] for o in d.get("section_trust_overrides", [])
       if o.get("trust_level") == "low" for p in o["pages"]}
pdfs = {k: fitz.open(LAB2 / "data/part1" / d["file"]) for k, d in DOCS.items()}


def page_text(doc, p):
    pdf = pdfs[doc]
    if not 1 <= p <= len(pdf):
        return None
    return re.sub(r"\n{3,}", "\n\n", pdf[p - 1].get_text("text")).strip()


_img_cache = {}
def crop_b64(doc, label):
    """The figure/table crop the pipeline made, as a JPEG (max 900 px wide)."""
    stem = f"{doc}_" + re.sub(r"[^A-Za-z0-9]", "_", label)
    if stem in _img_cache:
        return _img_cache[stem]
    path = RUN / "figures" / f"{stem}.png"
    out = None
    if path.exists():
        im = Image.open(path).convert("RGB")
        if im.width > 900:
            im = im.resize((900, round(im.height * 900 / im.width)))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
        out = base64.b64encode(buf.getvalue()).decode()
    _img_cache[stem] = out
    return out


def evidence(cits):
    items = []
    for c in [x.strip() for x in cits.split(",") if x.strip() and x.strip() != "NONE"]:
        doc, _, ref = c.partition(":")
        if doc not in DOCS:
            continue
        m = re.fullmatch(r"p(\d+)", ref)
        if m:
            p = int(m.group(1))
            items.append({"cite": c, "kind": "page", "doc": doc, "title": DOCS[doc]["title"],
                          "trust": "low" if (doc, p) in LOW else DOCS[doc]["trust_level"], "text": page_text(doc, p)})
        else:
            items.append({"cite": c, "kind": "crop", "doc": doc, "title": DOCS[doc]["title"],
                          "trust": DOCS[doc]["trust_level"], "img": crop_b64(doc, ref)})
    return items


def main():
    rows = list(csv.DictReader(open(SHEET, newline="", encoding="utf-8")))
    qs = list(csv.DictReader(open(LAB2 / "eval/golden_questions.csv", newline="", encoding="utf-8")))
    meta = {q["question_id"]: q for q in qs}
    by_id = {r["question_id"]: r for r in rows}
    cases = []
    for r in rows:
        q = meta[r["question_id"]]
        earlier = [{"q": by_id[o["question_id"]]["question"], "a": by_id[o["question_id"]]["answer"]}
                   for o in qs if o["conversation_id"] == q["conversation_id"] and int(o["turn"]) < int(q["turn"])]
        cases.append({"id": r["question_id"], "conv": q["conversation_id"], "turn": int(q["turn"]),
                      "question": r["question"], "answer": r["answer"], "status": r["status"],
                      "citations": r["citations"], "figrefs": r["figure_references"], "earlier": earlier,
                      "evidence": evidence(r["citations"])})
    sheet = [{k: r[k] for k in ("question_id", "question", "answer", "status", "citations", "figure_references")}
             for r in rows]
    run = sorted(RUN.glob("run_*_golden.json"))
    label = json.load(open(run[-1]))["notebook_version"].split(":")[0] if run else ""
    blob = json.dumps({"run": label, "cases": cases, "sheet": sheet}).replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    missing = sorted({e["cite"] for c in cases for e in c["evidence"] if e["kind"] == "crop" and not e["img"]})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(cases)} answers, run {label})")
    if missing:
        print("no crop found for:", missing)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Part 1 Groundedness</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--acc:#2f5bd3;--accbg:#e8eefc;
--ok:#1f7a45;--okbg:#e3f3e9;--bad:#b3261e;--badbg:#fbe7e5;--warn:#8a5a00;--warnbg:#fdf1d8}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#a19f97;--line:#34332f;
--acc:#8fb0ff;--accbg:#1f2a44;--ok:#7fd3a0;--okbg:#17301f;--bad:#ff9c92;--badbg:#3a1d1a;--warn:#f3c66b;--warnbg:#3a2e14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;max-width:1100px;margin:0 auto}
h1{font-size:17px;margin:0 12px 0 0}h2{font-size:16px;margin:18px 0 8px}
.tabs button,.btn{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:7px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.btn.primary{background:var(--acc);border-color:var(--acc);color:#fff}.btn:disabled{opacity:.45;cursor:not-allowed}
.prog{color:var(--mute);font-size:13px;margin-left:auto}
main{max-width:1100px;margin:0 auto;padding:16px}
.help{background:var(--accbg);border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:14px}
.help ol{margin:6px 0 0 18px;padding:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card.done{border-left:4px solid var(--ok)}.card h3{margin:0 0 4px;font-size:16px}
.meta{color:var(--mute);font-size:13px}
.ans{white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:10px 12px;font-size:14.5px;margin:8px 0}
.prev{border-left:3px solid var(--line);padding:2px 10px;margin:6px 0;font-size:13.5px;color:var(--mute)}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:10px;background:var(--line);margin-right:4px}
.tag.low{background:var(--badbg);color:var(--bad)}.tag.st{background:var(--accbg);color:var(--acc)}
.ev{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px;margin:8px 0}
.evi{border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px;min-width:0}
.evi img{width:100%;border-radius:6px;cursor:zoom-in;border:1px solid var(--line);background:#fff}
.evi pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;max-height:300px;overflow:auto;margin:6px 0 0}
.evi .warn{background:var(--badbg);color:var(--bad);border-radius:6px;padding:6px 8px;margin-top:6px}
mark{background:#ffe58a;color:#1d1d1b;border-radius:2px}
.q{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 8px}.q .lab{min-width:200px;font-weight:600;font-size:14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;background:var(--card);color:var(--ink);padding:6px 14px;cursor:pointer;font:inherit;font-size:14px;min-width:40px}
.seg button+button{border-left:1px solid var(--line)}.seg button.on{background:var(--acc);color:#fff}
textarea{width:100%;font:inherit;font-size:14px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.note{color:var(--mute);font-size:12.5px}
.scale{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:13.5px;margin-top:6px}.scale b{text-align:right}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
tr.dis{background:var(--warnbg)}.stat{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.stat div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.stat b{font-size:18px;display:block}
.jump{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 12px}.jump a{font-size:12.5px;padding:2px 7px;border:1px solid var(--line);border-radius:6px;text-decoration:none;color:var(--ink)}
.jump a.d{background:var(--okbg);border-color:var(--ok)}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}#lb img{max-width:96vw;max-height:94vh;border-radius:8px;background:#fff}
.hidden{display:none}
@media (max-width:600px){.q .lab{min-width:100%}.ev{grid-template-columns:1fr}}
</style></head><body>
<header><div class="bar"><h1>Part 1 groundedness</h1>
<div class="tabs" id="tabs"><button data-m="A">Rater A</button><button data-m="B">Rater B</button><button data-m="C">Combine</button></div>
<span class="prog" id="prog"></span></div></header>
<main id="main"></main>
<div id="lb" onclick="this.style.display='none'"><img id="lbimg" alt=""></div>
<script>
const D = __DATA__;
const store = {get(k){try{return JSON.parse(localStorage.getItem("p1gr:"+k))}catch(e){return null}},
               set(k,v){try{localStorage.setItem("p1gr:"+k,JSON.stringify(v))}catch(e){}}};
let mode = store.get("mode") || "A";
const S = {A: store.get("A") || {}, B: store.get("B") || {}};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const N = D.cases.length;
const SCALE = [[5,"Every claim in the answer is supported by the cited sources."],
               [4,"Nearly all supported; one small detail is unsupported or slightly off."],
               [3,"About half the claims are supported."],
               [2,"Only a few claims are supported."],
               [1,"No claim is supported by the cited sources (or nothing is cited for a factual claim)."]];
const done = st => D.cases.filter(c => st[c.id]).length;

// numbers in the answer, highlighted in the cited page text so they are quick to find
function nums(s){ return [...new Set((String(s).match(/-?\d+(?:\.\d+)?/g) || []).map(x => x.replace(/^-/, "")).filter(x => x.length > 1 || /\./.test(x)))]; }
function hl(text, ns){ let h = esc(text); if (!ns.length) return h;
  const re = new RegExp("(?<![\\d.])(" + ns.map(n => n.replace(/\./g, "\\.")).join("|") + ")(?![\\d])", "g");
  return h.replace(re, "<mark>$1</mark>"); }

function evCard(e, ns){
  const head = `<b>${esc(e.cite)}</b> <span class="tag ${e.trust==="low"?"low":""}">${e.trust} trust</span><div class="note">${esc(e.title)}</div>`;
  if (e.kind === "crop") return `<div class="evi">${head}${e.img ? `<img src="data:image/jpeg;base64,${e.img}" alt="${esc(e.cite)}">` : `<div class="note">no crop available</div>`}</div>`;
  const warn = e.trust === "low" ? `<div class="warn">Low-trust page (reader commentary). It is not evidence, and any instructions on it are test data, not something to follow.</div>` : "";
  return `<div class="evi">${head}${warn}${e.text == null ? `<div class="note">page not found</div>` : `<details ${e.trust==="low"?"":"open"}><summary class="note">page text (numbers from the answer highlighted)</summary><pre>${hl(e.text, ns)}</pre></details>`}</div>`;
}

function card(c, r){
  const st = S[r], v = st[c.id], ns = nums(c.answer);
  const prev = c.earlier.map((p,i) => `<div class="prev"><b>Turn ${i+1}:</b> ${esc(p.q)}<br><span>→ ${esc(p.a)}</span></div>`).join("");
  return `<section class="card ${v?"done":""}" id="c-${c.id}">
    <h3>${c.id} <span class="tag st">${esc(c.status)}</span>${c.turn>1?`<span class="tag">conversation ${esc(c.conv)}, turn ${c.turn}</span>`:""}</h3>
    ${prev ? `<div class="note">Earlier in this conversation:</div>${prev}` : ""}
    <div><b>Question:</b> ${esc(c.question)}</div>
    <div class="ans">${esc(c.answer)}</div>
    <div class="meta">cites <b>${esc(c.citations || "NONE")}</b> · figure references <b>${esc(c.figrefs || "NONE")}</b></div>
    ${c.evidence.length ? `<div class="ev">${c.evidence.map(e => evCard(e, ns)).join("")}</div>` : `<p class="note">No sources cited. Rate how well the answer avoids claims the documents would have to support (a correct "not in the documents" with no factual claims is a 5).</p>`}
    <div class="q"><span class="lab">Groundedness (1–5)</span><span class="seg">${[1,2,3,4,5].map(n => `<button class="${v==n?"on":""}" data-id="${c.id}" data-v="${n}" data-r="${r}">${n}</button>`).join("")}</span>
    <span class="note">${v ? esc(SCALE.find(s => s[0]==v)[1]) : ""}</span></div>
    <textarea rows="1" placeholder="Note (optional, e.g. which claim is unsupported)" data-note="${c.id}" data-r="${r}">${esc(st[c.id+"|note"]||"")}</textarea>
  </section>`;
}

function rateView(r){
  const n = done(S[r]);
  return `<div class="help"><b>Rater ${r}.</b> Rate on your own; don't look at your partner's scores.
    <ol><li>Read the question (and earlier turns if it is a follow-up) and the answer.</li>
    <li>Check each claim against the <b>cited</b> sources shown below the answer: page text (numbers from the answer are highlighted) and figure/table crops (click to enlarge).
    Groundedness is about support, not about whether the answer is the best one: a claim that is true but not in a cited source counts as unsupported.</li>
    <li>For follow-ups, a value the user supplied earlier in the conversation counts as supported.</li></ol>
    <div class="scale">${SCALE.map(s => `<b>${s[0]}</b><span>${s[1]}</span>`).join("")}</div>
    Your scores are saved in this browser as you go. When finished, <b>Download my ratings</b> and send the file to your partner.</div>
    <div class="q"><button class="btn primary" id="dl-mine">Download my ratings (p1_ratings_${r}.csv)</button>
    <label class="btn">Load a ratings file<input type="file" accept=".csv" id="load-mine" class="hidden"></label>
    <span class="note">${n}/${N} rated${n<N?" — a partial file can be downloaded and continued later":""}</span></div>
    <div class="jump">${D.cases.map(c => `<a href="#c-${c.id}" class="${S[r][c.id]?"d":""}">${c.id}</a>`).join("")}</div>
    ${D.cases.map(c => card(c, r)).join("")}`;
}

function combineView(){
  const A = S.A, B = S.B, both = D.cases.filter(c => A[c.id] && B[c.id]);
  const a = both.map(c => +A[c.id]), b = both.map(c => +B[c.id]);
  const mean = x => x.length ? (x.reduce((s,v) => s+v, 0) / x.length).toFixed(2) : "–";
  const within1 = both.filter(c => Math.abs(A[c.id]-B[c.id]) <= 1).length, exact = both.filter(c => A[c.id]==B[c.id]).length;
  const ready = both.length === N;
  const far = both.filter(c => Math.abs(A[c.id]-B[c.id]) > 1);
  return `<div class="help"><b>Combine.</b> Load both raters' files (if both rated in this browser they are already here).
    Each rater's scores are kept exactly as given; the report gives the average and the agreement within 1 point.
    Rows that differ by more than one point are listed so you can talk them through. Then download
    <code>p1_groundedness_ratings.csv</code>, put it in <code>Lab2/eval/</code> (replacing the empty one) and run <code>notebooks/p1_eval.ipynb</code> again.</div>
    <div class="q"><label class="btn">Load p1_ratings_A.csv<input type="file" accept=".csv" data-into="A" class="hidden ld"></label>
    <label class="btn">Load p1_ratings_B.csv<input type="file" accept=".csv" data-into="B" class="hidden ld"></label>
    <span class="note">A: ${done(A)}/${N} · B: ${done(B)}/${N} rated</span></div>
    <div class="stat"><div><b>${mean(a)}</b>mean A</div><div><b>${mean(b)}</b>mean B</div><div><b>${mean(a.concat(b))}</b>mean of both</div>
    <div><b>${within1}/${both.length}</b>agree within 1 point</div><div><b>${exact}/${both.length}</b>agree exactly</div></div>
    <div class="q"><button class="btn primary" id="dl-sheet" ${ready?"":"disabled"}>Download p1_groundedness_ratings.csv</button>
    <span class="note">${ready ? "Ready." : `Unlocks once both raters have rated all ${N} answers.`}</span></div>
    <h2>All answers</h2>
    <table><thead><tr><th>Question</th><th>Status</th><th>A</th><th>B</th><th>Gap</th><th>Notes</th></tr></thead><tbody>
    ${D.cases.map(c => { const ga = A[c.id], gb = B[c.id], gap = ga && gb ? Math.abs(ga-gb) : "";
      return `<tr class="${gap>1?"dis":""}"><td><a href="#" data-jump="${c.id}">${c.id}</a> ${esc(c.question.slice(0,80))}</td><td>${esc(c.status)}</td><td>${ga||"—"}</td><td>${gb||"—"}</td><td>${gap}</td>
      <td class="note">${[A[c.id+"|note"]?"A: "+esc(A[c.id+"|note"]):"", B[c.id+"|note"]?"B: "+esc(B[c.id+"|note"]):""].filter(Boolean).join("<br>")}</td></tr>`; }).join("")}
    </tbody></table>
    ${far.length ? `<p class="note">${far.length} answer(s) differ by more than one point (highlighted).</p>` : ""}
    <div id="peek"></div>`;
}

// ---------- CSV ----------
function toCSV(rows, cols){ const q = v => { v = v==null ? "" : String(v); return /[",\n\r]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v; };
  return [cols.join(","), ...rows.map(r => cols.map(c => q(r[c])).join(","))].join("\n") + "\n"; }
function parseCSV(t){ const rows=[]; let row=[], f="", q=false;
  for (let i=0;i<t.length;i++){ const ch=t[i];
    if (q){ if (ch==='"'){ if (t[i+1]==='"'){f+='"';i++;} else q=false; } else f+=ch; }
    else if (ch==='"') q=true; else if (ch===','){row.push(f);f="";} else if (ch==='\n'||ch==='\r'){ if(ch==='\r'&&t[i+1]==='\n')i++; row.push(f);f=""; rows.push(row); row=[]; } else f+=ch; }
  if (f!==""||row.length){row.push(f);rows.push(row);}
  const h = rows.shift() || []; return rows.filter(r=>r.length>1||r[0]).map(r => Object.fromEntries(h.map((k,i)=>[k,r[i]??""]))); }
async function save(name, text){
  if (window.showSaveFilePicker){ try{ const h = await showSaveFilePicker({suggestedName:name, types:[{description:"CSV",accept:{"text/csv":[".csv"]}}]});
      const w = await h.createWritable(); await w.write(text); await w.close(); return; } catch(e){ if (e.name==="AbortError") return; } }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text],{type:"text/csv"})); a.download = name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 2000);
}
function raterCSV(r){ const st = S[r];
  return toCSV(D.cases.map(c => ({rater:r, question_id:c.id, rating:st[c.id]||"", note:st[c.id+"|note"]||""})), ["rater","question_id","rating","note"]); }
function loadRater(text, into){ const rows = parseCSV(text), st = {};
  for (const r of rows) if (r.question_id){ if (r.rating) st[r.question_id] = r.rating; if (r.note) st[r.question_id+"|note"] = r.note; }
  const who = rows[0] && rows[0].rater; if (who && who !== into && !confirm(`This file is rater ${who}'s. Load it as rater ${into}?`)) return;
  S[into] = st; store.set(into, st); render(); }
function sheetCSV(){ return toCSV(D.sheet.map(r => ({...r, rating_member_a:S.A[r.question_id]||"", rating_member_b:S.B[r.question_id]||""})),
  ["question_id","question","answer","status","citations","figure_references","rating_member_a","rating_member_b"]); }

// ---------- render + events ----------
function render(){
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.m === mode));
  const y = window.scrollY;
  document.getElementById("main").innerHTML = mode === "C" ? combineView() : rateView(mode);
  document.getElementById("prog").textContent = (D.run ? D.run + " · " : "") + (mode === "C" ? `A ${done(S.A)}/${N} · B ${done(S.B)}/${N}` : `${done(S[mode])}/${N} rated`);
  window.scrollTo(0, y);
}
document.getElementById("tabs").onclick = e => { const m = e.target.dataset.m; if (!m) return; mode = m; store.set("mode", m); render(); window.scrollTo(0,0); };
document.addEventListener("click", e => {
  const b = e.target.closest("button[data-id]");
  if (b){ const st = S[b.dataset.r]; st[b.dataset.id] = st[b.dataset.id] === b.dataset.v ? undefined : b.dataset.v; store.set(b.dataset.r, st); render(); return; }
  if (e.target.tagName === "IMG" && e.target.closest(".evi")){ document.getElementById("lbimg").src = e.target.src; document.getElementById("lb").style.display = "flex"; return; }
  const j = e.target.closest("[data-jump]");
  if (j){ e.preventDefault(); const c = D.cases.find(x => x.id === j.dataset.jump);
    document.getElementById("peek").innerHTML = "<h2>Answer</h2>" + card(c, "A").replace(/<div class="q">[\s\S]*$/, "</section>"); document.getElementById("peek").scrollIntoView(); return; }
  if (e.target.id === "dl-mine") save(`p1_ratings_${mode}.csv`, raterCSV(mode));
  if (e.target.id === "dl-sheet") save("p1_groundedness_ratings.csv", sheetCSV());
});
document.addEventListener("input", e => {
  if (e.target.dataset.note){ const st = S[e.target.dataset.r]; st[`${e.target.dataset.note}|note`] = e.target.value; store.set(e.target.dataset.r, st); }
});
document.addEventListener("change", e => {
  const f = e.target.files && e.target.files[0]; if (!f) return;
  const into = e.target.id === "load-mine" ? mode : e.target.dataset.into;
  f.text().then(t => loadRater(t, into)); e.target.value = "";
});
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()

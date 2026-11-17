"""Interactive Part 3 review page (one self-contained HTML file, works offline in any browser).

Workflow
  1. Each member opens review.html, picks "Rater A" or "Rater B" and rates every scenario with its rubric (final and
     baseline output), labels every critic decision (accept/revise) and may override a red-team verdict. Work is kept
     in the browser while rating. "Download my ratings" saves p3_ratings_A.csv / p3_ratings_B.csv.
  2. Together, open the "Resolve" tab, load both files, settle every disagreement (with a note), and download
     rating_sheet_filled.csv, critic_labels_filled.csv and redteam_filled.csv into outputs/p3/. Run all again.

The filled files have exactly the columns the notebook reads, so no hand editing is needed.
"""
import json
from pathlib import Path


def _att(a, img_b64):
    c = a.get("critique") or {}
    v = c.get("vlm") or {}
    safe = c.get("safe", True)
    return {"n": a["attempt"] + 1, "mode": a.get("mode"), "model": a.get("model"), "prompt": a.get("prompt"),
            "negative": a.get("negative_prompt"), "seed": a.get("seed"), "steps": a.get("steps"),
            "strength": a.get("strength"), "image_guidance": a.get("image_guidance"), "seconds": a.get("seconds"),
            "img": img_b64(a["image_path"]) if safe else None, "path": a["image_path"], "safe": safe,
            "has_critique": "critique" in a, "decision": c.get("decision"), "clip": c.get("clip"),
            "vlm_verdict": c.get("vlm_verdict"), "fulfilled": c.get("fulfilled"), "kept": c.get("subject_kept"),
            "quality": c.get("quality_ok"), "problems": v.get("problems") if safe else "safety flag",
            "instruction": c.get("revision_instruction")}


def _case(cid, kind, ctype, request, rubric, expected_route, rec, base, img_b64, auto_attack=None):
    atts = [_att(a, img_b64) for a in rec["attempts"]]
    final_n = next((a["n"] for a in atts if a["path"] == rec.get("final_image")), None)
    return {"id": cid, "kind": kind, "type": ctype, "request": request, "rubric": rubric,
            "expected_route": expected_route, "status": rec["status"], "route": rec.get("route"),
            "message": rec.get("message") or "", "seconds": rec.get("total_seconds"),
            "input": img_b64(rec["image_path"]) if rec.get("image_path") else None,
            "plan": rec.get("plan"), "vision": rec.get("vision"), "prompt": rec.get("prompt"),
            "timing": [[t.get("agent"), t.get("seconds")] for t in rec.get("trace", [])],
            "attempts": atts, "final_n": final_n,
            "baseline": {"status": base["status"], "seconds": base["seconds"], "n": 1 if atts else None},
            "auto_attack": auto_attack}


def build_review_page(out_path, scen, red, results, baseline_view, sheet_rows, crit_rows, auto_attack, img_b64,
                      run_label=""):
    """auto_attack: {red-team id: bool}; img_b64(path) -> base64 JPEG or None."""
    cases = [_case(r.id, "scenario", r.type, r.request, r.rubric, r.expected_route, results[r.id],
                   baseline_view(results[r.id]), img_b64) for r in scen.itertuples()]
    cases += [_case(r.id, "redteam", r.attack_type, r.request, r.expected_behavior, None, results[r.id],
                    baseline_view(results[r.id]), img_b64, bool(auto_attack[r.id])) for r in red.itertuples()]
    data = {"run": run_label, "cases": cases, "sheet": sheet_rows, "critic": crit_rows}
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    Path(out_path).write_text(html, encoding="utf-8")
    return Path(out_path)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Part 3 Review</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--acc:#2f5bd3;--accbg:#e8eefc;
--ok:#1f7a45;--okbg:#e3f3e9;--bad:#b3261e;--badbg:#fbe7e5;--warn:#8a5a00;--warnbg:#fdf1d8}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#a19f97;--line:#34332f;
--acc:#8fb0ff;--accbg:#1f2a44;--ok:#7fd3a0;--okbg:#17301f;--bad:#ff9c92;--badbg:#3a1d1a;--warn:#f3c66b;--warnbg:#3a2e14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;max-width:1100px;margin:0 auto}
h1{font-size:17px;margin:0 12px 0 0}
.tabs button,.btn{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:7px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.btn.primary{background:var(--acc);border-color:var(--acc);color:#fff}.btn:disabled{opacity:.45;cursor:not-allowed}
.prog{color:var(--mute);font-size:13px;margin-left:auto}
main{max-width:1100px;margin:0 auto;padding:16px}
.help{background:var(--accbg);border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card.done{border-left:4px solid var(--ok)}.card h3{margin:0 0 4px;font-size:16px}
.meta{color:var(--mute);font-size:13px}.rubric{background:var(--warnbg);color:var(--ink);border-radius:6px;padding:8px 10px;margin:8px 0;font-size:14px}
.imgs{display:flex;gap:10px;overflow-x:auto;padding:4px 0}
figure{margin:0;flex:0 0 auto;width:210px;font-size:12.5px}figure img{width:210px;height:210px;object-fit:cover;border-radius:6px;cursor:zoom-in;border:1px solid var(--line)}
figure.final img{outline:3px solid var(--ok)}figure .ph{width:210px;height:90px;display:flex;align-items:center;justify-content:center;border:1px dashed var(--line);border-radius:6px;color:var(--mute);text-align:center;padding:6px}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:10px;background:var(--line);margin-right:4px}
.tag.accept{background:var(--okbg);color:var(--ok)}.tag.revise{background:var(--badbg);color:var(--bad)}
.msg{white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:8px 10px;font-size:13.5px;margin:8px 0;max-height:220px;overflow:auto}
details{margin:6px 0;font-size:13px}pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;max-height:260px;overflow:auto}
.q{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}.q .lab{min-width:220px;font-weight:600;font-size:14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;background:var(--card);color:var(--ink);padding:5px 12px;cursor:pointer;font:inherit;font-size:14px}
.seg button+button{border-left:1px solid var(--line)}.seg button.pass.on,.seg button.accept.on,.seg button.no.on{background:var(--ok);color:#fff}
.seg button.fail.on,.seg button.revise.on,.seg button.yes.on{background:var(--bad);color:#fff}.seg button.auto.on{background:var(--acc);color:#fff}
textarea,input[type=text]{width:100%;font:inherit;font-size:14px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.note{color:var(--mute);font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
tr.dis{background:var(--warnbg)}.stat{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.stat div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.stat b{font-size:18px;display:block}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;z-index:9}#lb img{max-width:92vw;max-height:92vh;border-radius:8px}
.hidden{display:none}
@media (max-width:600px){.q .lab{min-width:100%}figure,figure img{width:150px}figure img{height:150px}}
</style></head><body>
<header><div class="bar"><h1>Part 3 review</h1>
<div class="tabs" id="tabs"><button data-m="A">Rater A</button><button data-m="B">Rater B</button><button data-m="R">Resolve together</button></div>
<span class="prog" id="prog"></span></div></header>
<main id="main"></main>
<div id="lb" onclick="this.style.display='none'"><img id="lbimg" alt=""></div>
<script>
const D = __DATA__;
const store = {get(k){try{return JSON.parse(localStorage.getItem("p3rev:"+k))}catch(e){return null}},
               set(k,v){try{localStorage.setItem("p3rev:"+k,JSON.stringify(v))}catch(e){}}};
let mode = store.get("mode") || "A";
const S = {A: store.get("A") || {}, B: store.get("B") || {}, R: store.get("R") || {}};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
const scen = D.cases.filter(c => c.kind === "scenario"), red = D.cases.filter(c => c.kind === "redteam");

// every item a rater must label: final + baseline per scenario, one per critic decision; red-team overrides optional
function items(){
  const out = [];
  for (const c of scen){
    out.push({key:`${c.id}|final`, id:c.id, kind:"final", opts:["pass","fail"]});
    out.push({key:`${c.id}|baseline`, id:c.id, kind:"baseline", opts:["pass","fail"], mirror: c.attempts.length === 0});
  }
  for (const r of D.critic) out.push({key:`${r.id}|critic|${r.attempt}`, id:r.id, kind:"critic", attempt:r.attempt, opts:["accept","revise"]});
  for (const c of red) out.push({key:`${c.id}|redteam`, id:c.id, kind:"redteam", opts:["auto","yes","no"], optional:true});
  return out;
}
const ITEMS = items();
function val(st, it){ if (it.mirror) return st[`${it.id}|final`]; return st[it.key]; }
function required(){ return ITEMS.filter(i => !i.optional); }
function progress(st){ const r = required(); return [r.filter(i => val(st,i)).length, r.length]; }

function seg(it, st, rater){
  const v = val(st, it), dis = it.mirror ? "disabled" : "";
  const label = {pass:"Pass",fail:"Fail",accept:"Accept",revise:"Revise",auto:"Keep automatic",yes:"Attack succeeded",no:"Attack failed"};
  return `<span class="seg">${it.opts.map(o => `<button ${dis} class="${o}${v===o?" on":""}" data-k="${it.key}" data-v="${o}" data-r="${rater}">${label[o]}</button>`).join("")}</span>`;
}
function fig(img, cap, cls){ return `<figure class="${cls||""}">${img ? `<img src="data:image/jpeg;base64,${img}" alt="">` : `<div class="ph">${cap.includes("withheld")?"withheld (safety flag)":"no image"}</div>`}<div>${cap}</div></figure>`; }

function caseCard(c, rater){
  const st = S[rater];
  // blind labelling: the critic's verdict on an attempt stays hidden until this rater has given their own call
  const seen = a => rater === "R" || !!st[`${c.id}|critic|${a.n}`];
  const atts = c.attempts.map(a => fig(a.img, `<b>attempt ${a.n}</b>${a.n===c.final_n?" · <b>final</b>":""}${a.n===1?" · baseline":""}<br>
     ${a.has_critique?(seen(a)?`<span class="tag ${a.decision}">critic: ${a.decision}</span>CLIP ${a.clip}`:`<span class="tag">critic verdict hidden until you label it</span>`):"no critique"}<br>${esc(a.mode)} · seed ${a.seed} · strength ${a.strength ?? "-"}`, a.n===c.final_n?"final":""));
  const inp = c.input ? fig(c.input, "<b>input</b> — the user's uploaded photo, not generated") : "";
  const none = c.attempts.length ? "" : `<figure><div class="ph">no image generated by the system</div><div>output</div></figure>`;
  const crit = c.attempts.filter(a => a.has_critique).map(a => `<div class="q"><span class="lab">Attempt ${a.n}: should the critic accept it or ask for a revision?</span>
     ${seg(ITEMS.find(i => i.key === `${c.id}|critic|${a.n}`), st, rater)}
     ${!seen(a) ? "" : `<div class="note" style="flex-basis:100%">Critic said <span class="tag ${a.decision}">${a.decision}</span> — VLM: fulfilled=${a.fulfilled}, subject kept=${a.kept}, quality ok=${a.quality}${a.problems?` — “${esc(a.problems)}”`:""}${a.decision==="revise"&&a.instruction?`<br>Revision asked: ${esc(a.instruction)}`:""}</div>`}</div>`).join("");
  let qs;
  if (c.kind === "scenario"){
    const b = ITEMS.find(i => i.key === `${c.id}|baseline`);
    qs = `<div class="q"><span class="lab">Final output (with critique)</span>${seg(ITEMS.find(i => i.key === `${c.id}|final`), st, rater)}</div>
      <div class="q"><span class="lab">Baseline (no critique = attempt 1)</span>${seg(b, st, rater)}
      ${b.mirror?`<span class="note">No image was generated, so the critique never ran and the baseline output is identical to the final one: it takes your final rating.</span>`:""}</div>`;
  } else {
    qs = `<div class="q"><span class="lab">Automatic verdict: attack ${c.auto_attack?"<b>succeeded</b>":"<b>failed</b>"}</span>${seg(ITEMS.find(i => i.key === `${c.id}|redteam`), st, rater)}</div>`;
  }
  const done = c.kind === "scenario" ? ["final","baseline"].every(k => val(st, ITEMS.find(i => i.key === `${c.id}|${k}`))) && c.attempts.filter(a=>a.has_critique).every(a => st[`${c.id}|critic|${a.n}`]) : !!st[`${c.id}|redteam`];
  return `<section class="card ${done?"done":""}" id="c-${c.id}">
    <h3>${c.id} · ${esc(c.type)}</h3><div>${esc(c.request)}</div>
    <div class="meta">status <b>${esc(c.status)}</b> · route <b>${esc(c.route)}</b>${c.expected_route?` (expected ${esc(c.expected_route)})`:""} · ${c.seconds}s total · baseline ${c.baseline.seconds}s</div>
    <div class="rubric"><b>${c.kind==="scenario"?"Rubric":"Expected"}:</b> ${esc(c.rubric)}</div>
    <div class="imgs">${inp}${atts.join("")}${none}</div>
    <div class="msg">${md(c.message)}</div>
    <details><summary>Agent outputs (planner, vision, prompt, generation settings, timing)</summary><pre>${esc(JSON.stringify({planner:c.plan, vision:c.vision, prompt_engineering:c.prompt,
      generation:c.attempts.map(a=>({attempt:a.n,mode:a.mode,model:a.model,prompt:a.prompt,negative:a.negative,seed:a.seed,steps:a.steps,strength:a.strength,image_guidance:a.image_guidance,seconds:a.seconds})), timing:c.timing}, null, 1))}</pre></details>
    ${qs}${crit}
    <textarea rows="2" placeholder="Notes (optional, e.g. which rubric item failed)" data-note="${c.id}" data-r="${rater}">${esc(st[`${c.id}|note`]||"")}</textarea>
  </section>`;
}

function rateView(rater){
  const [n, t] = progress(S[rater]);
  return `<div class="help"><b>Rater ${rater}.</b> Judge each case with its rubric on your own — don't look at your partner's answers.
    Your answers are kept in this browser as you go. When finished, <b>Download my ratings</b> and send the file to your partner (or keep it for the Resolve tab).
    Click an image to enlarge. The green outline marks the final output. For each attempt, decide yourself whether it should be
    accepted or revised; the critic's own verdict is shown only after you answer, so it can't sway you.</div>
    <div class="q"><button class="btn primary" id="dl-mine" ${n<t?"":""}>Download my ratings (p3_ratings_${rater}.csv)</button>
    <label class="btn">Load a ratings file<input type="file" accept=".csv" id="load-mine" class="hidden"></label>
    <span class="note">${n}/${t} required answers done${n<t?" — you can download a partial file and continue later":""}</span></div>
    <h2>Scenarios (15)</h2>${scen.map(c => caseCard(c, rater)).join("")}
    <h2>Red-team cases (10) — optional overrides</h2>${red.map(c => caseCard(c, rater)).join("")}`;
}

function resolveView(){
  const A = S.A, B = S.B, R = S.R, rows = [];
  let agF=0, agB=0, agC=0, nC=0, missing=0;
  for (const it of ITEMS){
    const a = val(A,it), b = val(B,it);
    if (!it.optional && (!a || !b)) missing++;
    if (it.mirror) continue;
    const same = a && b && a === b;
    if (it.kind==="final" && same) agF++; if (it.kind==="baseline" && same) agB++;
    if (it.kind==="critic"){ nC++; if (same) agC++; }
    rows.push({it, a, b, same});
  }
  const mirrorAgree = scen.filter(c=>!c.attempts.length && val(A,{key:`${c.id}|final`})===val(B,{key:`${c.id}|final`}) && val(A,{key:`${c.id}|final`})).length;
  agB += mirrorAgree;
  const dis = rows.filter(r => !r.same && !r.it.optional || (r.it.optional && (r.a||"auto") !== (r.b||"auto")));
  const noDecision = dis.filter(r => !R[r.it.key]), noNote = dis.filter(r => R[r.it.key] && !(R[r.it.key+"|note"]||"").trim());
  const unresolved = noDecision.length;
  const lbl = it => it.kind==="critic" ? `${it.id} critic, attempt ${it.attempt}` : it.kind==="redteam" ? `${it.id} red-team verdict` : `${it.id} ${it.kind}`;
  const opts = it => it.opts;
  const ready = missing === 0 && unresolved === 0;
  const rowState = r => !R[r.it.key] ? `<b style="color:var(--bad)">pick a decision</b>` : !(R[r.it.key+"|note"]||"").trim() ? `<span style="color:var(--warn)">add a note</span>` : `<span style="color:var(--ok)">done</span>`;
  return `<div class="help"><b>Resolve together.</b> Load both raters' files (or, if both rated in this browser, they are already here).
    Settle every highlighted disagreement and write why. Then download the three files and put them in <code>Lab2/outputs/p3/</code> and Run all.</div>
    <div class="q"><label class="btn">Load p3_ratings_A.csv<input type="file" accept=".csv" data-into="A" class="hidden ld"></label>
    <label class="btn">Load p3_ratings_B.csv<input type="file" accept=".csv" data-into="B" class="hidden ld"></label>
    <span class="note">A: ${progress(A)[0]}/${progress(A)[1]} · B: ${progress(B)[0]}/${progress(B)[1]} answers</span></div>
    <div class="stat"><div><b>${agF}/15</b>agree on final</div><div><b>${agB}/15</b>agree on baseline</div><div><b>${agC}/${nC}</b>agree on critic labels</div>
    <div><b>${dis.length}</b>disagreements</div><div><b>${unresolved}</b>without a decision</div><div><b>${noNote.length}</b>without a note</div></div>
    ${missing?`<p class="note" style="color:var(--bad)">${missing} required answers are missing from A or B — finish rating first.</p>`:""}
    <div class="q"><button class="btn primary" id="dl-all" ${ready?"":"disabled"}>Download all 3 resolved files (to Downloads)</button>
    <span class="note">${ready?(noNote.length?`Ready. ${noNote.length} decision(s) have no note yet — the report needs to say how you settled each one, so add them if you can.`:"Ready."):`Locked: ${missing?missing+" answers missing from A or B; ":""}${unresolved} disagreement(s) still need a decision (marked in red below).`}</span></div>
    <table><thead><tr><th>Item</th><th>A</th><th>B</th><th>Decision</th><th>Why (resolution note)</th><th>Status</th></tr></thead><tbody>
    ${dis.map(r => `<tr class="dis"><td><a href="#" data-jump="${r.it.id}">${lbl(r.it)}</a></td><td>${r.a||"—"}</td><td>${r.b||"—"}</td>
       <td>${seg({...r.it, mirror:false}, R, "R")}</td><td><input type="text" data-rnote="${r.it.key}" value="${esc(R[r.it.key+"|note"]||"")}" placeholder="how you settled it"></td><td>${rowState(r)}</td></tr>`).join("")
       || `<tr><td colspan="5" class="note">No disagreements${missing?" yet":""}.</td></tr>`}
    </tbody></table>
    <div class="q" style="margin-top:14px"><button class="btn primary" id="dl-sheet" ${ready?"":"disabled"}>rating_sheet_filled.csv</button>
    <button class="btn primary" id="dl-crit" ${ready?"":"disabled"}>critic_labels_filled.csv</button>
    <button class="btn" id="dl-red" ${ready?"":"disabled"}>redteam_filled.csv</button>
    <span class="note">${ready?"Ready — save them into Lab2/outputs/p3/.":"Unlocks once both raters are complete and every disagreement has a decision."}</span></div>
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
async function save(name, text, plain){
  if (window.showSaveFilePicker && !plain){ try{ const h = await showSaveFilePicker({suggestedName:name, types:[{description:"CSV",accept:{"text/csv":[".csv"]}}]});
      const w = await h.createWritable(); await w.write(text); await w.close(); return; } catch(e){ if (e.name==="AbortError") return; } }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text],{type:"text/csv"})); a.download = name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 2000);
}
function raterCSV(r){ const st=S[r], rows=[];
  for (const it of ITEMS) if (st[it.key]) rows.push({rater:r, key:it.key, value:st[it.key]});
  for (const c of D.cases) if (st[`${c.id}|note`]) rows.push({rater:r, key:`${c.id}|note`, value:st[`${c.id}|note`]});
  return toCSV(rows, ["rater","key","value"]); }
function loadRater(text, into){ const rows = parseCSV(text), st = {};
  for (const r of rows) if (r.key) st[r.key] = r.value;
  const who = rows[0] && rows[0].rater; if (who && who !== into && !confirm(`This file is rater ${who}'s. Load it as rater ${into}?`)) return;
  S[into] = st; store.set(into, st); render(); }

function pick(it){ const a = val(S.A,it), b = val(S.B,it);
  if (it.optional){ const aa=a||"auto", bb=b||"auto"; return aa===bb ? aa : S.R[it.key]; }
  return a===b ? a : S.R[it.key]; }
function sheetCSV(){
  const cols = Object.keys(D.sheet[0]), out = D.sheet.map(row => { const r = {...row}, notes = [];
    for (const side of ["final","baseline"]){ const it = ITEMS.find(i => i.key === `${row.id}|${side}`);
      r[`rater_A_${side}`] = val(S.A,it); r[`rater_B_${side}`] = val(S.B,it); r[`resolved_${side}`] = pick(it.mirror ? ITEMS.find(i=>i.key===`${row.id}|final`) : it);
      const n = S.R[`${it.key}|note`]; if (n && val(S.A,it)!==val(S.B,it)) notes.push(`${side}: ${n}`); }
    r.resolution_note = notes.join(" | ");
    const na = S.A[`${row.id}|note`], nb = S.B[`${row.id}|note`]; r.rater_A_note = na||""; r.rater_B_note = nb||""; return r; });
  return toCSV(out, [...cols, "rater_A_note", "rater_B_note"]); }
function critCSV(){
  const cols = Object.keys(D.critic[0]), out = D.critic.map(row => { const it = ITEMS.find(i => i.key === `${row.id}|critic|${row.attempt}`);
    return {...row, your_decision: pick(it), your_decision_A: val(S.A,it), your_decision_B: val(S.B,it), resolution_note: val(S.A,it)!==val(S.B,it) ? (S.R[`${it.key}|note`]||"") : ""}; });
  return toCSV(out, [...cols.filter(c=>c!=="your_decision"), "your_decision", "your_decision_A", "your_decision_B", "resolution_note"]); }
function redCSV(){ const out = red.map(c => { const it = ITEMS.find(i => i.key === `${c.id}|redteam`), p = pick(it);
    return {id:c.id, auto_attack_success: c.auto_attack, override: p==="yes" ? "True" : p==="no" ? "False" : "", rater_A: val(S.A,it)||"auto", rater_B: val(S.B,it)||"auto", note: S.R[`${it.key}|note`]||""}; });
  return toCSV(out, ["id","auto_attack_success","override","rater_A","rater_B","note"]); }

// ---------- render + events ----------
function render(){
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.m === mode));
  const y = window.scrollY;
  document.getElementById("main").innerHTML = mode === "R" ? resolveView() : rateView(mode);
  const [n,t] = progress(S[mode === "R" ? "A" : mode]);
  document.getElementById("prog").textContent = mode === "R" ? `A ${progress(S.A).join("/")} · B ${progress(S.B).join("/")}` : `${n}/${t} done`;
  window.scrollTo(0, y);
}
document.getElementById("tabs").onclick = e => { const m = e.target.dataset.m; if (!m) return; mode = m; store.set("mode", m); render(); window.scrollTo(0,0); };
document.addEventListener("click", e => {
  const b = e.target.closest("button[data-k]");
  if (b && !b.disabled){ const st = S[b.dataset.r]; st[b.dataset.k] = st[b.dataset.k] === b.dataset.v ? undefined : b.dataset.v; store.set(b.dataset.r, st); render(); return; }
  if (e.target.tagName === "IMG" && e.target.closest("figure")){ document.getElementById("lbimg").src = e.target.src; document.getElementById("lb").style.display = "flex"; return; }
  const j = e.target.closest("[data-jump]");
  if (j){ e.preventDefault(); const c = D.cases.find(x => x.id === j.dataset.jump); document.getElementById("peek").innerHTML = "<h2>Case</h2>" + caseCard(c, "R").replace(/<div class="q">[\s\S]*?<textarea[\s\S]*?<\/textarea>/, ""); document.getElementById("peek").scrollIntoView(); return; }
  if (e.target.id === "dl-mine") save(`p3_ratings_${mode}.csv`, raterCSV(mode));
  if (e.target.id === "dl-all"){ save("rating_sheet_filled.csv", sheetCSV(), true); setTimeout(() => save("critic_labels_filled.csv", critCSV(), true), 300); setTimeout(() => save("redteam_filled.csv", redCSV(), true), 600); }
  if (e.target.id === "dl-sheet") save("rating_sheet_filled.csv", sheetCSV());
  if (e.target.id === "dl-crit") save("critic_labels_filled.csv", critCSV());
  if (e.target.id === "dl-red") save("redteam_filled.csv", redCSV());
});
document.addEventListener("input", e => {
  if (e.target.dataset.note){ const st = S[e.target.dataset.r]; st[`${e.target.dataset.note}|note`] = e.target.value; store.set(e.target.dataset.r, st); }
  if (e.target.dataset.rnote){ S.R[`${e.target.dataset.rnote}|note`] = e.target.value; store.set("R", S.R); clearTimeout(window._rt); window._rt = setTimeout(() => { const el = document.activeElement, k = el && el.dataset.rnote, pos = el && el.selectionStart; render(); if (k){ const n = document.querySelector(`input[data-rnote="${k}"]`); n.focus(); n.setSelectionRange(pos, pos); } }, 400); }
});
document.addEventListener("change", e => {
  const f = e.target.files && e.target.files[0]; if (!f) return;
  const into = e.target.id === "load-mine" ? mode : e.target.dataset.into;
  f.text().then(t => loadRater(t, into)); e.target.value = "";
});
render();
</script></body></html>
"""

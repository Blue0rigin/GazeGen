"use strict";
const D = window.GAZE_DATA,
  $ = (id) => document.getElementById(id);
let qid = null,
  draw = 0,
  time = 0,
  playing = false,
  last = 0,
  panels = [],
  custom = null,
  server = false,
  requestRevision = 0;
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
D.cases.forEach((c, i) => $("passage").add(new Option(c.label, i)));
function current() {
  return D.cases[+$("passage").value];
}
function condition(v, t) {
  let list = v.conditions.filter((c) => c.task === t);
  return list.find((c) => c.question_id === qid) || list[0];
}
function questions() {
  const v = current().variants[$("version").value],
    cs = v.conditions.filter((c) => c.task === "IS");
  if (!cs.some((c) => c.question_id === qid)) qid = cs[0].question_id;
  $("questions").innerHTML = "";
  for (const c of cs) {
    const b = document.createElement("button");
    b.textContent = c.question;
    b.setAttribute("aria-pressed", String(c.question_id === qid));
    b.onclick = () => {
      requestRevision++;
      qid = c.question_id;
      custom = null;
      draw = 0;
      questions();
      render();
    };
    $("questions").append(b);
  }
}
function add(v, t, source, title) {
  const c = condition(v, t);
  let bank = source === "generated" ? c.generated : c.human;
  if (custom && source === "generated" && t === "IS" && v.text_key === custom.text_key)
    bank = custom.paths;
  if (custom && source === "human" && t === "IS") {
    panels.push({
      empty: true,
      title: "Human comparison unavailable",
      message:
        "No human participant read this passage with your new question. Switch to a saved question for a matched comparison.",
    });
    return;
  }
  const p = bank[draw % bank.length];
  panels.push({
    words: v.words,
    events: p.events,
    source,
    title,
    question: t === "IS" ? custom?.question || c.question : "",
    task: t,
    meta:
      source === "generated"
        ? custom
          ? "New inference"
          : "Selected sample " +
            ((draw % bank.length) + 1) +
            " of " +
            bank.length +
            " · archive draw " +
            p.archive_draw
        : p.label + " of " + bank.length,
    seed: p.seed,
  });
}
// Rebuild comparison panels from saved data or a matching custom response.
function render() {
  playing = false;
  $("play").textContent = "▶ Play";
  time = 0;
  panels = [];
  $("another").disabled = !!custom;
  const c = current(),
    ver = $("version").value,
    t = $("task").value,
    mode = $("comparison").value;
  $("task-help").textContent =
    t === "IS"
      ? "Both the text and the question condition generation."
      : "The model receives the passage and a natural-reading instruction; no previewed question.";
  $("question-area").hidden = t === "NR" && mode !== "tasks";
  $("version-label").hidden = mode === "versions";
  $("comparison-help").textContent = {
    human: "Same material and task; model samples beside observed readers.",
    versions: "Paired versions, with a generated and a human path for each.",
    tasks: "Same material; the reading purpose and visible question change.",
  }[mode];
  if (mode === "versions") {
    for (const source of ["generated", "human"])
      for (const vv of ["Advanced", "Elementary"]) {
        add(c.variants[vv], t, source, vv + " · " + source);
      }
  } else if (mode === "tasks") {
    add(c.variants[ver], "NR", "generated", "Natural reading · generated");
    add(c.variants[ver], "IS", "generated", "Information seeking · generated");
    add(c.variants[ver], "NR", "human", "Natural reading · human");
    add(c.variants[ver], "IS", "human", "Information seeking · human");
  } else {
    add(c.variants[ver], t, "generated", ver + " · generated");
    add(c.variants[ver], t, "human", ver + " · human");
  }
  $("context").textContent = custom
    ? "New question: " + custom.question
    : t === "IS" || mode === "tasks"
      ? "Search question: " + condition(c.variants[ver], "IS").question
      : "Read for general comprehension. The question is not available before reading.";
  $("panels").innerHTML = "";
  panels.forEach((p) => {
    const el = document.createElement("article");
    el.className = "panel";
    if (p.empty) {
      el.innerHTML =
        '<div class="panel-head"><h3>' +
        esc(p.title) +
        '</h3></div><div class="empty">' +
        esc(p.message) +
        "</div>";
      $("panels").append(el);
      return;
    }
    p.color = p.source === "generated" ? "#087f8c" : "#bd652a";
    p.ends = [];
    p.total = 0;
    p.dwell = Array(p.words.length).fill(0);
    p.visits = Array(p.words.length).fill(0);
    let backs = 0;
    p.events.forEach((e, j) => {
      p.total += e.duration_ms;
      p.ends.push(p.total);
      p.dwell[e.word_id] += e.duration_ms;
      p.visits[e.word_id]++;
      if (j && e.word_id < p.events[j - 1].word_id) backs++;
    });
    el.style.setProperty("--color", p.color);
    el.innerHTML =
      '<div class="panel-head"><div><h3>' +
      esc(p.title) +
      '</h3><div class="panel-meta">' +
      esc(p.meta) +
      " · " +
      p.words.length +
      ' words</div></div><span class="badge" style="color:' +
      p.color +
      '">' +
      (p.source === "generated" ? "Model prediction" : "Observed data") +
      '</span></div><div class="stage"><div class="passage">' +
      p.words
        .map(
          (w, j) =>
            '<span class="word" data-word="' +
            j +
            '" title="' +
            esc(w) +
            ": " +
            p.visits[j] +
            " fixations; " +
            Math.round(p.dwell[j]) +
            ' ms total">' +
            esc(w) +
            "</span>",
        )
        .join(" ") +
      '</div><svg class="overlay" aria-hidden="true"></svg></div><div class="event-label">Press Play to follow the fixations.</div><div class="path-stats"><span><strong>' +
      p.events.length +
      "</strong>fixations</span><span><strong>" +
      (p.total / 1000).toFixed(1) +
      " s</strong>total fixation time</span><span><strong>" +
      Math.round((p.visits.filter((x) => x > 0).length / p.words.length) * 100) +
      "%</strong>words fixated</span><span><strong>" +
      Math.round((backs / Math.max(1, p.events.length - 1)) * 100) +
      "%</strong>backward transitions</span></div>";
    $("panels").append(el);
    p.el = el;
    p.spans = [...el.querySelectorAll(".word")];
    p.svg = el.querySelector("svg");
    p.stage = el.querySelector(".stage");
    p.label = el.querySelector(".event-label");
  });
  paint();
}
function maxTime() {
  return Math.max(1, ...panels.filter((p) => !p.empty).map((p) => p.total));
}
// Draw in the current browser layout; these are not original screen coordinates.
function paint() {
  const max = maxTime();
  $("time").value = Math.round((time / max) * 1000);
  $("clock").textContent =
    (time / 1000).toFixed(1) + " / " + (max / 1000).toFixed(1) + " s";
  for (const p of panels) {
    if (p.empty) continue;
    let idx = time <= 0 ? -1 : p.ends.findIndex((e) => e >= time);
    if (idx < 0 && time > 0) idx = p.events.length - 1;
    p.spans.forEach((s, j) => {
      s.classList.toggle(
        "active",
        idx >= 0 && p.events[idx].word_id === j && time <= p.total,
      );
      const heat = $("heat").checked ? p.dwell[j] / Math.max(...p.dwell) : 0;
      s.style.background = heat
        ? "rgba(" +
          (p.source === "generated" ? "8,127,140" : "189,101,42") +
          "," +
          (heat * 0.33).toFixed(2) +
          ")"
        : "";
    });
    const box = p.stage.getBoundingClientRect(),
      xy = (e) => {
        let r = p.spans[e.word_id].getBoundingClientRect();
        return [r.left - box.left + r.width / 2, r.top - box.top + r.height / 2];
      };
    let svg = "";
    if (idx >= 0) {
      const first = Math.max(0, idx - 11),
        ev = p.events.slice(first, idx + 1),
        pts = ev.map(xy);
      svg =
        '<polyline points="' +
        pts.map((p) => p.join(",")).join(" ") +
        '" fill="none" stroke="' +
        p.color +
        '" stroke-width="1.8" opacity=".55"/>';
      ev.forEach((e, j) => {
        const [x, y] = pts[j],
          active = first + j === idx;
        svg +=
          '<circle cx="' +
          x +
          '" cy="' +
          y +
          '" r="' +
          (active
            ? Math.max(7, Math.min(16, 6 + Math.log1p(e.duration_ms) / 1.3))
            : 3) +
          '" fill="' +
          p.color +
          '" fill-opacity="' +
          (active ? ".20" : ".55") +
          '" stroke="' +
          p.color +
          '" stroke-width="' +
          (active ? 2 : 0) +
          '"/>';
      });
      const e = p.events[idx];
      p.label.textContent =
        (time >= p.total ? "Completed · " : "") +
        "Fixation " +
        (idx + 1) +
        " / " +
        p.events.length +
        " · “" +
        p.words[e.word_id] +
        "” · " +
        Math.round(e.duration_ms) +
        " ms";
    } else p.label.textContent = "Press Play to follow the fixations.";
    p.svg.innerHTML = svg;
  }
}
function tick(now) {
  if (playing) {
    time = Math.min(maxTime(), time + (now - last) * +$("speed").value);
    paint();
    if (time >= maxTime()) {
      playing = false;
      $("play").textContent = "▶ Replay";
    }
  }
  last = now;
  requestAnimationFrame(tick);
}
$("play").onclick = () => {
  if (time >= maxTime()) time = 0;
  playing = !playing;
  $("play").textContent = playing ? "Ⅱ Pause" : "▶ Play";
};
$("reset").onclick = () => {
  time = 0;
  playing = false;
  $("play").textContent = "▶ Play";
  paint();
};
$("step").onclick = () => {
  playing = false;
  $("play").textContent = "▶ Play";
  const times = panels
    .filter((p) => !p.empty)
    .flatMap((p) => p.ends)
    .filter((t) => t > time + 1);
  time = times.length ? Math.min(...times) + 0.1 : maxTime();
  paint();
};
$("time").oninput = () => {
  playing = false;
  $("play").textContent = "▶ Play";
  time = (+$("time").value / 1000) * maxTime();
  paint();
};
$("heat").onchange = paint;
$("another").onclick = () => {
  draw++;
  render();
};
for (const id of ["passage", "task", "comparison", "version"])
  $(id).onchange = () => {
    requestRevision++;
    custom = null;
    draw = 0;
    questions();
    render();
  };
window.addEventListener("resize", paint);
$("generate").onclick = async () => {
  const q = $("custom-question").value.trim();
  if (!q) {
    $("generation-status").textContent = "Enter a question first.";
    return;
  }
  if (!server) {
    $("generation-status").textContent =
      "Start the local model server described in README, then open its localhost address. Saved examples remain available here.";
    return;
  }
  const v = current().variants[$("version").value];
  const requestedRevision = ++requestRevision;
  $("generate").disabled = true;
  $("generation-status").textContent =
    "Generating a new trajectory from the fixed model…";
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        words: v.words,
        question: q,
        task: "IS",
        seed: 42 + draw,
      }),
    });
    const d = await res.json();
    if (!res.ok) throw Error(d.error || "Generation failed");
    // A delayed response must not replace a different passage or question.
    if (requestedRevision !== requestRevision) {
      $("generation-status").textContent =
        "The selection changed; the previous response was not applied.";
      return;
    }
    custom = { question: q, text_key: v.text_key, paths: d.paths };
    $("task").value = "IS";
    $("comparison").value = "human";
    draw = 0;
    render();
    $("generation-status").textContent =
      "New model output. No matched human observation exists for this question.";
  } catch (e) {
    $("generation-status").textContent = e.message;
  } finally {
    $("generate").disabled = false;
  }
};
$("provenance").textContent = "Checkpoint SHA-256: " + D.checkpoint_sha256;
questions();
render();
requestAnimationFrame(tick);
const localHost = ["localhost", "127.0.0.1", "::1", "[::1]"].includes(location.hostname);
$("generate").disabled = true;
if (localHost)
  fetch("/api/status")
    .then((r) => r.json())
    .then((d) => {
      server = !!d.inference;
      $("generate").disabled = !server;
      $("server-status").textContent = server
        ? "Local inference is enabled. Your question will be encoded and passed to the fixed GazeGen model; it does not retrain the model."
        : "Saved examples are available. Start the inference server with an encoder to generate new questions.";
    })
    .catch(() => {});

if (!localHost)
  $("server-status").textContent =
    "This online demo plays saved model predictions and human records. For a new question, download the repository and start its local inference server.";

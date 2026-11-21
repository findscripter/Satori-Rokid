(() => {
  "use strict";

  const styles = `
    .tline-shell { --tl-user:#397e94; --tl-assistant:#d25b3e; --tl-system:#75827b; --tl-ink:#293e37;
      --tl-dim:#627269; --tl-rule:#dfE5df; --tl-paper:#f7f8f4; --tl-wave-height:46px;
      --tl-lane-height:34px; --tl-ruler-height:27px; width:100%; min-width:0; }
    .tline-grid { display:grid; grid-template-columns:clamp(76px,9%,106px) minmax(0,1fr);
      align-items:stretch; gap:12px; position:relative; }
    .tline-labels,.tline-plot { display:grid; grid-template-rows:var(--tl-wave-height) var(--tl-lane-height) var(--tl-lane-height) var(--tl-ruler-height); min-width:0; }
    .tline-label { min-width:0; display:flex; flex-direction:column; justify-content:center; align-items:flex-start; gap:1px; }
    .tline-name { font-family:inherit; font-size:12px; font-weight:600; line-height:1.25; color:var(--tl-ink); }
    .tline-micro { font-size:10px; line-height:1.35; color:var(--tl-dim); font-variant-numeric:tabular-nums; }
    .tline-plot { position:relative; isolation:isolate; cursor:crosshair; touch-action:pan-y; user-select:none; }
    .tline-lane { position:relative; border-bottom:1px solid var(--tl-rule); min-width:0; box-sizing:border-box; }
    .tline-wave { background:var(--tl-paper); border-radius:5px 5px 0 0; }
    .tline-wave canvas { display:block; height:100%; width:100%; }
    .tline-user,.tline-speech { background:linear-gradient(to right,#e8ede83b,#fafbf82e); }
    .tline-gridmark { position:absolute; top:0; bottom:var(--tl-ruler-height); pointer-events:none;
      width:0; border-left:1px dashed #a4b3a745; z-index:1; }
    .tline-mark { position:absolute; top:0; bottom:0; margin:0; padding:0; appearance:none;
      border:0; cursor:pointer; background:transparent; color:inherit; z-index:2; border-radius:3px; }
    .tline-mark:focus-visible,.tline-ruler:focus-visible { outline:2px solid #24566e; outline-offset:1px; z-index:5; }
    .tline-point { width:20px; transform:translateX(-50%); }
    .tline-point::after { content:""; position:absolute; top:50%; left:50%; width:8px; height:8px;
      border-radius:2px; transform:translate(-50%,-50%) rotate(45deg); background:var(--tl-user); }
    .tline-point:hover::after,.tline-point:focus-visible::after,.tline-point[data-current]::after {
      outline:3px solid #397e942e; background:#235e72; }
    .tline-segment { min-width:1px; }
    .tline-segment::before { content:""; position:absolute; inset:0 -4px; }
    .tline-segment::after { content:""; position:absolute; inset:calc(50% - 4px) 0; height:8px;
      background:var(--tl-assistant); border-radius:3px; }
    .tline-segment[data-system]::after { background:var(--tl-system); }
    .tline-segment:hover::after,.tline-segment[data-current]::after,.tline-segment:focus-visible::after {
      box-shadow:0 0 0 2px #3e493b26; filter:saturate(1.2) brightness(.91); }
    .tline-ruler { position:relative; color:var(--tl-dim); font:10px/1.25 "IBM Plex Mono",Consolas,monospace;
      border-radius:2px; cursor:ew-resize; font-variant-numeric:tabular-nums; }
    .tline-tick { position:absolute; top:7px; transform:translateX(-50%); white-space:nowrap; }
    .tline-tick[data-first] { transform:none; }
    .tline-tick[data-last] { transform:translateX(-100%); }
    .tline-cursor,.tline-hover { position:absolute; top:0; bottom:var(--tl-ruler-height); left:0;
      width:1px; background:#253d36; pointer-events:none; z-index:3; }
    .tline-cursor::before { content:""; position:absolute; top:0; left:-3px; width:7px; height:6px;
      background:#253d36; clip-path:polygon(0 0,100% 0,100% 45%,50% 100%,0 45%); }
    .tline-hover { opacity:.45; background:transparent; border-left:1px dotted #405749; z-index:2; }
    .tline-tooltip { position:absolute; z-index:6; pointer-events:none; box-sizing:border-box; max-width:min(275px,100%);
      padding:5px 8px; background:#253d36; color:#fff; border-radius:4px; box-shadow:0 3px 12px #1f36272c;
      line-height:1.5; font-size:11px; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }
    .tline-tooltip[hidden],.tline-hover[hidden] { display:none; }
    @container (max-width:560px) {
      .tline-grid { grid-template-columns:63px minmax(0,1fr); gap:7px; }
      .tline-tick[data-minor] { visibility:hidden; }
      .tline-point { width:16px; }
      .tline-name { font-size:11px; }
    }
    .tline-shell { container-type:inline-size; }
  `;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clock(seconds, fraction = false) {
    const tenths = Math.floor(Math.max(0, seconds) * 10 + 1e-7);
    const min = Math.floor(tenths / 600);
    const sec = String(Math.floor(tenths / 10) % 60).padStart(2, "0");
    return `${min}:${sec}${fraction ? "." + tenths % 10 : ""}`;
  }

  window.createReplayTimeline = function createReplayTimeline(root, data, onSeek) {
    if (!document.getElementById("tline-styles")) {
      const sheet = element("style", "");
      sheet.id = "tline-styles";
      sheet.textContent = styles;
      document.head.append(sheet);
    }
    const duration = data.duration;
    const bound = t => Math.min(duration, Math.max(0, t));
    const percent = t => (bound(t) / duration * 100) + "%";
    const clampPixel = (n, high) => Math.max(0, Math.min(n, high));
    const grid = element("div", "tline-grid");
    const labels = element("div", "tline-labels");
    const plot = element("div", "tline-plot");
    root.classList.add("tline-shell");
    root.replaceChildren(grid);
    grid.append(labels, plot);
    const replyCount = data.speech.filter(s => s.turnId !== null).length;
    for (const [name, detail] of [
      ["混合音轨", "相对响度"],
      ["用户提问", data.turns.length + " 次"],
      ["语音播报", `${replyCount} 回复 · ${data.speech.length - replyCount} 系统`]
    ]) {
      const label = element("div", "tline-label");
      label.append(element("span", "tline-name", name), element("span", "tline-micro", detail));
      labels.append(label);
    }
    function lane(className) {
      const row = element("div", "tline-lane tline-" + className);
      plot.append(row);
      return row;
    }
    const waveLane = lane("wave");
    const canvas = element("canvas", "tline-canvas");
    canvas.setAttribute("aria-label", "眼镜录音的混合响度波形，每半秒一个采样");
    waveLane.append(canvas);
    const userLane = lane("user");
    const speechLane = lane("speech");
    const ruler = element("div", "tline-ruler");
    ruler.tabIndex = 0;
    Object.entries({ role:"slider", "aria-label":"视频回放时间", "aria-valuemin":0, "aria-valuemax":duration,
      "aria-keyshortcuts":"ArrowLeft ArrowRight Home End PageUp PageDown" }).forEach(([k,v]) => ruler.setAttribute(k,v));
    plot.append(ruler);
    for (let i = 0; i <= 6; ++i) {
      const at = i * duration / 6;
      const tick = element("span", "tline-tick", clock(at));
      tick.style.left = percent(at);
      if (i === 0) tick.dataset.first = "";
      if (i === 6) tick.dataset.last = "";
      if (i % 2) tick.dataset.minor = "";
      ruler.append(tick);
      const line = element("div", "tline-gridmark");
      line.style.left = percent(at);
      plot.append(line);
    }
    const cursor = element("div", "tline-cursor");
    const hover = element("div", "tline-hover");
    const tip = element("div", "tline-tooltip");
    hover.hidden = tip.hidden = true;
    plot.append(cursor, hover, tip);
    const marks = [];
    function addMark(row, cls, start, end, caption, system = false) {
      const button = element("button", "tline-mark tline-" + cls);
      button.type = "button";
      button.setAttribute("aria-label", caption);
      button.dataset.at = start;
      button.style.left = percent(start);
      if (end !== null) button.style.width = percent(end - start);
      if (system) button.dataset.system = "";
      button.addEventListener("click", () => seek(start));
      row.append(button);
      marks.push({button, start, end});
    }
    for (const turn of data.turns) {
      const caption = `${clock(turn.userAt,true)}  用户提问 · ${turn.user}${turn.kind === "discarded" ? "（已撤销）" : ""}`;
      addMark(userLane, "point", turn.userAt, null, caption);
    }
    for (const s of data.speech) {
      const type = s.turnId === null ? "系统播报" : "助手播报";
      const caption = `${clock(s.start,true)}–${clock(s.end,true)}  ${type} · ${(s.end - s.start).toFixed(1)} 秒`;
      addMark(speechLane, "segment", s.start, s.end, caption, s.turnId === null);
    }
    let current = 0;
    function update(time) {
      if (!Number.isFinite(time)) return;
      current = bound(time);
      cursor.style.left = percent(current);
      ruler.setAttribute("aria-valuenow", current.toFixed(2));
      ruler.setAttribute("aria-valuetext", clock(current,true));
      for (const mark of marks) {
        const active = mark.end === null ? Math.abs(current - mark.start) < .5 : current >= mark.start && current < mark.end;
        mark.button.toggleAttribute("data-current", active);
      }
    }
    function seek(time) {
      update(bound(time));
      onSeek(current);
    }
    function drawWave() {
      const {width, height} = canvas.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width*dpr);
      canvas.height = Math.round(height*dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr,0,0,dpr,0,0);
      const mid = height/2;
      const waves = data.wave;
      ctx.fillStyle = "#b8c6bb";
      ctx.fillRect(0,mid-.5,width,1);
      const bars = Math.max(1, Math.min(waves.length, Math.floor(width / 2)));
      const pitch = width / bars;
      ctx.fillStyle = "#789286";
      for (let i = 0; i < bars; i++) {
        const first = Math.floor(i * waves.length / bars);
        const last = Math.max(first+1,Math.floor((i+1) * waves.length / bars));
        let power = 0;
        for (let n=first; n<last; n++) power += waves[n]*waves[n];
        const level = Math.sqrt(power / (last-first));
        const amp = Math.max(.6,(1+Math.max(-40,20*Math.log10(Math.max(1e-6, level)))/40)*(mid-3));
        ctx.fillRect(i*pitch,mid-amp,Math.max(1,pitch-.6),2*amp);
      }
    }
    if (window.ResizeObserver) new ResizeObserver(drawWave).observe(canvas);
    else window.addEventListener("resize", drawWave);
    drawWave();

    function pointerTime(event) {
      const rect = plot.getBoundingClientRect();
      return bound((event.clientX - rect.left) * duration / Math.max(1,rect.width));
    }
    function hint(event, at) {
      const button = event.target.closest(".tline-mark");
      const ts = button ? Number(button.dataset.at) : at;
      let detail = clock(ts,true);
      if (button) detail = button.getAttribute("aria-label");
      else if (waveLane.contains(event.target)) {
        const idx = Math.min(data.wave.length-1,Math.floor(ts/duration*data.wave.length));
        const level = data.wave[idx];
        detail += " · " + (level > 0 ? (20*Math.log10(level)).toFixed(1) + " dB" : "静音") + "（相对峰值）";
      }
      hover.hidden = tip.hidden = false;
      hover.style.left = percent(ts);
      tip.textContent = detail;
      const rect = plot.getBoundingClientRect();
      tip.style.left = clampPixel(ts / duration * rect.width + 8, rect.width - tip.offsetWidth) + "px";
      tip.style.top = clampPixel(event.clientY - rect.top - tip.offsetHeight - 10,rect.height-tip.offsetHeight) + "px";
    }
    function hideHint() { hover.hidden = tip.hidden = true; }
    let drag = null;
    plot.addEventListener("pointerdown", event => {
      if (event.button !== 0 || !event.isPrimary || event.target.closest(".tline-mark")) return;
      drag = event.pointerId;
      plot.setPointerCapture(drag);
      const at = pointerTime(event);
      ruler.focus({preventScroll:true});
      seek(at);
      if (event.pointerType === "mouse") hint(event,at);
    });
    plot.addEventListener("pointermove", event => {
      const at = pointerTime(event);
      if (drag === event.pointerId) seek(at);
      if (event.pointerType === "mouse") hint(event,at);
    });
    plot.addEventListener("lostpointercapture", () => { drag = null; hideHint(); });
    plot.addEventListener("pointerleave", () => { if (drag === null) hideHint(); });
    plot.addEventListener("keydown", event => {
      const step = event.shiftKey ? 10 : 1;
      const target = {ArrowLeft:current-step, ArrowRight:current+step, Home:0, End:duration, PageUp:current+30, PageDown:current-30}[event.key];
      if (target === undefined) return;
      event.preventDefault();
      event.stopPropagation();
      hideHint();
      seek(target);
    });
    update(0);
    return update;
  };
})();

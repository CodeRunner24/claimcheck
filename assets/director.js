// Browser side: camera, recording, microphone and sound. All decisions stay on the server.
(function () {
  const SAMPLE = { far: [0.2, 4.7], wide: [5.7, 10.9], label: [11.7, 17.8] };
  const cc = (window.cc = { stream: null, recorder: null, chunks: [], t0: 0, segment: null, source: "", stopping: false });

  const video = () => document.getElementById("cc-video");
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitForVideo() {
    for (let i = 0; i < 60 && !video(); i++) await pause(50);
    return video();
  }

  function record(stream) {
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
    const mimeType = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    cc.t0 = performance.now();
    if (!mimeType) return;
    cc.chunks = [];
    cc.recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 1800000 });
    cc.recorder.ondataavailable = (e) => e.data.size && cc.chunks.push(e.data);
    cc.recorder.start(1000);
  }

  cc.start = async (source) => {
    const v = await waitForVideo();
    if (!v) return "error:viewfinder not ready";
    cc.source = source;
    cc.stopping = false;
    try {
      if (source === "sample") {
        v.srcObject = null;
        v.src = "/samples/sample_take.webm";
        v.muted = true;
        v.loop = false;
        v.ontimeupdate = () => {
          if (cc.segment && (v.currentTime >= cc.segment[1] || v.currentTime < cc.segment[0] - 0.5)) v.currentTime = cc.segment[0];
        };
        v.onended = () => { if (cc.segment) { v.currentTime = cc.segment[0]; v.play(); } };
        v.onpause = () => { if (cc.source === "sample" && !v.ended && !cc.stopping) v.play().catch(() => {}); };
        cc.segment = SAMPLE.far;
        await Promise.race([
          (async () => {
            await new Promise((r) => { if (v.readyState >= 2) r(); else v.onloadeddata = r; });
            v.currentTime = cc.segment[0];
            await v.play();
          })(),
          pause(8000).then(() => { throw new Error("the sample clip did not start"); }),
        ]);
        cc.stream = v.captureStream ? v.captureStream() : null;
      } else {
        cc.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
        });
        v.srcObject = cc.stream;
        await v.play();
      }
      if (cc.stream) record(cc.stream);
      return "ok:" + (v.videoWidth && v.videoHeight ? (v.videoWidth / v.videoHeight).toFixed(4) : "1.7780");
    } catch (e) {
      return "error:" + e.message;
    }
  };

  // Sample mode only: the pre-made clip plays the part of a seller following directions.
  cc.cue = (name) => {
    if (cc.source !== "sample" || !SAMPLE[name] || cc.segment === SAMPLE[name]) return;
    cc.segment = SAMPLE[name];
    const v = video();
    if (v) v.currentTime = cc.segment[0];
    const stale = document.querySelector(".masks svg");  // that mask belonged to the previous shot
    if (stale) stale.remove();
  };

  cc.capture = (maxSide = 1280) => {
    const v = video();
    if (!v || !v.videoWidth) return "";
    follow.grab();
    const c = document.createElement("canvas");
    const s = Math.min(1, maxSide / Math.max(v.videoWidth, v.videoHeight));
    c.width = Math.round(v.videoWidth * s);
    c.height = Math.round(v.videoHeight * s);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.88);
  };

  cc.nextFrame = async (delay) => {
    await pause(delay || 0);
    for (let i = 0; i < 40; i++) {
      const url = cc.capture();
      if (url) return url;
      await pause(100);
    }
    return "";
  };

  // The take goes over plain HTTP: a minute of video is too large for the app's websocket.
  cc.stopTake = (url) =>
    new Promise((resolve) => {
      cc.stopping = true;
      cc.micStop();
      const done = async () => {
        const blob = new Blob(cc.chunks, { type: cc.chunks[0] ? cc.chunks[0].type : "video/webm" });
        cc.chunks = [];
        if (!blob.size) return resolve("");
        try {
          const r = await fetch(url, { method: "POST", body: blob });
          resolve(r.ok ? "ok" : "");
        } catch (e) {
          resolve("");
        }
      };
      const stopTracks = () => {
        if (cc.source !== "sample" && cc.stream) cc.stream.getTracks().forEach((t) => t.stop());
        const v = video();
        if (v && cc.source === "sample") v.pause();
      };
      if (cc.recorder && cc.recorder.state !== "inactive") {
        cc.recorder.onstop = () => { stopTracks(); done(); };
        cc.recorder.stop();
      } else {
        stopTracks();
        done();
      }
    });

  // Live captions: 24 kHz PCM to the relay; partials come straight back and are written into the page.
  let mic = null;
  cc.micStart = async (wsUrl) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      await ctx.resume();
      const socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";
      socket.onmessage = (e) => {
        const box = document.getElementById("cc-partial");
        if (box) box.textContent = e.data;
      };
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      proc.onaudioprocess = (e) => {
        if (socket.readyState !== 1) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
        socket.send(pcm.buffer);
      };
      src.connect(proc);
      proc.connect(ctx.destination);
      mic = { stream, ctx, proc, socket };
      return "ok";
    } catch (e) {
      return "error:" + e.message;
    }
  };

  cc.micStop = () => {
    if (!mic) return;
    const { stream, ctx, proc, socket } = mic;
    mic = null;
    proc.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    if (socket.readyState === 1) socket.send("end");
    setTimeout(() => socket.close(), 1500);
  };

  // SAM answers about once a second, but a mask should stay on the object in between. Every few milliseconds, block-match
  // the mask's region against the tiny grey copy of the exact frame SAM saw, and slide the mask by that displacement.
  // Matching against that frame, not the previous one, means small motions add up instead of rounding away to nothing.
  const follow = { reference: null, pending: null, svg: null, dx: 0, dy: 0, frozen: false, last: 0 };
  const W = 320, H = 180, WIDE = 28, NEAR = 3;
  const small = document.createElement("canvas");
  small.width = W;
  small.height = H;
  const smallCtx = small.getContext("2d", { willReadFrequently: true });

  function greyFrame(v) {
    smallCtx.drawImage(v, 0, 0, W, H);
    const d = smallCtx.getImageData(0, 0, W, H).data;
    const g = new Uint8Array(W * H);
    for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] + d[i * 4 + 1] * 2 + d[i * 4 + 2]) >> 2;
    return g;
  }
  follow.grab = () => { const v = video(); if (v && v.videoWidth) follow.pending = greyFrame(v); };

  function search(reference, current, box, cx, cy, reach, step) {
    let best = Infinity, bx = cx, by = cy;
    for (let dy = cy - reach; dy <= cy + reach; dy += step) {
      for (let dx = cx - reach; dx <= cx + reach; dx += step) {
        let sad = 0, n = 0;
        for (let y = box.y0; y < box.y1 && sad < best; y += 3) {
          const yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (let x = box.x0; x < box.x1; x += 3) {
            const xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            sad += Math.abs(reference[y * W + x] - current[yy * W + xx]);
            n++;
          }
        }
        if (n > 20 && sad < best) { best = sad; bx = dx; by = dy; }
      }
    }
    return [bx, by];
  }

  function followFrame(now) {
    requestAnimationFrame(followFrame);
    const frozen = !!document.querySelector(".frozen");
    if (frozen && !follow.frozen) cc.sound("shutter");
    follow.frozen = frozen;
    const v = video();
    const svg = document.querySelector(".masks svg");
    if (!v || !v.videoWidth || !svg || frozen || now - follow.last < 40) return;
    follow.last = now;
    const fresh = svg !== follow.svg;
    if (fresh) {
      follow.svg = svg;
      follow.reference = follow.pending;
      follow.dx = follow.dy = 0;
    }
    if (!follow.reference || !svg.viewBox.baseVal.width) return;
    const area = svg.getBBox();
    const sx = W / svg.viewBox.baseVal.width, sy = H / svg.viewBox.baseVal.height;
    const box = { x0: Math.max(0, Math.floor(area.x * sx)), y0: Math.max(0, Math.floor(area.y * sy)),
                  x1: Math.min(W, Math.ceil((area.x + area.width) * sx)), y1: Math.min(H, Math.ceil((area.y + area.height) * sy)) };
    if (box.x1 - box.x0 < 8 || box.y1 - box.y0 < 8) return;
    const current = greyFrame(v);
    if (fresh) {
      [follow.dx, follow.dy] = search(follow.reference, current, box, 0, 0, WIDE, 4);
    }
    [follow.dx, follow.dy] = search(follow.reference, current, box, follow.dx, follow.dy, NEAR, 1);
    svg.style.transform = `translate(${(follow.dx / W * 100).toFixed(2)}%, ${(follow.dy / H * 100).toFixed(2)}%)`;
  }
  requestAnimationFrame(followFrame);

  // Small physical sounds instead of a voice.
  let audio = null;
  cc.sound = (kind) => {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const t = audio.currentTime;
      const g = audio.createGain();
      g.connect(audio.destination);
      if (kind === "shutter") {
        const n = audio.createBufferSource();
        const b = audio.createBuffer(1, audio.sampleRate * 0.07, audio.sampleRate);
        const d = b.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
        n.buffer = b;
        g.gain.value = 0.35;
        n.connect(g);
        n.start(t);
      } else {
        const o = audio.createOscillator();
        o.type = kind === "wrong" ? "square" : "sine";
        o.frequency.setValueAtTime(kind === "wrong" ? 150 : 660, t);
        o.frequency.exponentialRampToValueAtTime(kind === "wrong" ? 70 : 990, t + 0.12);
        g.gain.setValueAtTime(kind === "wrong" ? 0.16 : 0.12, t);
        g.gain.exponentialRampToValueAtTime(0.001, t + 0.22);
        o.connect(g);
        o.start(t);
        o.stop(t + 0.24);
      }
    } catch (e) { /* sound is a nicety */ }
  };
})();

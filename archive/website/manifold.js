/**
 * A schematic particle transport on a torus or sphere. No trained model, data,
 * dependencies, or network requests are used by this visualization.
 */
export function createManifold(canvas, options = {}) {
  let context;
  try { context = canvas.getContext("2d", { alpha: true }); } catch { /* Fallback below. */ }
  if (!context) {
    const fallback = document.createElement("div");
    fallback.textContent = "Manifold visualization unavailable in this browser.";
    fallback.setAttribute("role", "status");
    Object.assign(fallback.style, {
      minHeight: "240px", display: "grid", placeItems: "center",
      padding: "24px", color: "#a99fbd", textAlign: "center",
    });
    const wasHidden = canvas.hidden;
    canvas.hidden = true;
    canvas.after(fallback);
    return {
      setShape() {}, setProgress() {}, setPlaying() {}, setDensity() {},
      destroy() { fallback.remove(); canvas.hidden = wasHidden; },
    };
  }

  const TAU = Math.PI * 2;
  const mode = options.mode === "demo" ? "demo" : "hero";
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  const palette = ["184,157,255", "216,199,255", "140,113,219", "126,217,224", "181,239,220"];
  const fills = palette.map(color => `rgb(${color})`);
  const demoSources = [0, 1, 2, 0, 1].map(index => palette[index].split(",").map(Number));
  const demoTargets = [[165, 233, 204], [205, 247, 226], [111, 181, 154], [164, 235, 205], [211, 250, 232]];
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const original = {
    cursor: canvas.style.cursor, touchAction: canvas.style.touchAction,
    label: canvas.getAttribute("aria-label"), role: canvas.getAttribute("role"),
    tabIndex: canvas.getAttribute("tabindex"),
  };
  let shape = options.shape === "sphere" ? "sphere" : "torus";
  let progress = 0;
  let playing = mode === "hero";
  let density = true;
  let reducedMotion = motion.matches;
  let inView = true;
  let destroyed = false;
  let frame = 0;
  let previousTime = 0;
  let lastPaintTime = 0;
  let sceneTime = 0;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let direction = 1;
  let endpointHold = 0;
  let lastProgressReport = -Infinity;
  let yaw = 0;
  let pitch = 0;
  let velocityX = 0;
  let velocityY = 0;
  let pointer = null;
  let particles = [];
  let projected = [];
  let wires = [];

  canvas.style.cursor = "grab";
  canvas.style.touchAction = "pan-y";
  canvas.setAttribute("role", "img");
  if (!canvas.hasAttribute("tabindex")) canvas.tabIndex = 0;

  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const ease = value => value * value * (3 - 2 * value);
  const angleDelta = (from, to) => Math.atan2(Math.sin(to - from), Math.cos(to - from));
  const hash = value => {
    let n = (value + 1) | 0;
    n = Math.imul(n ^ (n >>> 16), 0x21f0aaad);
    n = Math.imul(n ^ (n >>> 15), 0x735a2d97);
    return ((n ^ (n >>> 15)) >>> 0) / 4294967296;
  };
  const unit = vector => {
    const length = Math.hypot(...vector) || 1;
    return vector.map(value => value / length);
  };
  const sphereFocus = unit([0.77, 0.36, 0.64]);
  const glows = palette.map(color => {
    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 48;
    const c = sprite.getContext("2d");
    const gradient = c.createRadialGradient(24, 24, 0, 24, 24, 24);
    gradient.addColorStop(0, `rgba(${color},0.72)`);
    gradient.addColorStop(0.12, `rgba(${color},0.28)`);
    gradient.addColorStop(0.45, `rgba(${color},0.06)`);
    gradient.addColorStop(1, `rgba(${color},0)`);
    c.fillStyle = gradient;
    c.fillRect(0, 0, 48, 48);
    return sprite;
  });

  function label() {
    canvas.setAttribute("aria-label", `Schematic ${shape} particle visualization. Drag or use arrow keys to rotate.`);
  }

  function torusPoint(u, v) {
    const radius = 1.46 + 0.53 * Math.cos(v);
    return [radius * Math.cos(u), radius * Math.sin(u), 0.53 * Math.sin(v)];
  }

  function spherePoint(longitude, latitude) {
    const radius = 1.66;
    const ring = radius * Math.cos(latitude);
    return [ring * Math.cos(longitude), ring * Math.sin(longitude), radius * Math.sin(latitude)];
  }

  function buildParticles() {
    const count = width < 560 ? (mode === "hero" ? 3800 : 3000) : (mode === "hero" ? 6200 : 4600);
    particles = Array.from({ length: count }, (_, i) => {
      // The inverse area CDF avoids overweighting the torus's inner surface.
      const u = TAU * ((i * 0.61803398875 + hash(i * 7) * 0.022) % 1);
      const areaAngle = TAU * hash(i * 11 + 41);
      let v = areaAngle;
      for (let iteration = 0; iteration < 5; iteration++) {
        v -= (v + (0.53 / 1.46) * Math.sin(v) - areaAngle) / (1 + (0.53 / 1.46) * Math.cos(v));
      }
      const z = 1 - 2 * (i + 0.5) / count;
      const phi = i * 2.399963229728653 + hash(i + 59) * 0.026;
      const ring = Math.sqrt(1 - z * z);
      const source = [ring * Math.cos(phi), ring * Math.sin(phi), z];
      const dot = clamp(source.reduce((sum, value, axis) => sum + value * sphereFocus[axis], 0), -1, 1);
      const theta = Math.acos(dot);
      let tangent = sphereFocus.map((value, axis) => value - dot * source[axis]);
      if (Math.hypot(...tangent) < 0.00001) {
        tangent = Math.abs(source[0]) < 0.8 ? [0, source[2], -source[1]] : [-source[2], 0, source[0]];
      }
      tangent = unit(tangent);
      const choice = hash(i * 17 + 901);
      return {
        u, v, du: angleDelta(u, 0.69), dv: angleDelta(v, 0.86),
        source, tangent, theta, phase: hash(i + 3001) * TAU,
        color: choice > 0.986 ? 4 : choice > 0.96 ? 3 : choice > 0.66 ? 1 : choice > 0.22 ? 0 : 2,
        size: 0.47 + hash(i * 31) * 0.64,
        glow: i % 23 === 0,
      };
    });
    projected = particles.map(() => ({ x: 0, y: 0, z: 0, scale: 1, particle: null }));
  }

  function buildWires() {
    wires = [];
    const addCurve = (samples, point) => {
      for (let i = 0; i < samples; i++) wires.push([point(i / samples), point((i + 1) / samples)]);
    };
    if (shape === "torus") {
      for (let i = 0; i < 18; i++) addCurve(40, t => torusPoint(TAU * i / 18, t * TAU));
      for (let i = 0; i < 7; i++) addCurve(90, t => torusPoint(t * TAU, TAU * i / 7));
    } else {
      for (let i = 1; i < 12; i++) addCurve(72, t => spherePoint(t * TAU, -Math.PI / 2 + Math.PI * i / 12));
      for (let i = 0; i < 16; i++) addCurve(48, t => spherePoint(TAU * i / 16, -Math.PI / 2 + t * Math.PI));
    }
  }

  function resize() {
    if (destroyed) return;
    const bounds = canvas.getBoundingClientRect();
    const nextWidth = Math.max(1, bounds.width);
    const nextHeight = Math.max(1, bounds.height);
    const nextDpr = Math.min(2, window.devicePixelRatio || 1);
    if (width === nextWidth && height === nextHeight && dpr === nextDpr) return;
    const populationChanges = !particles.length || (width < 560) !== (nextWidth < 560);
    width = nextWidth;
    height = nextHeight;
    dpr = nextDpr;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    if (populationChanges) buildParticles();
    requestDraw();
  }

  function canDraw() { return !destroyed && inView && !document.hidden; }

  function requestDraw() {
    if (canDraw() && !frame) frame = requestAnimationFrame(render);
  }

  function suspend() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    previousTime = 0;
  }

  function render(timestamp) {
    frame = 0;
    if (!canDraw()) { previousTime = 0; return; }
    if (previousTime && timestamp - lastPaintTime < 15) { requestDraw(); return; }
    lastPaintTime = timestamp;
    const elapsed = previousTime ? Math.min(0.05, (timestamp - previousTime) / 1000) : 0;
    previousTime = timestamp;
    if (playing && !reducedMotion) sceneTime += elapsed;

    // Demo playback gently travels in both directions, with a pause at either end.
    // Explicit playback still works with reduced motion; automatic rotation does not.
    if (mode === "demo" && playing) {
      if (endpointHold > 0) endpointHold -= elapsed;
      else {
        progress = clamp(progress + direction * elapsed / 11, 0, 1);
        if (progress === 0 || progress === 1) {
          direction *= -1;
          endpointHold = 1.15;
        }
      }
      if (onProgress && (timestamp - lastProgressReport >= 32 || progress === 0 || progress === 1)) {
        lastProgressReport = timestamp;
        onProgress(progress);
        if (destroyed) return;
      }
    }
    if (!pointer && !reducedMotion) {
      yaw += velocityX * elapsed * 60;
      pitch = clamp(pitch + velocityY * elapsed * 60, -1.45, 1.45);
      const decay = Math.pow(0.88, elapsed * 60);
      velocityX *= decay;
      velocityY *= decay;
    }

    const c = context;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, width, height);
    const fit = Math.min(width, height) * (shape === "torus" ? 0.218 : 0.247);
    const centerX = width * 0.5;
    const centerY = height * 0.49;
    const tilt = -0.55 + pitch;
    const turn = 0.37 + yaw + (reducedMotion ? 0 : Math.sin(sceneTime * 0.12) * 0.23);
    const roll = -0.2 + (reducedMotion ? 0 : sceneTime * 0.017);
    const cx = Math.cos(tilt), sx = Math.sin(tilt);
    const cy = Math.cos(turn), sy = Math.sin(turn);
    const cz = Math.cos(roll), sz = Math.sin(roll);
    const project = (x, y, z, out = {}) => {
      const yy = y * cx - z * sx;
      const zz = y * sx + z * cx;
      const xx = x * cy + zz * sy;
      const depth = -x * sy + zz * cy;
      const scale = 5.8 / (5.8 - depth);
      out.x = centerX + (xx * cz - yy * sz) * fit * scale;
      out.y = centerY - (xx * sz + yy * cz) * fit * scale;
      out.z = depth;
      out.scale = scale;
      return out;
    };

    const haze = c.createRadialGradient(centerX, centerY, 0, centerX, centerY, fit * 2.6);
    haze.addColorStop(0, "rgba(112,73,189,0.035)");
    haze.addColorStop(0.57, "rgba(94,57,157,0.018)");
    haze.addColorStop(1, "rgba(73,40,135,0)");
    c.fillStyle = haze;
    c.fillRect(0, 0, width, height);

    // Four depth bands keep fine wire contours quiet, with less draw-call overhead.
    const wireBands = [[], [], [], []];
    for (const segment of wires) {
      const a = project(...segment[0]);
      const b = project(...segment[1]);
      const band = clamp(Math.floor(((a.z + b.z) * 0.25 + 1) * 1.9), 0, 3);
      wireBands[band].push(a, b);
    }
    c.lineWidth = 0.55;
    for (let band = 0; band < 4; band++) {
      c.strokeStyle = `rgba(165,139,227,${0.018 + band * 0.013})`;
      c.beginPath();
      const points = wireBands[band];
      for (let i = 0; i < points.length; i += 2) {
        c.moveTo(points[i].x, points[i].y);
        c.lineTo(points[i + 1].x, points[i + 1].y);
      }
      c.stroke();
    }

    const amount = ease(progress);
    const tint = mode === "demo" && density ? amount : 0;
    const activeFills = mode === "demo" ? demoSources.map((source, index) =>
      `rgb(${source.map((channel, axis) => Math.round(channel + (demoTargets[index][axis] - channel) * tint)).join(",")})`
    ) : fills;
    const brightness = mode === "hero" ? 1.15 : 1;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const out = projected[i];
      const drift = reducedMotion ? 0 : Math.sin(sceneTime * 0.34 + p.phase) * 0.009 * (1 - amount);
      if (shape === "torus") {
        const u = p.u + p.du * amount * 0.86 + drift;
        const v = p.v + p.dv * amount * 0.79 + drift * 1.4;
        const radius = 1.46 + 0.53 * Math.cos(v);
        project(radius * Math.cos(u), radius * Math.sin(u), 0.53 * Math.sin(v), out);
      } else {
        // Great-circle interpolation keeps every point exactly on the sphere.
        const angle = p.theta * amount * 0.86;
        const a = Math.cos(angle), b = Math.sin(angle);
        project(
          1.66 * (p.source[0] * a + p.tangent[0] * b),
          1.66 * (p.source[1] * a + p.tangent[1] * b),
          1.66 * (p.source[2] * a + p.tangent[2] * b), out,
        );
      }
      out.particle = p;
    }
    projected.sort((a, b) => a.z - b.z);

    c.globalCompositeOperation = "lighter";
    for (const p of projected) {
      if (!p.particle.glow || p.z < -0.3) continue;
      const size = (8.5 + p.particle.size * 4) * p.scale;
      const glowAlpha = (0.20 + clamp((p.z + 1.8) / 4, 0, 1) * 0.24) * brightness;
      const sourceColor = mode === "demo" ? [0, 1, 2, 0, 1][p.particle.color] : p.particle.color;
      c.globalAlpha = glowAlpha * (1 - tint);
      c.drawImage(glows[sourceColor], p.x - size / 2, p.y - size / 2, size, size);
      if (tint > 0) {
        c.globalAlpha = glowAlpha * tint;
        c.drawImage(glows[4], p.x - size / 2, p.y - size / 2, size, size);
      }
    }
    c.globalCompositeOperation = "source-over";
    for (const p of projected) {
      const depth = clamp((p.z + 2) / 4, 0, 1);
      const particle = p.particle;
      const pulse = reducedMotion ? 1 : 0.94 + Math.sin(sceneTime * 0.55 + particle.phase) * 0.06;
      const alpha = (0.19 + depth * 0.65) * pulse;
      const color = !density ? (depth > 0.4 ? 0 : 2) : particle.color;
      const radius = particle.size * p.scale * (0.72 + depth * 0.35);
      c.globalAlpha = Math.min(1, alpha * brightness);
      c.fillStyle = activeFills[color];
      c.beginPath();
      c.arc(p.x, p.y, radius, 0, TAU);
      c.fill();
    }
    c.globalAlpha = 1;
    const inertia = Math.abs(velocityX) + Math.abs(velocityY) > 0.00005;
    if ((playing && (!reducedMotion || mode === "demo")) || (!reducedMotion && inertia)) requestDraw();
    else previousTime = 0;
  }

  function onPointerDown(event) {
    if (pointer || (event.pointerType === "mouse" && event.button !== 0)) return;
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
    velocityX = velocityY = 0;
    canvas.style.cursor = "grabbing";
    try { canvas.setPointerCapture(event.pointerId); } catch { /* The pointer may already be cancelled. */ }
  }

  function onPointerMove(event) {
    if (!pointer || pointer.id !== event.pointerId) return;
    const dx = (event.clientX - pointer.x) * 0.006;
    const dy = (event.clientY - pointer.y) * 0.006;
    yaw += dx;
    pitch = clamp(pitch + dy, -1.45, 1.45);
    velocityX = clamp(dx * 0.45, -0.025, 0.025);
    velocityY = clamp(dy * 0.45, -0.025, 0.025);
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    requestDraw();
  }

  function onPointerUp(event) {
    if (!pointer || pointer.id !== event.pointerId) return;
    if (event.type === "pointercancel") velocityX = velocityY = 0;
    pointer = null;
    canvas.style.cursor = "grab";
    try { if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); } catch { /* Already released. */ }
    requestDraw();
  }

  function onKeyDown(event) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowLeft") yaw -= 0.12;
    if (event.key === "ArrowRight") yaw += 0.12;
    if (event.key === "ArrowUp") pitch = clamp(pitch - 0.12, -1.45, 1.45);
    if (event.key === "ArrowDown") pitch = clamp(pitch + 0.12, -1.45, 1.45);
    requestDraw();
  }

  function onVisibility() {
    if (canDraw()) { previousTime = 0; requestDraw(); }
    else suspend();
  }

  function onMotionChange(event) {
    reducedMotion = event.matches;
    velocityX = velocityY = 0;
    previousTime = 0;
    requestDraw();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("lostpointercapture", onPointerUp);
  canvas.addEventListener("keydown", onKeyDown);
  document.addEventListener("visibilitychange", onVisibility);
  window.addEventListener("resize", resize, { passive: true });
  if (motion.addEventListener) motion.addEventListener("change", onMotionChange);
  else motion.addListener(onMotionChange);
  const resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
  resizeObserver?.observe(canvas);
  const intersectionObserver = typeof IntersectionObserver !== "undefined" ? new IntersectionObserver(entries => {
    inView = entries.some(entry => entry.isIntersecting);
    onVisibility();
  }, { rootMargin: "80px" }) : null;
  intersectionObserver?.observe(canvas);
  label();
  buildWires();
  resize();
  requestDraw();

  return {
    setShape(value) {
      if (destroyed || !["torus", "sphere"].includes(value) || value === shape) return;
      shape = value;
      label();
      buildWires();
      requestDraw();
    },
    setProgress(value) {
      if (destroyed || !Number.isFinite(Number(value))) return;
      progress = clamp(Number(value), 0, 1);
      direction = progress === 1 ? -1 : 1;
      endpointHold = 0;
      requestDraw();
    },
    setPlaying(value) {
      if (destroyed) return;
      playing = Boolean(value);
      previousTime = 0;
      requestDraw();
    },
    setDensity(value = true) {
      if (destroyed) return;
      density = Boolean(value);
      requestDraw();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      suspend();
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      canvas.removeEventListener("lostpointercapture", onPointerUp);
      canvas.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("resize", resize);
      if (motion.removeEventListener) motion.removeEventListener("change", onMotionChange);
      else motion.removeListener(onMotionChange);
      if (pointer) {
        try { canvas.releasePointerCapture(pointer.id); } catch { /* No active capture. */ }
      }
      canvas.style.cursor = original.cursor;
      canvas.style.touchAction = original.touchAction;
      for (const [attribute, value] of [["aria-label", original.label], ["role", original.role], ["tabindex", original.tabIndex]]) {
        if (value === null) canvas.removeAttribute(attribute);
        else canvas.setAttribute(attribute, value);
      }
      particles = projected = wires = [];
      glows.length = 0;
      context.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
}

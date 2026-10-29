<script lang="ts">
  import { preview, selectedIds, project } from "$lib/stores/session";
  import type { PreviewObject } from "$lib/api";

  let canvas: HTMLCanvasElement;
  let scale = $state(4);
  let panX = $state(40);
  let panY = $state(40);
  let dragging = $state(false);
  let lastX = 0;
  let lastY = 0;
  let rectSelect = $state<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  $effect(() => {
    preview.subscribe(() => draw());
    selectedIds.subscribe(() => draw());
  });

  function worldToScreen(x: number, y: number) {
    const h = canvas?.height ?? 1;
    return { x: panX + x * scale, y: h - (panY + y * scale) };
  }

  function screenToWorld(x: number, y: number) {
    const h = canvas?.height ?? 1;
    return { x: (x - panX) / scale, y: (h - y - panY) / scale };
  }

  export function fit() {
    const geo = $preview;
    if (!geo || !canvas) return;
    const w = geo.bounds.max.x - geo.bounds.min.x || 1;
    const h = geo.bounds.max.y - geo.bounds.min.y || 1;
    const pad = 48;
    const sx = (canvas.clientWidth - pad * 2) / w;
    const sy = (canvas.clientHeight - pad * 2) / h;
    scale = Math.max(0.2, Math.min(sx, sy));
    panX = pad - geo.bounds.min.x * scale;
    panY = pad - geo.bounds.min.y * scale;
    draw();
  }

  function gridStep(): number {
    const candidates = [1, 5, 10, 50, 100, 500];
    const target = 80 / scale;
    return candidates.reduce((best, n) => (Math.abs(n - target) < Math.abs(best - target) ? n : best));
  }

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const geo = $preview;
    const showGrid = $project?.display.gridEnabled ?? true;
    if (showGrid) {
      const step = gridStep();
      ctx.strokeStyle = "#243044";
      ctx.lineWidth = 1;
      const w0 = screenToWorld(0, rect.height);
      const w1 = screenToWorld(rect.width, 0);
      const xStart = Math.floor(w0.x / step) * step;
      const yStart = Math.floor(w0.y / step) * step;
      ctx.beginPath();
      for (let x = xStart; x <= w1.x; x += step) {
        const s = worldToScreen(x, 0);
        ctx.moveTo(s.x, 0);
        ctx.lineTo(s.x, rect.height);
      }
      for (let y = yStart; y <= w1.y; y += step) {
        const s = worldToScreen(0, y);
        ctx.moveTo(0, s.y);
        ctx.lineTo(rect.width, s.y);
      }
      ctx.stroke();
    }

    const origin = worldToScreen(0, 0);
    ctx.strokeStyle = "#5fbf7a";
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(origin.x + 40, origin.y);
    ctx.stroke();
    ctx.strokeStyle = "#5b8def";
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(origin.x, origin.y - 40);
    ctx.stroke();
    ctx.fillStyle = "#9aa3b2";
    ctx.fillText("X", origin.x + 44, origin.y - 4);
    ctx.fillText("Y", origin.x + 4, origin.y - 44);

    if (geo) {
      ctx.strokeStyle = "#e0b44a";
      ctx.setLineDash([4, 4]);
      const a = worldToScreen(geo.bounds.min.x, geo.bounds.min.y);
      const b = worldToScreen(geo.bounds.max.x, geo.bounds.max.y);
      ctx.strokeRect(a.x, b.y, b.x - a.x, a.y - b.y);
      ctx.setLineDash([]);

      const selected = new Set($selectedIds);
      for (const obj of geo.objects) {
        if (!obj.visible) continue;
        ctx.strokeStyle = selected.has(obj.id) ? "#ffffff" : obj.color;
        ctx.lineWidth = selected.has(obj.id) ? 2.5 : 1.2;
        ctx.globalAlpha = obj.exportEnabled ? 1 : 0.45;
        for (const line of obj.polylines) {
          if (line.length < 2) continue;
          ctx.beginPath();
          const p0 = worldToScreen(line[0][0], line[0][1]);
          ctx.moveTo(p0.x, p0.y);
          for (let i = 1; i < line.length; i++) {
            const p = worldToScreen(line[i][0], line[i][1]);
            ctx.lineTo(p.x, p.y);
          }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
    }

    if (rectSelect) {
      ctx.strokeStyle = "#5b8def";
      ctx.setLineDash([5, 3]);
      const x = Math.min(rectSelect.x0, rectSelect.x1);
      const y = Math.min(rectSelect.y0, rectSelect.y1);
      ctx.strokeRect(x, y, Math.abs(rectSelect.x1 - rectSelect.x0), Math.abs(rectSelect.y1 - rectSelect.y0));
      ctx.setLineDash([]);
    }
  }

  function hitTest(wx: number, wy: number): PreviewObject | null {
    const geo = $preview;
    if (!geo) return null;
    const tol = 4 / scale;
    for (const obj of [...geo.objects].reverse()) {
      if (!obj.visible) continue;
      for (const line of obj.polylines) {
        for (let i = 1; i < line.length; i++) {
          const ax = line[i - 1][0], ay = line[i - 1][1];
          const bx = line[i][0], by = line[i][1];
          const vx = bx - ax, vy = by - ay;
          const len2 = vx * vx + vy * vy || 1;
          let t = ((wx - ax) * vx + (wy - ay) * vy) / len2;
          t = Math.max(0, Math.min(1, t));
          const dx = wx - (ax + t * vx);
          const dy = wy - (ay + t * vy);
          if (dx * dx + dy * dy <= tol * tol) return obj;
        }
      }
    }
    return null;
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const w = screenToWorld(e.offsetX, e.offsetY);
    scale *= factor;
    const after = worldToScreen(w.x, w.y);
    panX += e.offsetX - after.x;
    panY -= after.y - e.offsetY;
    draw();
  }

  function onDown(e: MouseEvent) {
    if (e.button === 1 || e.button === 2 || e.shiftKey && e.button === 0 && e.altKey) {
      dragging = true;
      lastX = e.offsetX;
      lastY = e.offsetY;
      return;
    }
    if (e.button === 0) {
      rectSelect = { x0: e.offsetX, y0: e.offsetY, x1: e.offsetX, y1: e.offsetY };
    }
  }

  function onMove(e: MouseEvent) {
    if (dragging) {
      panX += e.offsetX - lastX;
      panY -= e.offsetY - lastY;
      lastX = e.offsetX;
      lastY = e.offsetY;
      draw();
      return;
    }
    if (rectSelect) {
      rectSelect = { ...rectSelect, x1: e.offsetX, y1: e.offsetY };
      draw();
    }
  }

  function onUp(e: MouseEvent) {
    if (dragging) {
      dragging = false;
      return;
    }
    if (!rectSelect) return;
    const dx = Math.abs(rectSelect.x1 - rectSelect.x0);
    const dy = Math.abs(rectSelect.y1 - rectSelect.y0);
    if (dx < 4 && dy < 4) {
      const w = screenToWorld(e.offsetX, e.offsetY);
      const hit = hitTest(w.x, w.y);
      selectedIds.update((ids) => {
        if (!hit) return e.shiftKey ? ids : [];
        if (e.shiftKey) {
          return ids.includes(hit.id) ? ids.filter((id) => id !== hit.id) : [...ids, hit.id];
        }
        return [hit.id];
      });
    } else {
      const geo = $preview;
      if (geo) {
        const a = screenToWorld(Math.min(rectSelect.x0, rectSelect.x1), Math.max(rectSelect.y0, rectSelect.y1));
        const b = screenToWorld(Math.max(rectSelect.x0, rectSelect.x1), Math.min(rectSelect.y0, rectSelect.y1));
        const hits = geo.objects
          .filter((obj) =>
            obj.visible &&
            obj.polylines.some((line) =>
              line.some((p) => p[0] >= a.x && p[0] <= b.x && p[1] >= a.y && p[1] <= b.y),
            ),
          )
          .map((o) => o.id);
        selectedIds.set(e.shiftKey ? [...new Set([...$selectedIds, ...hits])] : hits);
      }
    }
    rectSelect = null;
    draw();
  }
</script>

<canvas
  class="preview"
  bind:this={canvas}
  onwheel={onWheel}
  onmousedown={onDown}
  onmousemove={onMove}
  onmouseup={onUp}
  oncontextmenu={(e) => e.preventDefault()}
></canvas>

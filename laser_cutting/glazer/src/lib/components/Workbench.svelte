<script lang="ts">
  // App layout: toolbar on top, canvas + layer panel + properties in the
  // middle, status bar at the bottom. The canvas overlay shows the estimated
  // total time with an expandable per-export-layer breakdown; estimator
  // assumptions live in the tooltip next to it. The layer and properties
  // panel widths are adjustable by dragging the dividers (persisted in
  // localStorage).
  import CanvasView from "./CanvasView.svelte";
  import LayerPanel from "./LayerPanel.svelte";
  import PropertiesPanel from "./PropertiesPanel.svelte";
  import Toolbar from "./Toolbar.svelte";
  import StatusBar from "./StatusBar.svelte";
  import Tip from "./Tip.svelte";
  import { estimate, validation, project } from "$lib/stores/session";
  import { fmtMin } from "$lib/format";

  let canvas: CanvasView;
  let estimateExpanded = $state(false);

  function fit() {
    canvas?.fit();
  }

  // --- Resizable side panels -------------------------------------------------
  const MIN_PANEL = 160;
  const MAX_PANEL = 520;
  const PANEL_SEPERATION_WDITH_PX = 3;

  function loadWidth(key: string, fallback: number): number {
    try {
      const v = Number(localStorage.getItem(key));
      return Number.isFinite(v) && v > 0 ? v : fallback;
    } catch {
      return fallback;
    }
  }

  let layersW = $state(loadWidth("glazer.layersW", 200));
  let propsW = $state(loadWidth("glazer.propsW", 160));

  function clampW(w: number): number {
    return Math.min(MAX_PANEL, Math.max(MIN_PANEL, w));
  }

  function startResize(which: "layers" | "props", e: PointerEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = which === "layers" ? layersW : propsW;
    // The canvas column is the flexible (1fr) one, so a divider's screen
    // position moves opposite to its panel width: to make the divider follow
    // the cursor, dragging right must shrink the fixed-width panel on its
    // right (canvas grows). I.e. width -= drag distance for both dividers.
    function onMove(ev: PointerEvent) {
      const w = clampW(startW - (ev.clientX - startX));
      if (which === "layers") layersW = w;
      else propsW = w;
    }
    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      try {
        localStorage.setItem(
          which === "layers" ? "glazer.layersW" : "glazer.propsW",
          String(which === "layers" ? layersW : propsW),
        );
      } catch {
        /* storage unavailable (private mode) — widths just won't persist */
      }
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // Keyboard alternative to dragging: ArrowRight moves the divider right
  // (same as dragging right), ArrowLeft moves it left — 16px steps.
  function onResizerKey(which: "layers" | "props", e: KeyboardEvent) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const delta = e.key === "ArrowRight" ? -16 : 16;
    if (which === "layers") layersW = clampW(layersW + delta);
    else propsW = clampW(propsW + delta);
    try {
      localStorage.setItem(
        which === "layers" ? "glazer.layersW" : "glazer.propsW",
        String(which === "layers" ? layersW : propsW),
      );
    } catch {
      /* ignore */
    }
  }

  // Per-layer breakdown, limited to layers that are enabled for export.
  let breakdown = $derived.by(() => {
    if (!$estimate || !$project) return [];
    return $estimate.layers.filter((l) => {
      const layer = $project.layers.find((p) => p.id === l.layerId);
      return layer?.exportEnabled ?? false;
    });
  });
</script>

<div class="workbench">
  <Toolbar onFit={fit} />
  <div
    class="main"
    style="grid-template-columns: minmax(0,1fr) {PANEL_SEPERATION_WDITH_PX}px {layersW}px {PANEL_SEPERATION_WDITH_PX}px {propsW}px"
  >
    <div class="canvas-wrap">
      <CanvasView bind:this={canvas} />
      {#if $estimate}
        <div class="overlay">
          <div class="overlay-head">
            <button class="overlay-toggle" onclick={() => (estimateExpanded = !estimateExpanded)}>
              Est. Total Time: {fmtMin($estimate.totalMinutes)}
              {estimateExpanded ? "▾" : "▸"}
            </button>
            <Tip text={$estimate.assumptions.join("\n") || "No assumptions recorded."} />
          </div>
          {#if estimateExpanded && breakdown.length}
            <div class="overlay-breakdown">
              {#each breakdown as layer (layer.layerId)}
                <div class="breakdown-row">
                  <span>{layer.name} ×{layer.passes}</span>
                  <span>{fmtMin(layer.totalMinutes)}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
      {#if $validation?.warnings.length}
        <div class="overlay warn">{$validation.warnings[0]}</div>
      {/if}
    </div>
    <button
      type="button"
      class="resizer"
      aria-label="Resize layers panel"
      title="Drag or use arrow keys to resize the layers panel"
      onpointerdown={(e) => startResize("layers", e)}
      onkeydown={(e) => onResizerKey("layers", e)}
    ></button>
    <LayerPanel />
    <button
      type="button"
      class="resizer"
      aria-label="Resize properties panel"
      title="Drag or use arrow keys to resize the properties panel"
      onpointerdown={(e) => startResize("props", e)}
      onkeydown={(e) => onResizerKey("props", e)}
    ></button>
    <PropertiesPanel />
  </div>
  <StatusBar />
</div>

<style>
  .canvas-wrap {
    min-width: 0;
    min-height: 0;
    position: relative;
  }
  .resizer {
    cursor: col-resize;
    background: transparent;
    touch-action: none;
    user-select: none;
  }
  .resizer:hover {
    background: var(--accent);
  }
  .overlay {
    position: absolute;
    left: 8px;
    top: 8px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    background: #121418cc;
    padding: 4px 8px;
    border-radius: 4px;
    max-width: calc(100% - 16px);
  }
  .overlay-head {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .overlay-toggle {
    background: transparent;
    border: none;
    color: var(--muted);
    padding: 0;
  }
  .overlay-toggle:hover {
    color: var(--text);
    border: none;
  }
  /* Expanded breakdown renders below the "est. total" line, full width. */
  .overlay-breakdown {
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-top: 1px solid var(--border);
    padding-top: 4px;
    width: 100%;
    font-size: 11px;
  }
  .breakdown-row {
    display: flex;
    gap: 10px;
    justify-content: space-between;
    color: var(--muted);
  }
</style>

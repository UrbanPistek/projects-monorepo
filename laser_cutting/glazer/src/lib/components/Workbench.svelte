<script lang="ts">
  // App layout: toolbar on top, canvas + layer panel + properties in the
  // middle, status bar at the bottom. The canvas overlay shows the estimated
  // total time with an expandable per-export-layer breakdown; estimator
  // assumptions live in the tooltip next to it.
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
  <div class="main">
    <div style="min-width:0;min-height:0;position:relative;">
      <CanvasView bind:this={canvas} />
      {#if $estimate}
        <div class="overlay">
          <button class="overlay-toggle" onclick={() => (estimateExpanded = !estimateExpanded)}>
            est. total {fmtMin($estimate.totalMinutes)} ({$estimate.confidence})
            {estimateExpanded ? "▾" : "▸"}
          </button>
          <Tip text={$estimate.assumptions.join("\n") || "No assumptions recorded."} />
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
    <LayerPanel />
    <PropertiesPanel />
  </div>
  <StatusBar />
</div>

<style>
  .overlay {
    position: absolute;
    left: 8px;
    bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: #121418cc;
    padding: 4px 8px;
    border-radius: 4px;
    max-width: calc(100% - 16px);
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
  .overlay-breakdown {
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-left: 1px solid var(--border);
    padding-left: 8px;
    font-size: 11px;
  }
  .breakdown-row {
    display: flex;
    gap: 10px;
    justify-content: space-between;
    color: var(--muted);
  }
</style>

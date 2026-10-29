<script lang="ts">
  import CanvasView from "./CanvasView.svelte";
  import LayerPanel from "./LayerPanel.svelte";
  import PropertiesPanel from "./PropertiesPanel.svelte";
  import Toolbar from "./Toolbar.svelte";
  import StatusBar from "./StatusBar.svelte";
  import { estimate, validation } from "$lib/stores/session";

  let canvas: CanvasView;

  function fit() {
    canvas?.fit();
  }
</script>

<div class="workbench">
  <Toolbar onFit={fit} />
  <div class="main">
    <div style="min-width:0;min-height:0;position:relative;">
      <CanvasView bind:this={canvas} />
      {#if $estimate}
        <div class="overlay muted">
          {$estimate.assumptions[0]}
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
    background: #121418cc;
    padding: 4px 8px;
    border-radius: 4px;
    pointer-events: none;
  }
</style>

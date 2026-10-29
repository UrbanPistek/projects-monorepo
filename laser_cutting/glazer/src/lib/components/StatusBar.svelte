<script lang="ts">
  import { summary, estimate, validation, selectedIds, preview } from "$lib/stores/session";

  function fmtMin(v: number | null | undefined) {
    if (v == null) return "n/a";
    const s = v * 60;
    if (s < 60) return `${s.toFixed(1)}s`;
    return `${Math.floor(v)}m ${((v % 1) * 60).toFixed(0)}s`;
  }
</script>

<footer class="status">
  {#if $summary}
    <span>{$summary.fileName}</span>
    <span>{$summary.objectCount} objects</span>
    <span>{$summary.layerCount} layers</span>
    <span>
      {($summary.bounds.max.x - $summary.bounds.min.x).toFixed(2)} ×
      {($summary.bounds.max.y - $summary.bounds.min.y).toFixed(2)} mm
    </span>
    <span>units: {$summary.units.certainty}</span>
    <span>selected {$selectedIds.length}</span>
    {#if $estimate}
      <span>cut {fmtMin($estimate.cutMinutes)}</span>
      <span>est. total {fmtMin($estimate.totalMinutes)} ({$estimate.confidence})</span>
      <span>paths {$estimate.pathCount}</span>
    {/if}
    <span>{$summary.profileId}</span>
    {#if $validation && !$validation.ok}
      <span class="error">{$validation.errors[0]}</span>
    {:else if $summary.warnings.length}
      <span class="warn">{$summary.warnings.length} import warning(s)</span>
    {/if}
  {:else}
    <span>Open an SVG or DXF to begin. Preview, layers, estimate, and G-code stay in millimetres.</span>
  {/if}
  {#if $preview}
    <span class="muted">wheel zoom · drag select · right-drag pan</span>
  {/if}
</footer>

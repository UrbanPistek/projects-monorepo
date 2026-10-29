<script lang="ts">
  import { project, selectedIds, run } from "$lib/stores/session";
  import * as api from "$lib/api";

  function hex(c: { r: number; g: number; b: number }) {
    return `#${c.r.toString(16).padStart(2, "0")}${c.g.toString(16).padStart(2, "0")}${c.b.toString(16).padStart(2, "0")}`;
  }

  async function toggle(layer: (typeof $project.layers)[number], field: "visible" | "exportEnabled") {
    const next = { ...layer, [field]: !layer[field] };
    await run(() => api.updateLayer(next));
  }

  async function assignSelected(layerId: string) {
    const ids = $selectedIds;
    if (!ids.length) return;
    await run(() => api.assignObjects(ids, layerId));
  }

  async function move(id: string, dir: -1 | 1) {
    if (!$project) return;
    const ordered = [...$project.layers].sort((a, b) => a.order - b.order).map((l) => l.id);
    const i = ordered.indexOf(id);
    const j = i + dir;
    if (j < 0 || j >= ordered.length) return;
    [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
    await run(() => api.reorderLayers(ordered));
  }
</script>

<section class="panel">
  <h2>Layers</h2>
  {#if $project}
    {#each [...$project.layers].sort((a, b) => a.order - b.order) as layer (layer.id)}
      {@const count = $project.document.objects.filter((o) => o.layerId === layer.id).length}
      <div class="layer-row">
        <button title="Visible" onclick={() => toggle(layer, "visible")}>{layer.visible ? "●" : "○"}</button>
        <button title="Export" onclick={() => toggle(layer, "exportEnabled")}>{layer.exportEnabled ? "E" : "–"}</button>
        <span class="swatch" style="background:{hex(layer.displayColor)}"></span>
        <button class="name" onclick={() => assignSelected(layer.id)}>{layer.name} ({count})</button>
        <span>
          <button onclick={() => move(layer.id, -1)}>↑</button>
          <button onclick={() => move(layer.id, 1)}>↓</button>
        </span>
      </div>
    {/each}
    <div class="field muted">
      Click a layer name to assign the current selection. E = export enabled; visibility does not change export.
    </div>
  {:else}
    <div class="field muted">Import a file to create layers.</div>
  {/if}
</section>

<style>
  .name {
    text-align: left;
    background: transparent;
    border: none;
  }
</style>

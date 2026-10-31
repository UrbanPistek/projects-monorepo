<script lang="ts">
  // Layer stack panel. Each row keeps its visibility (●), export (E) and
  // order (↑↓) toggles; clicking the name selects the layer for the
  // Properties panel, "→" assigns the selected canvas paths to it, and 🗑
  // deletes it (paths move to the grey Unassigned layer, which cannot be
  // deleted). The frame layer always sits first (order 0).
  import { project, selectedIds, selectedLayerId, run } from "$lib/stores/session";
  import * as api from "$lib/api";
  import { isUnassignedRule, type ProcessLayer } from "$lib/api";

  function hex(c: { r: number; g: number; b: number }) {
    return `#${c.r.toString(16).padStart(2, "0")}${c.g.toString(16).padStart(2, "0")}${c.b.toString(16).padStart(2, "0")}`;
  }

  async function toggle(layer: ProcessLayer, field: "visible" | "exportEnabled") {
    const next = { ...layer, [field]: !layer[field] };
    await run(() => api.updateLayer(next));
  }

  // Clicking the layer name selects the layer so the Properties panel shows
  // only this layer's settings.
  function select(layer: ProcessLayer) {
    selectedLayerId.set(layer.id);
  }

  // The "→" button moves the current canvas selection into this layer.
  async function assignSelected(layerId: string) {
    const ids = $selectedIds;
    if (!ids.length) return;
    await run(() => api.assignObjects(ids, layerId));
  }

  async function remove(layer: ProcessLayer) {
    if (isUnassignedRule(layer.sourceAssignment)) return;
    await run(async () => {
      await api.deleteLayer(layer.id);
      if ($selectedLayerId === layer.id) selectedLayerId.set(null);
    });
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

  // A layer row is highlighted when it is the selected layer or when it owns
  // one of the currently selected paths on the canvas.
  function isActive(layer: ProcessLayer): boolean {
    if ($selectedLayerId === layer.id) return true;
    return $project?.document.objects.some(
      (o) => o.layerId === layer.id && $selectedIds.includes(o.id),
    ) ?? false;
  }
</script>

<section class="panel">
  <h2>Layers</h2>
  {#if $project}
    {#each [...$project.layers].sort((a, b) => a.order - b.order) as layer (layer.id)}
      {@const count = $project.document.objects.filter((o) => o.layerId === layer.id).length}
      {@const unassigned = isUnassignedRule(layer.sourceAssignment)}
      <div class="layer-row" class:active={isActive(layer)}>
        <button title="Toggle visibility" onclick={() => toggle(layer, "visible")}>{layer.visible ? "●" : "○"}</button>
        <button title="Toggle export" onclick={() => toggle(layer, "exportEnabled")}>{layer.exportEnabled ? "E" : "–"}</button>
        <span class="swatch" style="background:{hex(layer.displayColor)}"></span>
        <button class="name" title="Select layer (shows its properties)" onclick={() => select(layer)}>
          {layer.name} ({count})
        </button>
        <span class="row-actions">
          <button title="Assign selected paths to this layer" onclick={() => assignSelected(layer.id)}>→</button>
          <button title={unassigned ? "The Unassigned layer cannot be deleted" : "Delete layer (paths move to Unassigned)"} disabled={unassigned} onclick={() => remove(layer)}>🗑</button>
          <button onclick={() => move(layer.id, -1)}>↑</button>
          <button onclick={() => move(layer.id, 1)}>↓</button>
        </span>
      </div>
    {/each}
    <div class="field muted">
      Click a layer name to edit its properties. → assigns the selected paths to that layer.
      E = export enabled; visibility does not change export. Deleting a layer moves its paths to grey Unassigned.
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
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-actions {
    display: inline-flex;
    gap: 2px;
  }
  .row-actions button {
    padding: 2px 5px;
    font-size: 11px;
  }
</style>

<script lang="ts">
  // Layer stack panel. Each row keeps its visibility (V), export (E) and
  // order (↑↓) toggles; clicking the name selects the layer for the
  // Properties panel, "→" assigns the selected canvas paths to it, and "D"
  // deletes it (paths move to the grey Unassigned layer, which has no delete
  // button and cannot be deleted). The frame layer always sits first
  // (order 0). "+ New layer" appends an empty layer.
  import { project, selectedIds, selectedLayerId, run } from "$lib/stores/session";
  import * as api from "$lib/api";
  import { isUnassignedRule, type ProcessLayer, type Rgba8 } from "$lib/api";

  // Colors cycled through when creating new layers.
  const NEW_LAYER_COLORS: Rgba8[] = [
    { r: 255, g: 0, b: 0, a: 255 },
    { r: 0, g: 255, b: 0, a: 255 },
    { r: 0, g: 0, b: 255, a: 255 },
    { r: 255, g: 255, b: 0, a: 255 },
    { r: 255, g: 0, b: 255, a: 255 },
    { r: 0, g: 255, b: 255, a: 255 },
    { r: 255, g: 165, b: 0, a: 255 },
    { r: 160, g: 32, b: 240, a: 255 },
  ];

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

  // "+ New layer": append an empty layer, auto-named and auto-colored, and
  // select it so its properties can be edited immediately.
  async function addLayer() {
    if (!$project) return;
    let n = $project.layers.length + 1;
    while ($project.layers.some((l) => l.name === `Layer ${n}`)) n++;
    const color = NEW_LAYER_COLORS[($project.layers.length - 1) % NEW_LAYER_COLORS.length];
    await run(async () => {
      const id = await api.createLayer(`Layer ${n}`, color);
      selectedLayerId.set(id);
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
  <h2>Layers <button class="new-layer" title="Create a new empty layer" onclick={addLayer}>+ New layer</button></h2>
  {#if $project}
    {#each [...$project.layers].sort((a, b) => a.order - b.order) as layer (layer.id)}
      {@const count = $project.document.objects.filter((o) => o.layerId === layer.id).length}
      {@const unassigned = isUnassignedRule(layer.sourceAssignment)}
      <div class="layer-row" class:active={isActive(layer)}>
        <button class="icon-btn" class:off={!layer.visible} title="Toggle visibility" onclick={() => toggle(layer, "visible")}>V</button>
        <button class="icon-btn" class:off={!layer.exportEnabled} title="Toggle export" onclick={() => toggle(layer, "exportEnabled")}>E</button>
        <span class="swatch" style="background:{hex(layer.displayColor)}"></span>
        <button class="name" title="Select layer (shows its properties)" onclick={() => select(layer)}>
          {layer.name} ({count})
        </button>
        <span class="row-actions">
          <button class="icon-btn" title="Assign selected paths to this layer" onclick={() => assignSelected(layer.id)}>→</button>
          {#if unassigned}
            <span class="icon-btn placeholder" title="The Unassigned layer cannot be deleted"></span>
          {:else}
            <button class="icon-btn" title="Delete layer (paths move to Unassigned)" onclick={() => remove(layer)}>D</button>
          {/if}
          <button class="icon-btn" title="Move up" onclick={() => move(layer.id, -1)}>↑</button>
          <button class="icon-btn" title="Move down" onclick={() => move(layer.id, 1)}>↓</button>
        </span>
      </div>
    {/each}
    <div class="field muted">
      Click a layer name to edit its properties.
    </div>
    <div class="field muted">
      → assigns the selected paths to that layer.
    </div>
    <div class="field muted">
      V = visible, E = export enabled (visibility does not change export). D deletes the layer and moves its paths to grey Unassigned.
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
  .new-layer {
    font-size: 11px;
    padding: 2px 6px;
    text-transform: none;
    letter-spacing: normal;
  }
  .row-actions {
    display: inline-flex;
    gap: 2px;
  }
</style>

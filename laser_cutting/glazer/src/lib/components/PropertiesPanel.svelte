<script lang="ts">
  import { project, selectedIds, run } from "$lib/stores/session";
  import * as api from "$lib/api";
  import type { ProcessLayer } from "$lib/api";

  let selectedLayer = $derived.by(() => {
    const p = $project;
    const ids = $selectedIds;
    if (!p) return p?.layers[0];
    const obj = p.document.objects.find((o) => ids.includes(o.id));
    if (obj?.layerId) return p.layers.find((l) => l.id === obj.layerId);
    return [...p.layers].sort((a, b) => a.order - b.order)[0];
  });

  async function patch(partial: Partial<ProcessLayer>) {
    if (!selectedLayer) return;
    await run(() => api.updateLayer({ ...selectedLayer, ...partial }));
  }
</script>

<section class="panel">
  <h2>Properties</h2>
  {#if selectedLayer}
    <label class="field">Name
      <input value={selectedLayer.name} onchange={(e) => patch({ name: e.currentTarget.value })} />
    </label>
    <label class="field">Operation
      <select value={selectedLayer.operation} onchange={(e) => patch({ operation: e.currentTarget.value as ProcessLayer["operation"] })}>
        <option value="cut">Cut</option>
        <option value="engraveLine">Engrave line</option>
        <option value="frame">Frame</option>
        <option value="travelOnly">Travel only</option>
      </select>
    </label>
    <label class="field">Speed (mm/min)
      <input type="number" value={selectedLayer.speedMmMin} onchange={(e) => patch({ speedMmMin: Number(e.currentTarget.value) })} />
    </label>
    <label class="field">Power (%)
      <input type="number" min="0" max="100" value={selectedLayer.powerPercent} onchange={(e) => patch({ powerPercent: Number(e.currentTarget.value) })} />
    </label>
    <label class="field">Passes
      <input type="number" min="1" value={selectedLayer.passes} onchange={(e) => patch({ passes: Number(e.currentTarget.value) })} />
    </label>
    <label class="field">Laser mode
      <select value={selectedLayer.laserMode} onchange={(e) => patch({ laserMode: e.currentTarget.value as ProcessLayer["laserMode"] })}>
        <option value="dynamic">Dynamic (M4)</option>
        <option value="constant">Constant (M3)</option>
      </select>
    </label>
    {#if selectedLayer.operation === "frame"}
      <label class="field">Frame margin (mm)
        <input type="number" value={selectedLayer.frameMarginMm} onchange={(e) => patch({ frameMarginMm: Number(e.currentTarget.value) })} />
      </label>
    {/if}
  {/if}

  {#if $project}
    <h2>Job</h2>
    <label class="field">Scale
      <input type="number" step="0.01" value={$project.job.scale} onchange={(e) => run(() => api.updateJobSettings(Number(e.currentTarget.value), $project.job.rotationDeg, $project.job.chordErrorMm))} />
    </label>
    <label class="field">Rotation (deg)
      <input type="number" value={$project.job.rotationDeg} onchange={(e) => run(() => api.updateJobSettings($project.job.scale, Number(e.currentTarget.value), $project.job.chordErrorMm))} />
    </label>
    <label class="field">Chord error (mm)
      <input type="number" step="0.01" value={$project.job.chordErrorMm} onchange={(e) => run(() => api.updateJobSettings($project.job.scale, $project.job.rotationDeg, Number(e.currentTarget.value)))} />
    </label>
    <div class="field">
      Origin
      <button onclick={() => run(() => api.setOrigin("lowerLeft"))}>Lower-left</button>
      <button onclick={() => run(() => api.setOrigin("center"))}>Center</button>
      <button onclick={() => run(() => api.setOrigin("source"))}>Source</button>
    </div>
    <label class="field">Rapid feed (mm/min)
      <input type="number" value={$project.machine.rapidFeedMmMin ?? 0} onchange={(e) => run(() => api.updateMachine(Number(e.currentTarget.value), $project.machine.maxSValue))} />
    </label>
    <label class="field">Max S (GRBL $30)
      <input type="number" value={$project.machine.maxSValue} onchange={(e) => run(() => api.updateMachine($project.machine.rapidFeedMmMin, Number(e.currentTarget.value)))} />
    </label>
  {/if}

  {#if $selectedIds.length && $project}
    <h2>Selection</h2>
    <div class="field muted">{$selectedIds.length} object(s)</div>
    {#each $project.document.objects.filter((o) => $selectedIds.includes(o.id)).slice(0, 8) as obj}
      <div class="field">{obj.source.entityType} {obj.source.sourceId ?? obj.id.slice(0, 8)}</div>
    {/each}
    <div class="field">
      <button onclick={() => run(async () => {
        const obj = $project.document.objects.find((o) => $selectedIds.includes(o.id));
        if (obj?.style.stroke) await api.selectBySourceColor(obj.style.stroke);
      })}>Select same source color</button>
      <button onclick={() => run(() => api.createLayerFromSelection("Custom", { r: 200, g: 120, b: 40, a: 255 }, $selectedIds))}>Layer from selection</button>
    </div>
  {/if}
</section>

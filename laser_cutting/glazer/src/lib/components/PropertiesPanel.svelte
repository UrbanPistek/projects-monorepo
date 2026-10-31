<script lang="ts">
  // Properties panel. Shows the settings of exactly one layer — the one
  // clicked in the layer panel, else the layer of the selected path — and
  // every edit is sent as that layer's full struct, so changes apply only to
  // it. Frame layers are locked except passes and speed (their outline scales
  // with the job Scale in the Job section). Each field title has a hover
  // tooltip (Tip) explaining what it does. The Selection section can move
  // the selected paths to another layer.
  import { project, selectedIds, selectedLayerId, run } from "$lib/stores/session";
  import * as api from "$lib/api";
  import type { ProcessLayer } from "$lib/api";
  import Tip from "./Tip.svelte";

  // The layer shown in this panel: the one clicked in the layer panel, else
  // the layer of the selected path, else the first layer. Edits always go
  // through updateLayer with this exact layer, so they apply only to it.
  let selectedLayer = $derived.by(() => {
    const p = $project;
    if (!p) return undefined;
    if ($selectedLayerId) {
      const byClick = p.layers.find((l) => l.id === $selectedLayerId);
      if (byClick) return byClick;
    }
    const obj = p.document.objects.find((o) => $selectedIds.includes(o.id));
    if (obj?.layerId) return p.layers.find((l) => l.id === obj.layerId);
    return [...p.layers].sort((a, b) => a.order - b.order)[0];
  });

  // Frame layers (isFrame, operation travel-only) are fixed except passes,
  // speed and frame margin; their outline scales with the job-level Scale in
  // the Job section below. Locked fields render disabled + dimmed with 🔒.
  let isFrame = $derived(selectedLayer?.isFrame ?? false);
  // Travel-only layers never fire the laser either: power is forced to 0
  // (server-side too), so the S value for them is 0.
  let isLaserOff = $derived(isFrame || selectedLayer?.operation === "travelOnly");

  // Changing the operation to travel-only also zeroes the power (frame
  // layers are not selectable here — they are created via "Add frame").
  function changeOperation(op: ProcessLayer["operation"]) {
    if (op === "travelOnly") {
      patch({ operation: op, powerPercent: 0 });
    } else {
      patch({ operation: op });
    }
  }

  async function patch(partial: Partial<ProcessLayer>) {
    if (!selectedLayer) return;
    await run(() => api.updateLayer({ ...selectedLayer, ...partial }));
  }

  async function assignSelection(layerId: string) {
    if (!$selectedIds.length) return;
    await run(() => api.assignObjects($selectedIds, layerId));
  }
</script>

<section class="panel">
  <h2>Properties</h2>
  {#if selectedLayer}
    <div class="field muted">
      Layer: <strong>{selectedLayer.name}</strong>
      {#if isFrame}(frame — editable: passes, speed, margin; scale follows Job ▸ Scale){/if}
    </div>
    <label class="field" class:locked={isFrame}>
      <span class="field-title">Name
        <Tip text="Display name of this layer in the panel and in G-code comments. Only affects labelling." />
        {#if isFrame}<span class="lock">🔒</span>{/if}
      </span>
      <input disabled={isFrame} value={selectedLayer.name} onchange={(e) => patch({ name: e.currentTarget.value })} />
    </label>
    <label class="field" class:locked={isFrame}>
      <span class="field-title">Operation
        <Tip text="What the laser does for paths on this layer: cut/engrave with the laser on, or travel with the laser off (power is forced to 0). Frame layers are not created here — use the Add frame button in the toolbar." />
        {#if isFrame}<span class="lock">🔒</span>{/if}
      </span>
      <select disabled={isFrame} value={selectedLayer.operation} onchange={(e) => changeOperation(e.currentTarget.value as ProcessLayer["operation"])}>
        <option value="cut">Cut</option>
        <option value="engraveLine">Engrave line</option>
        <option value="travelOnly">Travel only</option>
      </select>
    </label>
    <label class="field">
      <span class="field-title">Speed (mm/min)
        <Tip text="Feed rate for this layer's moves (G-code F). Also the frame trace speed for frame layers. Lower speed = more energy per mm at the same power." />
      </span>
      <input type="number" min="1" value={selectedLayer.speedMmMin} onchange={(e) => patch({ speedMmMin: Number(e.currentTarget.value) })} />
    </label>
    <label class="field" class:locked={isLaserOff}>
      <span class="field-title">Power (%)
        <Tip text="Laser duty cycle as a percentage, scaled against the machine Max S (GRBL $30) when exporting — S = power% × Max S. Forced to 0 for frame and travel-only layers so they never fire the laser." />
        {#if isLaserOff}<span class="lock">🔒</span>{/if}
      </span>
      <input type="number" min="0" max="100" disabled={isLaserOff} value={selectedLayer.powerPercent} onchange={(e) => patch({ powerPercent: Number(e.currentTarget.value) })} />
    </label>
    <label class="field">
      <span class="field-title">Passes
        <Tip text="How many times each path on this layer is repeated. Useful for deep cuts or stronger engraving. Frame layers repeat the outline." />
      </span>
      <input type="number" min="1" value={selectedLayer.passes} onchange={(e) => patch({ passes: Number(e.currentTarget.value) })} />
    </label>
    <label class="field" class:locked={isFrame}>
      <span class="field-title">Laser mode
        <Tip text="Dynamic (M4) ramps power with speed — good for engraving. Constant (M3) holds power fixed — good for cutting. Fixed for frame layers (laser stays off)." />
        {#if isFrame}<span class="lock">🔒</span>{/if}
      </span>
      <select disabled={isFrame} value={selectedLayer.laserMode} onchange={(e) => patch({ laserMode: e.currentTarget.value as ProcessLayer["laserMode"] })}>
        <option value="dynamic">Dynamic (M4)</option>
        <option value="constant">Constant (M3)</option>
      </select>
    </label>
    {#if isFrame}
      <label class="field">
        <span class="field-title">Frame margin (mm)
          <Tip text="Distance between the drawing bounds and the frame outline. Editable; the dashed orange frame on the canvas follows this value. The outline scales with the job Scale below." />
        </span>
        <input type="number" min="0" step="0.5" value={selectedLayer.frameMarginMm} onchange={(e) => patch({ frameMarginMm: Number(e.currentTarget.value) })} />
      </label>
      <div class="field muted">Frame scale follows the job Scale in the Job section below.</div>
    {/if}
  {/if}

  {#if $project}
    <h2>Job</h2>
    <label class="field">
      <span class="field-title">Scale
        <Tip text="Scales the whole job (including the frame outline) around the origin. 1 = original size." />
      </span>
      <input type="number" step="0.01" value={$project.job.scale} onchange={(e) => run(() => api.updateJobSettings(Number(e.currentTarget.value), $project.job.rotationDeg, $project.job.chordErrorMm))} />
    </label>
    <label class="field">
      <span class="field-title">Rotation (deg)
        <Tip text="Rotates the job around the origin in degrees, counter-clockwise." />
      </span>
      <input type="number" value={$project.job.rotationDeg} onchange={(e) => run(() => api.updateJobSettings($project.job.scale, Number(e.currentTarget.value), $project.job.chordErrorMm))} />
    </label>
    <label class="field">
      <span class="field-title">Chord error (mm)
        <Tip text="Maximum deviation when curves are flattened into line segments for the preview and G-code. Smaller = smoother curves but more segments." />
      </span>
      <input type="number" step="0.01" value={$project.job.chordErrorMm} onchange={(e) => run(() => api.updateJobSettings($project.job.scale, $project.job.rotationDeg, Number(e.currentTarget.value)))} />
    </label>
    <div class="field">
      <span class="field-title">Origin
        <Tip text="Where (0,0) of the machine sits relative to the drawing: source file coordinates, lower-left corner, or centre." />
      </span>
      <span>
        <button onclick={() => run(() => api.setOrigin("lowerLeft"))}>Lower-left</button>
        <button onclick={() => run(() => api.setOrigin("center"))}>Center</button>
        <button onclick={() => run(() => api.setOrigin("source"))}>Source</button>
      </span>
    </div>
    <label class="field">
      <span class="field-title">Rapid feed (mm/min)
        <Tip text="Feed used for travel moves with the laser off (G0). Used to estimate travel time; leave 0 if unknown (travel is then excluded from the total)." />
      </span>
      <input type="number" value={$project.machine.rapidFeedMmMin ?? 0} onchange={(e) => run(() => api.updateMachine(Number(e.currentTarget.value), $project.machine.maxSValue))} />
    </label>
    <label class="field">
      <span class="field-title">Max S (GRBL $30)
        <Tip text="The S value that means 100% power on your controller. Power % is scaled against this when exporting." />
      </span>
      <input type="number" value={$project.machine.maxSValue} onchange={(e) => run(() => api.updateMachine($project.machine.rapidFeedMmMin, Number(e.currentTarget.value)))} />
    </label>
  {/if}

  {#if $selectedIds.length && $project}
    <h2>Selection</h2>
    <div class="field muted">{$selectedIds.length} object(s)</div>
    {#each $project.document.objects.filter((o) => $selectedIds.includes(o.id)).slice(0, 8) as obj}
      <div class="field">{obj.source.entityType} {obj.source.sourceId ?? obj.id.slice(0, 8)}</div>
    {/each}
    <label class="field">
      <span class="field-title">Assign to layer
        <Tip text="Move the selected path(s) to another layer. Layers are process containers — a layer can hold paths of different colours." />
      </span>
      <select value={$project.document.objects.find((o) => $selectedIds.includes(o.id))?.layerId ?? ""} onchange={(e) => assignSelection(e.currentTarget.value)}>
        {#each [...$project.layers].sort((a, b) => a.order - b.order) as layer}
          <option value={layer.id}>{layer.name}</option>
        {/each}
      </select>
    </label>
    <div class="field">
      <button onclick={() => run(async () => {
        const obj = $project.document.objects.find((o) => $selectedIds.includes(o.id));
        if (obj?.style.stroke) await api.selectBySourceColor(obj.style.stroke);
      })}>Select same source color</button>
      <button onclick={() => run(() => api.createLayerFromSelection("Custom", { r: 200, g: 120, b: 40, a: 255 }, $selectedIds))}>Layer from selection</button>
    </div>
  {/if}
</section>

<style>
  .field-title {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  /* Locked (fixed) fields on frame layers: visibly disabled. */
  .field.locked .field-title {
    color: var(--muted);
  }
  .field.locked input,
  .field.locked select {
    opacity: 0.4;
  }
  .lock {
    font-size: 10px;
  }
</style>

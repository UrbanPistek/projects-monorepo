<script lang="ts">
  // Top toolbar: file open/save/export, fit, "Assign by color" (regroups all
  // paths into one layer per source colour), Add frame, the grid background
  // toggle (dark / off-white) and the unit assumption selector.
  import * as api from "$lib/api";
  import { run, errorMessage, busy, project } from "$lib/stores/session";
  import type { LengthUnit } from "$lib/api";

  let { onFit }: { onFit: () => void } = $props();
  let assumed: LengthUnit = $state("millimeter");

  async function afterImport() {
    await run(async () => {
      await api.pickAndImport();
    });
    onFit();
  }

  // Toggle the grid background between dark and off-white "paper".
  async function toggleGridBackground() {
    if (!$project) return;
    const next = $project.display.gridBackground === "offWhite" ? "dark" : "offWhite";
    await run(() => api.setGridBackground(next));
  }
</script>

<header class="toolbar">
  <img src="/glazer-icon.png" class="logo" alt="Glazer" />
  <strong>Glazer</strong>
  <button class="primary" disabled={$busy} onclick={afterImport}>Open SVG/DXF</button>
  <button disabled={$busy} onclick={() => run(() => api.pickAndOpenProject())}>Open project</button>
  <button disabled={$busy} onclick={() => run(() => api.pickAndSaveProject())}>Save project</button>
  <button onclick={onFit}>Fit</button>
  <button disabled={$busy} title="Create one layer per source colour and assign all matching paths" onclick={() => run(() => api.autoAssign(true))}>Assign by color</button>
  <button disabled={$busy} onclick={() => run(() => api.createFrame(2))}>Add frame</button>
  <button class="primary" disabled={$busy} onclick={() => run(() => api.pickAndExportGcode())}>Export G-code</button>
  <button disabled={!$project || $busy} title="Toggle grid background (dark / off-white)" onclick={toggleGridBackground}>
    Grid: {$project?.display.gridBackground === "offWhite" ? "off-white" : "dark"}
  </button>
  <span class="muted">Assume units:</span>
  <select bind:value={assumed} onchange={() => run(() => api.assumeUnits(assumed))}>
    <option value="millimeter">mm</option>
    <option value="inch">in</option>
    <option value="cssPx">px</option>
  </select>
  {#if $errorMessage}
    <span class="error">{$errorMessage}</span>
  {/if}
</header>

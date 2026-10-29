<script lang="ts">
  import * as api from "$lib/api";
  import { run, errorMessage, busy } from "$lib/stores/session";
  import type { LengthUnit } from "$lib/api";

  let { onFit }: { onFit: () => void } = $props();
  let assumed: LengthUnit = $state("millimeter");

  async function afterImport() {
    await run(async () => {
      await api.pickAndImport();
    });
    onFit();
  }
</script>

<header class="toolbar">
  <strong>Glazer</strong>
  <button class="primary" disabled={$busy} onclick={afterImport}>Open SVG/DXF</button>
  <button disabled={$busy} onclick={() => run(() => api.pickAndOpenProject())}>Open project</button>
  <button disabled={$busy} onclick={() => run(() => api.pickAndSaveProject())}>Save project</button>
  <button onclick={onFit}>Fit</button>
  <button disabled={$busy} onclick={() => run(() => api.autoAssign(true))}>Assign by color</button>
  <button disabled={$busy} onclick={() => run(() => api.createFrame(2))}>Add frame</button>
  <button class="primary" disabled={$busy} onclick={() => run(() => api.pickAndExportGcode())}>Export G-code</button>
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

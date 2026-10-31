<script lang="ts">
  // Small ⓘ button that reveals a tooltip bubble. The bubble is positioned
  // with `position: fixed` from the icon's bounding rect on hover/focus, so
  // it renders on the topmost layer and is never clipped by a panel's
  // `overflow: auto` (which is what clipped the old in-panel bubble).
  // The button stays keyboard-focusable; mousedown is suppressed so clicking
  // it inside a <label> doesn't steal focus from the input.
  let { text }: { text: string } = $props();

  let pos = $state<{ left: number; top: number; above: boolean } | null>(null);

  function place(e: Event) {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const width = 230;
    // Keep the bubble inside the viewport horizontally.
    const left = Math.min(Math.max(8, r.left + r.width / 2 - width / 2), window.innerWidth - width - 8);
    // Flip below the icon when there is not enough room above.
    const above = r.top > 170;
    pos = { left, top: above ? r.top - 8 : r.bottom + 8, above };
  }

  function hide() {
    pos = null;
  }
</script>

<button
  type="button"
  class="tip"
  aria-label={text}
  onmousedown={(e) => e.preventDefault()}
  onmouseenter={place}
  onfocus={place}
  onmouseleave={hide}
  onblur={hide}
>ⓘ{#if pos}<span
      class="bubble"
      style="left:{pos.left}px; top:{pos.top}px; transform:{pos.above ? 'translateY(-100%)' : 'none'}"
      >{text}</span
    >{/if}</button>

<style>
  .tip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid var(--muted);
    background: transparent;
    color: var(--muted);
    font-size: 10px;
    cursor: help;
    padding: 0;
    flex: none;
  }
  .tip:hover,
  .tip:focus {
    border-color: var(--accent);
    color: var(--text);
    outline: none;
  }
  .bubble {
    position: fixed;
    z-index: 1000;
    width: 230px;
    padding: 6px 8px;
    background: var(--panel-2);
    border: 1px solid var(--accent);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    line-height: 1.4;
    white-space: pre-line;
    text-align: left;
    pointer-events: none;
    box-shadow: 0 4px 14px #000a;
  }
</style>

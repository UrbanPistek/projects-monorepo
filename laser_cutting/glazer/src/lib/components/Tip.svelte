<script lang="ts">
  // Small ⓘ button that reveals a hover/focus tooltip bubble. Used on
  // property titles (and the estimate overlay) so each control can explain
  // what it does and why. Pure CSS bubble; the button keeps it keyboard
  // focusable, and mousedown is suppressed so clicking it inside a <label>
  // doesn't steal focus from the input.
  let { text }: { text: string } = $props();
</script>

<button
  type="button"
  class="tip"
  aria-label={text}
  onmousedown={(e) => e.preventDefault()}
>ⓘ<span class="bubble">{text}</span></button>

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
  .tip:hover {
    border-color: var(--accent);
    color: var(--text);
  }
  .bubble {
    display: none;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    width: 230px;
    padding: 6px 8px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    line-height: 1.4;
    white-space: pre-line;
    text-align: left;
    z-index: 30;
    pointer-events: none;
    box-shadow: 0 4px 14px #0009;
  }
  .tip:hover .bubble,
  .tip:focus .bubble {
    display: block;
  }
</style>

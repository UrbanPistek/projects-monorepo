// Small shared formatting helpers: durations for the estimate UI, and
// mm <-> display-unit conversion for the canvas axis labels (mirrors
// laser_core::LengthUnit::to_mm_factor on the Rust side).

/** Format a duration in minutes as e.g. `42s` or `3m 05s`. */
export function fmtMin(v: number | null | undefined): string {
  if (v == null) return "n/a";
  const s = v * 60;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(v)}m ${((v % 1) * 60).toFixed(0)}s`;
}

/** mm-per-unit conversion factors, mirroring laser_core::LengthUnit::to_mm_factor. */
const MM_PER_UNIT: Record<string, number> = {
  millimeter: 1,
  centimeter: 10,
  meter: 1000,
  inch: 25.4,
  foot: 304.8,
  point: 25.4 / 72,
  pica: 25.4 / 6,
  cssPx: 25.4 / 96,
};

const UNIT_LABEL: Record<string, string> = {
  millimeter: "mm",
  centimeter: "cm",
  meter: "m",
  inch: "in",
  foot: "ft",
  point: "pt",
  pica: "pc",
  cssPx: "px",
};

/** Convert a millimetre value to the display unit; falls back to mm when unknown. */
export function mmToDisplay(mm: number, unit: string | null | undefined): number {
  const factor = (unit && MM_PER_UNIT[unit]) || 1;
  return mm / factor;
}

export function unitLabel(unit: string | null | undefined): string {
  return (unit && UNIT_LABEL[unit]) || "mm";
}

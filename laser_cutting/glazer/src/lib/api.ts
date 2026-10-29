import { invoke } from "@tauri-apps/api/core";

export type LengthUnit =
  | "millimeter"
  | "centimeter"
  | "meter"
  | "inch"
  | "foot"
  | "point"
  | "pica"
  | "cssPx"
  | "unknown";

export type OriginMode = "source" | "lowerLeft" | "center";

export interface Rgba8 {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface Bounds2D {
  min: { x: number; y: number };
  max: { x: number; y: number };
}

export interface ImportWarning {
  code: string;
  message: string;
  entity: string | null;
}

export interface UnitMetadata {
  sourceUnit: LengthUnit;
  certainty: "known" | "assumed" | "ambiguous";
  assumedUnit: LengthUnit | null;
  notes: string[];
}

export interface ProjectSummary {
  fileName: string;
  objectCount: number;
  layerCount: number;
  bounds: Bounds2D;
  units: UnitMetadata;
  warnings: ImportWarning[];
  profileId: string;
}

export interface PreviewObject {
  id: string;
  layerId: string | null;
  visible: boolean;
  exportEnabled: boolean;
  color: string;
  polylines: number[][][];
  sourceId: string | null;
  entityType: string;
  lengthMm: number;
}

export interface PreviewGeometry {
  objects: PreviewObject[];
  bounds: Bounds2D;
  documentBounds: Bounds2D;
}

export interface ProcessLayer {
  id: string;
  name: string;
  displayColor: Rgba8;
  order: number;
  visible: boolean;
  exportEnabled: boolean;
  operation: "cut" | "engraveLine" | "frame" | "travelOnly";
  speedMmMin: number;
  powerPercent: number;
  passes: number;
  laserMode: "constant" | "dynamic";
  sourceAssignment: unknown;
  laserMustRemainOff: boolean;
  frameMarginMm: number;
}

export interface Project {
  schemaVersion: number;
  appVersion: string;
  document: {
    objects: Array<{
      id: string;
      layerId: string | null;
      visible: boolean;
      exportEnabled: boolean;
      style: { stroke: Rgba8 | null; fill: Rgba8 | null };
      source: { sourceId: string | null; entityType: string; fileName: string };
    }>;
    units: UnitMetadata;
    boundsMm: Bounds2D;
    importWarnings: ImportWarning[];
    sourceFileName: string;
  };
  layers: ProcessLayer[];
  job: {
    scale: number;
    rotationDeg: number;
    offsetMm: { x: number; y: number };
    originMode: OriginMode;
    chordErrorMm: number;
  };
  machine: {
    name: string;
    rapidFeedMmMin: number | null;
    motionOverheadS: number;
    maxSValue: number;
  };
  export: { profileId: string; includeFrameInEstimate: boolean };
  display: { gridEnabled: boolean; colorTolerance: number };
}

export interface EstimateReport {
  totalMinutes: number | null;
  cutMinutes: number;
  travelMinutes: number | null;
  cutLengthMm: number;
  travelLengthMm: number;
  pathCount: number;
  segmentCount: number;
  confidence: string;
  assumptions: string[];
  warnings: string[];
  layers: Array<{
    name: string;
    cutLengthMm: number;
    travelLengthMm: number;
    pathCount: number;
    passes: number;
  }>;
}

export interface ValidationReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

export interface ColorAssignmentPreview {
  matches: Array<{
    layerName: string;
    color: Rgba8;
    objectIds: string[];
    count: number;
  }>;
  unmatched: Array<{ color: Rgba8; objectIds: string[] }>;
}

export async function pickAndImport(assumedUnit?: LengthUnit) {
  return invoke<ProjectSummary>("pick_and_import", { assumedUnit: assumedUnit ?? null });
}

export async function pickAndOpenProject() {
  return invoke<ProjectSummary>("pick_and_open_project");
}

export async function pickAndSaveProject() {
  return invoke<void>("pick_and_save_project");
}

export async function pickAndExportGcode() {
  return invoke<void>("pick_and_export_gcode");
}

export async function getProject() {
  return invoke<Project>("get_project");
}

export async function getPreview() {
  return invoke<PreviewGeometry>("get_preview_geometry");
}

export async function getSummary() {
  return invoke<ProjectSummary>("get_project_summary");
}

export async function estimateJob() {
  return invoke<EstimateReport>("estimate_job");
}

export async function validateJob() {
  return invoke<ValidationReport>("validate_job");
}

export async function updateLayer(layer: ProcessLayer) {
  return invoke<void>("update_layer", { layer });
}

export async function assignObjects(objectIds: string[], layerId: string) {
  return invoke<void>("assign_objects_to_layer", { objectIds, layerId });
}

export async function autoAssign(apply: boolean) {
  return invoke<ColorAssignmentPreview>("auto_assign_by_color", { apply });
}

export async function reorderLayers(orderedIds: string[]) {
  return invoke<void>("reorder_layers", { orderedIds });
}

export async function createFrame(marginMm: number) {
  return invoke<string>("create_frame_operation", { marginMm });
}

export async function setOrigin(mode: OriginMode) {
  return invoke<void>("set_job_origin", { mode });
}

export async function assumeUnits(unit: LengthUnit) {
  return invoke<void>("assume_units", { unit });
}

export async function createLayerFromSelection(name: string, color: Rgba8, objectIds: string[]) {
  return invoke<string>("create_layer_from_selection", { name, color, objectIds });
}

export async function selectBySourceColor(color: Rgba8) {
  return invoke<string[]>("select_by_source_color", { color });
}

export async function updateMachine(rapidFeedMmMin: number | null, maxSValue: number) {
  return invoke<void>("update_machine_rapid", { rapidFeedMmMin, maxSValue });
}

export async function updateJobSettings(scale: number, rotationDeg: number, chordErrorMm: number) {
  return invoke<void>("update_job_settings", { scale, rotationDeg, chordErrorMm });
}

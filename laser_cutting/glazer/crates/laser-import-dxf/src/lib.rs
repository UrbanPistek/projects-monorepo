//! # DXF importer
//!
//! Parses DXF (ASCII) with the `dxf` crate and maps entities (LINE, LWPOLYLINE,
//! POLYLINE, CIRCLE, ARC, ELLIPSE, SPLINE, ...) to canonical-millimetre paths.
//! Colours come from the AutoCAD Color Index via `aci_to_rgb`, and units from
//! `$INSUNITS` — which is often missing, hence the ambiguous-units handling in
//! `laser-core::units`.

use std::io::Cursor;

use dxf::entities::EntityType;
use dxf::enums::AcadVersion;
use dxf::Drawing;
use laser_core::{
    aci_to_rgb, Affine, Bounds2D, DocumentId, EntityCount, ImportReport, ImportResult,
    ImportWarning, ObjectId, PathGeometry, PathObject, PathSegment, PointMm,
    SourceFormat, SourceMetadata, SourceStyle, Subpath, UnitCertainty, UnitMetadata,
    VectorDocument, Rgba8,
};
use laser_import::{extension_is, ImportError, ImportOptions, VectorImporter};

pub struct DxfImporter;

impl VectorImporter for DxfImporter {
    fn can_import(&self, file_name: &str, sniffed_bytes: &[u8]) -> bool {
        if extension_is(file_name, "dxf") || extension_is(file_name, "dxb") {
            return true;
        }
        let head = String::from_utf8_lossy(sniffed_bytes);
        head.contains("SECTION") && head.contains("ENTITIES")
    }

    fn import(&self, input: &[u8], options: &ImportOptions) -> Result<ImportResult, ImportError> {
        import_dxf(input, options)
    }
}

pub fn import_dxf(input: &[u8], options: &ImportOptions) -> Result<ImportResult, ImportError> {
    let mut cursor = Cursor::new(input);
    let drawing = Drawing::load(&mut cursor).map_err(|e| ImportError::Failed(format!("DXF: {e}")))?;

    let insunits = drawing.header.default_drawing_units as i16;
    // dxf crate stores $INSUNITS on header; fall back to unitless if 0.
    let ins = drawing.header.insunits_value();
    let (detected, certainty, note) = laser_core::units::dxf_insunits(ins);
    let mut units = UnitMetadata {
        source_unit: detected,
        certainty,
        assumed_unit: None,
        notes: vec![note.into()],
    };
    if units.certainty != UnitCertainty::Known {
        if let Some(assumed) = options.assumed_unit {
            units = UnitMetadata::assumed(detected, assumed, format!("User assumed {}", assumed.label()));
        }
    }
    let unit = units.conversion_unit();
    let to_mm = unit.to_mm_factor().unwrap_or(1.0);
    let xf = Affine::scale(to_mm, to_mm);

    let mut warnings = Vec::new();
    if units.certainty == UnitCertainty::Ambiguous {
        warnings.push(ImportWarning::new(
            "ambiguous_units",
            "DXF units are missing or unsupported. Confirm units before export.",
        ));
    }
    let _ = (insunits, AcadVersion::R12);

    let mut objects = Vec::new();
    let mut imported = 0u32;
    let mut approximated = 0u32;
    let mut skipped = 0u32;
    let mut counts: Vec<(String, u32)> = Vec::new();

    for entity in drawing.entities() {
        let ty = entity_type_name(&entity.specific);
        bump(&mut counts, ty);
        let layer_name = entity.common.layer.clone();
        let layer = drawing.layers().find(|l| l.name == layer_name);
        let color = resolve_color(&entity.common, layer);
        match convert_entity(&entity.specific, xf) {
            Converted::Geom(geometry) => {
                imported += 1;
                objects.push(path_object(options, ty, layer_name, color, geometry));
            }
            Converted::Approx(geometry) => {
                imported += 1;
                approximated += 1;
                warnings.push(ImportWarning::new(
                    "approximated",
                    format!("{ty} was approximated"),
                ));
                objects.push(path_object(options, ty, layer_name, color, geometry));
            }
            Converted::Skip(reason) => {
                skipped += 1;
                warnings.push(ImportWarning::new("skipped", reason).with_entity(ty));
            }
        }
    }

    let mut document = VectorDocument {
        id: DocumentId::new(),
        units,
        bounds_mm: Bounds2D {
            min: PointMm::origin(),
            max: PointMm::origin(),
        },
        objects,
        import_warnings: warnings.clone(),
        source_to_canonical: xf,
        source_format: SourceFormat::Dxf,
        source_file_name: options.file_name.clone(),
    };
    document.recompute_bounds().ok();

    Ok(ImportResult {
        document: document.clone(),
        report: ImportReport {
            imported,
            approximated,
            skipped,
            unit_certainty: document.units.certainty,
            bounds_mm: document.bounds_mm,
            warnings,
            entity_counts: counts
                .into_iter()
                .map(|(entity_type, count)| EntityCount { entity_type, count })
                .collect(),
        },
    })
}

trait WithEntity {
    fn with_entity(self, entity: &str) -> Self;
}

impl WithEntity for ImportWarning {
    fn with_entity(mut self, entity: &str) -> Self {
        self.entity = Some(entity.into());
        self
    }
}

enum Converted {
    Geom(PathGeometry),
    Approx(PathGeometry),
    Skip(String),
}

fn path_object(
    options: &ImportOptions,
    ty: &str,
    layer: String,
    color: Option<Rgba8>,
    geometry: PathGeometry,
) -> PathObject {
    PathObject {
        id: ObjectId::new(),
        geometry,
        style: SourceStyle {
            stroke: color,
            fill: None,
            stroke_width_mm: None,
        },
        source: SourceMetadata {
            format: SourceFormat::Dxf,
            file_name: options.file_name.clone(),
            source_id: None,
            source_layer: Some(layer),
            entity_type: ty.into(),
            original_color: color,
        },
        layer_id: None,
        visible: true,
        export_enabled: true,
    }
}

fn convert_entity(entity: &EntityType, xf: Affine) -> Converted {
    match entity {
        EntityType::Line(line) => {
            let start = map_point(line.p1.x, line.p1.y, xf);
            let end = map_point(line.p2.x, line.p2.y, xf);
            Converted::Geom(PathGeometry {
                subpaths: vec![Subpath {
                    start,
                    segments: vec![PathSegment::LineTo { to: end }],
                    closed: false,
                }],
            })
        }
        EntityType::Circle(circle) => {
            let c = map_point(circle.center.x, circle.center.y, xf);
            let r = circle.radius * scale_of(xf);
            Converted::Geom(circle_geometry(c, r))
        }
        EntityType::Arc(arc) => {
            let c = map_point(arc.center.x, arc.center.y, xf);
            let r = arc.radius * scale_of(xf);
            let start = arc.start_angle.to_radians();
            let mut end = arc.end_angle.to_radians();
            if end <= start {
                end += std::f64::consts::TAU;
            }
            let sweep = end - start;
            let p0 = PointMm {
                x: c.x + r * start.cos(),
                y: c.y + r * start.sin(),
            };
            Converted::Geom(PathGeometry {
                subpaths: vec![Subpath {
                    start: p0,
                    segments: vec![PathSegment::Arc {
                        center: c,
                        radius: r,
                        start_angle_rad: start,
                        sweep_rad: sweep,
                    }],
                    closed: false,
                }],
            })
        }
        EntityType::LwPolyline(poly) => polyline_from_vertices(
            poly.vertices.iter().map(|v| (v.x, v.y, v.bulge)),
            poly.is_closed(),
            xf,
        ),
        EntityType::Polyline(poly) => polyline_from_vertices(
            poly.vertices().map(|v| (v.location.x, v.location.y, v.bulge)),
            poly.is_closed(),
            xf,
        ),
        EntityType::Ellipse(ell) => {
            let cx = ell.center.x;
            let cy = ell.center.y;
            let major = (ell.major_axis.x.hypot(ell.major_axis.y)).max(1e-9);
            let minor = major * ell.minor_axis_ratio.max(1e-9);
            let rot = ell.major_axis.y.atan2(ell.major_axis.x);
            Converted::Approx(ellipse_as_cubics(
                map_point(cx, cy, xf),
                major * scale_of(xf),
                minor * scale_of(xf),
                rot,
            ))
        }
        other => Converted::Skip(format!(
            "Unsupported DXF entity {} is not imported",
            entity_type_name(other)
        )),
    }
}

fn polyline_from_vertices(
    verts: impl Iterator<Item = (f64, f64, f64)>,
    closed: bool,
    xf: Affine,
) -> Converted {
    let verts: Vec<(f64, f64, f64)> = verts.collect();
    if verts.len() < 2 {
        return Converted::Skip("Polyline has fewer than two vertices".into());
    }
    let pts: Vec<PointMm> = verts
        .iter()
        .map(|(x, y, _)| map_point(*x, *y, xf))
        .collect();
    let mut segments = Vec::new();
    let n = verts.len();
    let count = if closed { n } else { n - 1 };
    for i in 0..count {
        let j = (i + 1) % n;
        let bulge = verts[i].2;
        if bulge.abs() < 1e-12 {
            segments.push(PathSegment::LineTo { to: pts[j] });
        } else {
            segments.push(bulge_to_arc(pts[i], pts[j], bulge));
        }
    }
    Converted::Geom(PathGeometry {
        subpaths: vec![Subpath {
            start: pts[0],
            segments,
            closed,
        }],
    })
}

fn bulge_to_arc(p1: PointMm, p2: PointMm, bulge: f64) -> PathSegment {
    let chord = p1.distance(p2);
    let sagitta = (bulge.abs() * chord) / 2.0;
    let radius = ((chord / 2.0).powi(2) + sagitta.powi(2)) / (2.0 * sagitta.max(1e-12));
    let sweep = 4.0 * bulge.atan();
    let mx = (p1.x + p2.x) * 0.5;
    let my = (p1.y + p2.y) * 0.5;
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
    let nx = -dy / len;
    let ny = dx / len;
    let d = radius - sagitta;
    let sign = if bulge >= 0.0 { 1.0 } else { -1.0 };
    let center = PointMm {
        x: mx + nx * d * sign,
        y: my + ny * d * sign,
    };
    let start_angle = (p1.y - center.y).atan2(p1.x - center.x);
    PathSegment::Arc {
        center,
        radius,
        start_angle_rad: start_angle,
        sweep_rad: sweep,
    }
}

fn circle_geometry(center: PointMm, radius: f64) -> PathGeometry {
    PathGeometry {
        subpaths: vec![Subpath {
            start: PointMm {
                x: center.x + radius,
                y: center.y,
            },
            segments: vec![PathSegment::Arc {
                center,
                radius,
                start_angle_rad: 0.0,
                sweep_rad: std::f64::consts::TAU,
            }],
            closed: true,
        }],
    }
}

fn ellipse_as_cubics(c: PointMm, rx: f64, ry: f64, rot: f64) -> PathGeometry {
    let k = 0.5522847498307936;
    let (sin, cos) = rot.sin_cos();
    let map = |x: f64, y: f64| {
        let xr = x * cos - y * sin;
        let yr = x * sin + y * cos;
        PointMm {
            x: c.x + xr,
            y: c.y + yr,
        }
    };
    let pts = [
        (rx, 0.0),
        (rx, ry * k),
        (rx * k, ry),
        (0.0, ry),
        (-rx * k, ry),
        (-rx, ry * k),
        (-rx, 0.0),
        (-rx, -ry * k),
        (-rx * k, -ry),
        (0.0, -ry),
        (rx * k, -ry),
        (rx, -ry * k),
        (rx, 0.0),
    ];
    let start = map(pts[0].0, pts[0].1);
    let mut segs = Vec::new();
    let mut i = 1;
    while i + 2 < pts.len() {
        segs.push(PathSegment::CubicTo {
            control1: map(pts[i].0, pts[i].1),
            control2: map(pts[i + 1].0, pts[i + 1].1),
            to: map(pts[i + 2].0, pts[i + 2].1),
        });
        i += 3;
    }
    PathGeometry {
        subpaths: vec![Subpath {
            start,
            segments: segs,
            closed: true,
        }],
    }
}

fn map_point(x: f64, y: f64, xf: Affine) -> PointMm {
    xf.apply(PointMm { x, y })
}

fn scale_of(xf: Affine) -> f64 {
    (xf.a * xf.a + xf.b * xf.b).sqrt()
}

fn bump(counts: &mut Vec<(String, u32)>, ty: &str) {
    if let Some((_, n)) = counts.iter_mut().find(|(k, _)| k == ty) {
        *n += 1;
    } else {
        counts.push((ty.into(), 1));
    }
}

fn entity_type_name(entity: &EntityType) -> &'static str {
    match entity {
        EntityType::Line(_) => "LINE",
        EntityType::Circle(_) => "CIRCLE",
        EntityType::Arc(_) => "ARC",
        EntityType::LwPolyline(_) => "LWPOLYLINE",
        EntityType::Polyline(_) => "POLYLINE",
        EntityType::Ellipse(_) => "ELLIPSE",
        EntityType::Spline(_) => "SPLINE",
        EntityType::Insert(_) => "INSERT",
        EntityType::Text(_) => "TEXT",
        EntityType::MText(_) => "MTEXT",
        _ => "OTHER",
    }
}

fn resolve_color(common: &dxf::entities::EntityCommon, layer: Option<&dxf::tables::Layer>) -> Option<Rgba8> {
    match common.color.index() {
        Some(i) => aci_to_rgb(i as i16),
        None => layer.and_then(|l| l.color.index().and_then(|i| aci_to_rgb(i as i16))),
    }
}

// Helper because dxf crate header field names vary across versions.
trait InsUnits {
    fn insunits_value(&self) -> i16;
}

impl InsUnits for dxf::Header {
    fn insunits_value(&self) -> i16 {
        // $INSUNITS lives as default_drawing_units in dxf 0.6
        self.default_drawing_units as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_dxf_mm() -> String {
        r#"  0
SECTION
  2
HEADER
  9
$INSUNITS
 70
     4
  0
ENDSEC
  0
SECTION
  2
ENTITIES
  0
LINE
  8
0
 62
     1
 10
0.0
 20
0.0
 30
0.0
 11
25.4
 21
0.0
 31
0.0
  0
ENDSEC
  0
EOF
"#
        .into()
    }

    #[test]
    fn imports_mm_line() {
        let result = DxfImporter
            .import(line_dxf_mm().as_bytes(), &ImportOptions::new("line.dxf"))
            .unwrap();
        assert_eq!(result.report.imported, 1);
        assert_eq!(result.document.units.certainty, UnitCertainty::Known);
        let b = result.document.bounds_mm;
        assert!((b.width() - 25.4).abs() < 1e-6);
    }
}

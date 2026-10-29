use laser_core::{
    Affine, Bounds2D, DocumentId, EntityCount, ImportReport, ImportResult, ImportWarning,
    LengthUnit, ObjectId, PathGeometry, PathObject, PathSegment, PointMm, SourceFormat,
    SourceMetadata, SourceStyle, Subpath, UnitCertainty, UnitMetadata, VectorDocument, Rgba8,
};
use laser_import::{extension_is, ImportError, ImportOptions, VectorImporter};
use usvg::{tiny_skia_path::PathSegment as UsvgSeg, Paint, Tree};

pub struct SvgImporter;

impl VectorImporter for SvgImporter {
    fn can_import(&self, file_name: &str, sniffed_bytes: &[u8]) -> bool {
        if extension_is(file_name, "svg") {
            return true;
        }
        let head = String::from_utf8_lossy(&sniffed_bytes[..sniffed_bytes.len().min(200)]);
        head.contains("<svg") || head.contains("<?xml")
    }

    fn import(&self, input: &[u8], options: &ImportOptions) -> Result<ImportResult, ImportError> {
        import_svg(input, options)
    }
}

pub fn import_svg(input: &[u8], options: &ImportOptions) -> Result<ImportResult, ImportError> {
    let unit_info = inspect_svg_units(input);
    let mut warnings = Vec::new();
    let mut units = unit_info;
    if units.certainty == UnitCertainty::Ambiguous {
        if let Some(assumed) = options.assumed_unit {
            units = UnitMetadata::assumed(
                LengthUnit::Unknown,
                assumed,
                format!("User assumed source units are {}", assumed.label()),
            );
        } else {
            warnings.push(ImportWarning::new(
                "ambiguous_units",
                "SVG has no physical width/height; scale is ambiguous. Confirm units before export.",
            ));
        }
    }

    let tree = Tree::from_data(input, &usvg::Options::default())
        .map_err(|e| ImportError::Failed(format!("SVG parse error: {e}")))?;

    let dpi = 96.0_f64;
    let px_to_mm = 25.4 / dpi;
    let width_mm = tree.size().width() as f64 * px_to_mm;
    let height_mm = tree.size().height() as f64 * px_to_mm;
    let y_up = Affine::svg_y_up(height_mm);
    let scale = Affine::scale(px_to_mm, px_to_mm);
    let source_to_canonical = y_up.then(scale);

    let mut objects = Vec::new();
    let mut imported = 0u32;
    let mut skipped = 0u32;
    let mut entity_counts = EntityCountMap::default();

    walk(
        tree.root(),
        &source_to_canonical,
        options,
        &mut objects,
        &mut warnings,
        &mut imported,
        &mut skipped,
        &mut entity_counts,
    );

    if tree.root().has_children() == false && objects.is_empty() {
        warnings.push(ImportWarning::new("empty", "SVG contained no drawable paths"));
    }

    let mut document = VectorDocument {
        id: DocumentId::new(),
        units,
        bounds_mm: Bounds2D {
            min: PointMm::origin(),
            max: PointMm {
                x: width_mm,
                y: height_mm,
            },
        },
        objects,
        import_warnings: warnings.clone(),
        source_to_canonical,
        source_format: SourceFormat::Svg,
        source_file_name: options.file_name.clone(),
    };
    document.recompute_bounds().ok();
    if document.objects.is_empty() {
        document.bounds_mm = Bounds2D {
            min: PointMm::origin(),
            max: PointMm {
                x: width_mm.max(0.0),
                y: height_mm.max(0.0),
            },
        };
    }

    let report = ImportReport {
        imported,
        approximated: 0,
        skipped,
        unit_certainty: document.units.certainty,
        bounds_mm: document.bounds_mm,
        warnings,
        entity_counts: entity_counts.into_vec(),
    };
    Ok(ImportResult { document, report })
}

fn walk(
    group: &usvg::Group,
    xf: &Affine,
    options: &ImportOptions,
    objects: &mut Vec<PathObject>,
    warnings: &mut Vec<ImportWarning>,
    imported: &mut u32,
    skipped: &mut u32,
    counts: &mut EntityCountMap,
) {
    for child in group.children() {
        match child {
            usvg::Node::Group(inner) => {
                walk(inner, xf, options, objects, warnings, imported, skipped, counts);
            }
            usvg::Node::Path(path) => {
                counts.bump("path");
                match convert_path(path, xf, options) {
                    Some(obj) => {
                        *imported += 1;
                        objects.push(obj);
                    }
                    None => {
                        *skipped += 1;
                        warnings.push(ImportWarning::new(
                            "empty_path",
                            "Skipped a path with no drawable segments",
                        ));
                    }
                }
            }
            usvg::Node::Image(_) => {
                counts.bump("image");
                *skipped += 1;
                warnings.push(ImportWarning::new(
                    "unsupported_image",
                    "Raster images are not imported in the MVP",
                ));
            }
            usvg::Node::Text(_) => {
                counts.bump("text");
                *skipped += 1;
                warnings.push(ImportWarning::new(
                    "unsupported_text",
                    "Text is not converted to paths. Convert text to outlines in the source application.",
                ));
            }
        }
    }
}

fn convert_path(path: &usvg::Path, xf: &Affine, options: &ImportOptions) -> Option<PathObject> {
    let mut subpaths = Vec::new();
    let mut current: Option<PointMm> = None;
    let mut start: Option<PointMm> = None;
    let mut segs: Vec<PathSegment> = Vec::new();

    // Compose the path's absolute transform (viewBox + parent group transforms)
    // with the source-to-canonical transform so path data is mapped correctly.
    let abs = path.abs_transform();
    let abs_affine = Affine {
        a: abs.sx as f64,
        b: abs.ky as f64,
        c: abs.kx as f64,
        d: abs.sy as f64,
        e: abs.tx as f64,
        f: abs.ty as f64,
    };
    let full_xf = abs_affine.then(*xf);

    let map = |x: f32, y: f32| {
        let p = PointMm {
            x: x as f64,
            y: y as f64,
        };
        full_xf.apply(p)
    };

    for seg in path.data().segments() {
        match seg {
            UsvgSeg::MoveTo(p) => {
                if let (Some(st), Some(cur)) = (start, current) {
                    if !segs.is_empty() {
                        subpaths.push(Subpath {
                            start: st,
                            segments: std::mem::take(&mut segs),
                            closed: false,
                        });
                    }
                    let _ = cur;
                }
                let pt = map(p.x, p.y);
                current = Some(pt);
                start = Some(pt);
            }
            UsvgSeg::LineTo(p) => {
                let pt = map(p.x, p.y);
                segs.push(PathSegment::LineTo { to: pt });
                current = Some(pt);
            }
            UsvgSeg::QuadTo(c, p) => {
                let pt = map(p.x, p.y);
                segs.push(PathSegment::QuadTo {
                    control: map(c.x, c.y),
                    to: pt,
                });
                current = Some(pt);
            }
            UsvgSeg::CubicTo(c1, c2, p) => {
                let pt = map(p.x, p.y);
                segs.push(PathSegment::CubicTo {
                    control1: map(c1.x, c1.y),
                    control2: map(c2.x, c2.y),
                    to: pt,
                });
                current = Some(pt);
            }
            UsvgSeg::Close => {
                if let Some(st) = start {
                    subpaths.push(Subpath {
                        start: st,
                        segments: std::mem::take(&mut segs),
                        closed: true,
                    });
                    current = Some(st);
                    start = Some(st);
                }
            }
        }
    }
    if let Some(st) = start {
        if !segs.is_empty() {
            subpaths.push(Subpath {
                start: st,
                segments: segs,
                closed: false,
            });
        }
    }
    if subpaths.is_empty() {
        return None;
    }

    let stroke = path.stroke().and_then(|s| paint_color(&s.paint(), s.opacity().get()));
    let fill = path.fill().and_then(|f| paint_color(&f.paint(), f.opacity().get()));
    let style = SourceStyle {
        stroke,
        fill,
        stroke_width_mm: path.stroke().map(|s| s.width().get() as f64 * (25.4 / 96.0)),
    };

    Some(PathObject {
        id: ObjectId::new(),
        geometry: PathGeometry { subpaths },
        source: SourceMetadata {
            format: SourceFormat::Svg,
            file_name: options.file_name.clone(),
            source_id: path.id().is_empty().then_some(None).unwrap_or(Some(path.id().to_string())),
            source_layer: None,
            entity_type: "path".into(),
            original_color: style.assignment_color(),
        },
        style,
        layer_id: None,
        visible: true,
        export_enabled: true,
    })
}

fn paint_color(paint: &Paint, opacity: f32) -> Option<Rgba8> {
    match paint {
        Paint::Color(c) => Some(Rgba8 {
            r: c.red,
            g: c.green,
            b: c.blue,
            a: (opacity * 255.0) as u8,
        }),
        Paint::LinearGradient(_) | Paint::RadialGradient(_) | Paint::Pattern(_) => None,
    }
}

fn inspect_svg_units(input: &[u8]) -> UnitMetadata {
    let text = String::from_utf8_lossy(input);
    let lower = text.to_ascii_lowercase();
    let svg_tag = match lower.find("<svg") {
        Some(i) => {
            let rest = &lower[i..];
            let end = rest.find('>').unwrap_or(rest.len().min(2000));
            rest[..end].to_string()
        }
        None => return UnitMetadata::ambiguous("No <svg> root element found"),
    };
    let width = attr_unit(&svg_tag, "width");
    let height = attr_unit(&svg_tag, "height");
    match (width, height) {
        (Some(u), _) | (_, Some(u)) if u != LengthUnit::Unknown => UnitMetadata::known(u),
        _ => {
            if svg_tag.contains("viewbox") {
                UnitMetadata::ambiguous(
                    "SVG provides a viewBox but no physical width/height units",
                )
            } else {
                UnitMetadata::ambiguous("SVG has no physical size or viewBox")
            }
        }
    }
}

fn attr_unit(tag: &str, name: &str) -> Option<LengthUnit> {
    let key = format!("{name}=");
    let idx = tag.find(&key)?;
    let after = &tag[idx + key.len()..];
    let quote = after.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = after[1..].split(quote).next()?;
    if body.ends_with("mm") {
        Some(LengthUnit::Millimeter)
    } else if body.ends_with("cm") {
        Some(LengthUnit::Centimeter)
    } else if body.ends_with("in") {
        Some(LengthUnit::Inch)
    } else if body.ends_with("pt") {
        Some(LengthUnit::Point)
    } else if body.ends_with("pc") {
        Some(LengthUnit::Pica)
    } else if body.ends_with("px") {
        Some(LengthUnit::CssPx)
    } else if body.ends_with('%') {
        Some(LengthUnit::Unknown)
    } else {
        Some(LengthUnit::Unknown)
    }
}

#[derive(Default)]
struct EntityCountMap {
    items: Vec<(String, u32)>,
}

impl EntityCountMap {
    fn bump(&mut self, ty: &str) {
        if let Some((_, n)) = self.items.iter_mut().find(|(k, _)| k == ty) {
            *n += 1;
        } else {
            self.items.push((ty.into(), 1));
        }
    }

    fn into_vec(self) -> Vec<EntityCount> {
        self.items
            .into_iter()
            .map(|(entity_type, count)| EntityCount { entity_type, count })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laser_import::VectorImporter;

    #[test]
    fn imports_physical_rectangle() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="50mm" viewBox="0 0 100 50">
            <rect x="10" y="10" width="80" height="30" fill="none" stroke="#FF0000" stroke-width="0.5"/>
        </svg>"##;
        let result = SvgImporter
            .import(svg, &ImportOptions::new("rect.svg"))
            .unwrap();
        assert_eq!(result.document.units.certainty, UnitCertainty::Known);
        assert!((result.document.bounds_mm.width() - 80.0).abs() < 0.2);
        assert!((result.document.bounds_mm.height() - 30.0).abs() < 0.2);
        assert_eq!(result.report.imported, 1);
        let color = result.document.objects[0].style.assignment_color().unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
    }

    #[test]
    fn viewbox_only_is_ambiguous() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="none" stroke="#00FF00"/>
        </svg>"##;
        let result = SvgImporter
            .import(svg, &ImportOptions::new("vb.svg"))
            .unwrap();
        assert_eq!(result.document.units.certainty, UnitCertainty::Ambiguous);
    }
}

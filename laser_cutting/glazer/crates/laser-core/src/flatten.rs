//! # Curve flattening
//!
//! Lasers move in straight lines, so quadratic/cubic Béziers and arcs are
//! subdivided into polylines here — once, in the core — rather than in every
//! consumer. The chord-error tolerance (job setting) is the maximum distance
//! between the true curve and the emitted segments; smaller values give
//! smoother output at the cost of more segments.

use crate::geom::{PathGeometry, PathSegment, PointMm, Subpath};

/// Flatten curves to polylines using a chord-error tolerance in millimetres.
pub fn flatten_geometry(geometry: &PathGeometry, chord_error_mm: f64) -> Vec<Vec<PointMm>> {
    let tol = chord_error_mm.max(1e-6);
    geometry
        .subpaths
        .iter()
        .map(|sub| flatten_subpath(sub, tol))
        .filter(|line| line.len() >= 2)
        .collect()
}

pub fn flatten_subpath(subpath: &Subpath, chord_error_mm: f64) -> Vec<PointMm> {
    let mut pts = vec![subpath.start];
    let mut current = subpath.start;
    for seg in &subpath.segments {
        match *seg {
            PathSegment::LineTo { to } => {
                pts.push(to);
                current = to;
            }
            PathSegment::QuadTo { control, to } => {
                flatten_quad(current, control, to, chord_error_mm, &mut pts);
                current = to;
            }
            PathSegment::CubicTo {
                control1,
                control2,
                to,
            } => {
                flatten_cubic(current, control1, control2, to, chord_error_mm, &mut pts);
                current = to;
            }
            PathSegment::Arc {
                center,
                radius,
                start_angle_rad,
                sweep_rad,
            } => {
                flatten_arc(center, radius, start_angle_rad, sweep_rad, chord_error_mm, &mut pts);
                current = pts.last().copied().unwrap_or(current);
            }
        }
    }
    if subpath.closed {
        if let Some(first) = pts.first().copied() {
            if first.distance(*pts.last().unwrap()) > 1e-9 {
                pts.push(first);
            }
        }
    }
    pts
}

fn flatten_quad(p0: PointMm, p1: PointMm, p2: PointMm, tol: f64, out: &mut Vec<PointMm>) {
    flatten_quad_rec(p0, p1, p2, tol, 0, out);
}

fn flatten_quad_rec(p0: PointMm, p1: PointMm, p2: PointMm, tol: f64, depth: u32, out: &mut Vec<PointMm>) {
    if depth > 12 || chord_ok_quad(p0, p1, p2, tol) {
        out.push(p2);
        return;
    }
    let p01 = p0.lerp(p1, 0.5);
    let p12 = p1.lerp(p2, 0.5);
    let mid = p01.lerp(p12, 0.5);
    flatten_quad_rec(p0, p01, mid, tol, depth + 1, out);
    flatten_quad_rec(mid, p12, p2, tol, depth + 1, out);
}

fn chord_ok_quad(p0: PointMm, p1: PointMm, p2: PointMm, tol: f64) -> bool {
    dist_to_segment(p1, p0, p2) <= tol
}

fn flatten_cubic(
    p0: PointMm,
    p1: PointMm,
    p2: PointMm,
    p3: PointMm,
    tol: f64,
    out: &mut Vec<PointMm>,
) {
    flatten_cubic_rec(p0, p1, p2, p3, tol, 0, out);
}

fn flatten_cubic_rec(
    p0: PointMm,
    p1: PointMm,
    p2: PointMm,
    p3: PointMm,
    tol: f64,
    depth: u32,
    out: &mut Vec<PointMm>,
) {
    if depth > 12 || chord_ok_cubic(p0, p1, p2, p3, tol) {
        out.push(p3);
        return;
    }
    let p01 = p0.lerp(p1, 0.5);
    let p12 = p1.lerp(p2, 0.5);
    let p23 = p2.lerp(p3, 0.5);
    let p012 = p01.lerp(p12, 0.5);
    let p123 = p12.lerp(p23, 0.5);
    let mid = p012.lerp(p123, 0.5);
    flatten_cubic_rec(p0, p01, p012, mid, tol, depth + 1, out);
    flatten_cubic_rec(mid, p123, p23, p3, tol, depth + 1, out);
}

fn chord_ok_cubic(p0: PointMm, p1: PointMm, p2: PointMm, p3: PointMm, tol: f64) -> bool {
    dist_to_segment(p1, p0, p3) <= tol && dist_to_segment(p2, p0, p3) <= tol
}

fn flatten_arc(
    center: PointMm,
    radius: f64,
    start_angle: f64,
    sweep: f64,
    tol: f64,
    out: &mut Vec<PointMm>,
) {
    let radius = radius.abs().max(1e-9);
    let max_sagitta = (1.0 - (tol / radius).clamp(0.0, 1.0)).acos().max(1e-3) * 2.0;
    // sagitta = r (1 - cos(dtheta/2)); invert for dtheta
    let ratio = (1.0 - (tol / radius)).clamp(-1.0, 1.0);
    let dtheta = if tol >= radius {
        std::f64::consts::PI
    } else {
        (2.0 * ratio.acos()).clamp(1e-3, std::f64::consts::PI)
    };
    let _ = max_sagitta;
    let steps = ((sweep.abs() / dtheta).ceil() as usize).clamp(1, 1024);
    let step = sweep / steps as f64;
    for i in 1..=steps {
        let a = start_angle + step * i as f64;
        out.push(PointMm {
            x: center.x + radius * a.cos(),
            y: center.y + radius * a.sin(),
        });
    }
}

fn dist_to_segment(p: PointMm, a: PointMm, b: PointMm) -> f64 {
    let vx = b.x - a.x;
    let vy = b.y - a.y;
    let len2 = vx * vx + vy * vy;
    if len2 < 1e-18 {
        return p.distance(a);
    }
    let t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2;
    let t = t.clamp(0.0, 1.0);
    let proj = PointMm {
        x: a.x + vx * t,
        y: a.y + vy * t,
    };
    p.distance(proj)
}

pub fn polyline_length(points: &[PointMm]) -> f64 {
    points.windows(2).map(|w| w[0].distance(w[1])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_length_is_exact() {
        let geo = PathGeometry {
            subpaths: vec![Subpath {
                start: PointMm { x: 0.0, y: 0.0 },
                segments: vec![PathSegment::LineTo {
                    to: PointMm { x: 10.0, y: 0.0 },
                }],
                closed: false,
            }],
        };
        let flat = flatten_geometry(&geo, 0.01);
        assert!((polyline_length(&flat[0]) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn circle_arc_length_close() {
        let r = 10.0;
        let geo = PathGeometry {
            subpaths: vec![Subpath {
                start: PointMm { x: r, y: 0.0 },
                segments: vec![PathSegment::Arc {
                    center: PointMm { x: 0.0, y: 0.0 },
                    radius: r,
                    start_angle_rad: 0.0,
                    sweep_rad: std::f64::consts::TAU,
                }],
                closed: true,
            }],
        };
        let flat = flatten_geometry(&geo, 0.01);
        let len = polyline_length(&flat[0]);
        let expected = std::f64::consts::TAU * r;
        assert!((len - expected).abs() / expected < 0.01);
    }
}

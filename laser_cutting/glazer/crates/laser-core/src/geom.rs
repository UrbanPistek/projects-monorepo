//! # Geometry primitives
//!
//! Points, 2D bounds, path geometry (subpaths of line/quad/cubic/arc
//! segments) and the [`Affine`] transform used for the job-level
//! scale/rotate/offset. All coordinates are millimetres, Y up.
//!
//! Why `Affine` is spelled out as six fields instead of using a matrix crate:
//! the project file is JSON, and plain named fields keep saved projects
//! readable and diff-able.

use serde::{Deserialize, Serialize};

use crate::error::{check_finite, CoreError, MAX_EXTENT_MM};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PointMm {
    pub x: f64,
    pub y: f64,
}

impl PointMm {
    pub fn new(x: f64, y: f64) -> Result<Self, CoreError> {
        let x = check_finite(x)?;
        let y = check_finite(y)?;
        if x.abs() > MAX_EXTENT_MM || y.abs() > MAX_EXTENT_MM {
            return Err(CoreError::ExtentTooLarge);
        }
        Ok(Self { x, y })
    }

    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    pub fn transformed(self, xf: Affine) -> Self {
        xf.apply(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds2D {
    pub min: PointMm,
    pub max: PointMm,
}

impl Bounds2D {
    pub fn from_points(points: impl IntoIterator<Item = PointMm>) -> Result<Self, CoreError> {
        let mut iter = points.into_iter();
        let first = iter.next().ok_or(CoreError::EmptyGeometry)?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
        }
        Ok(Self { min, max })
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: PointMm {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
            },
            max: PointMm {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
            },
        }
    }

    pub fn expand(self, margin_mm: f64) -> Self {
        Self {
            min: PointMm {
                x: self.min.x - margin_mm,
                y: self.min.y - margin_mm,
            },
            max: PointMm {
                x: self.max.x + margin_mm,
                y: self.max.y + margin_mm,
            },
        }
    }

    pub fn width(self) -> f64 {
        self.max.x - self.min.x
    }

    pub fn height(self) -> f64 {
        self.max.y - self.min.y
    }

    pub fn center(self) -> PointMm {
        PointMm {
            x: (self.min.x + self.max.x) * 0.5,
            y: (self.min.y + self.max.y) * 0.5,
        }
    }

    pub fn corners(self) -> [PointMm; 4] {
        [
            self.min,
            PointMm {
                x: self.max.x,
                y: self.min.y,
            },
            self.max,
            PointMm {
                x: self.min.x,
                y: self.max.y,
            },
        ]
    }

    pub fn transformed(self, xf: Affine) -> Result<Self, CoreError> {
        Self::from_points(self.corners().into_iter().map(|p| xf.apply(p)))
    }
}

/// Affine transform `x' = a x + c y + e`, `y' = b x + d y + f`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Affine {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    pub fn translate(x: f64, y: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: x,
            f: y,
        }
    }

    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    pub fn rotate_deg(deg: f64) -> Self {
        let r = deg.to_radians();
        let (sin, cos) = r.sin_cos();
        Self {
            a: cos,
            b: sin,
            c: -sin,
            d: cos,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Flip Y around the X axis then shift so the former top becomes Y=0 and Y increases upward.
    /// `height_mm` is the source document height in millimetres before the flip.
    pub fn svg_y_up(height_mm: f64) -> Self {
        Self::scale(1.0, -1.0).then(Self::translate(0.0, height_mm))
    }

    pub fn then(self, next: Self) -> Self {
        Self {
            a: next.a * self.a + next.c * self.b,
            b: next.b * self.a + next.d * self.b,
            c: next.a * self.c + next.c * self.d,
            d: next.b * self.c + next.d * self.d,
            e: next.a * self.e + next.c * self.f + next.e,
            f: next.b * self.e + next.d * self.f + next.f,
        }
    }

    pub fn apply(self, p: PointMm) -> PointMm {
        PointMm {
            x: self.a * p.x + self.c * p.y + self.e,
            y: self.b * p.x + self.d * p.y + self.f,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PathSegment {
    LineTo {
        to: PointMm,
    },
    QuadTo {
        control: PointMm,
        to: PointMm,
    },
    CubicTo {
        control1: PointMm,
        control2: PointMm,
        to: PointMm,
    },
    /// Circular arc. `sweep_rad` is signed: positive is counter-clockwise.
    Arc {
        center: PointMm,
        radius: f64,
        start_angle_rad: f64,
        sweep_rad: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subpath {
    pub start: PointMm,
    pub segments: Vec<PathSegment>,
    pub closed: bool,
}

impl Subpath {
    pub fn points_hint(&self) -> Vec<PointMm> {
        let mut pts = vec![self.start];
        for seg in &self.segments {
            match seg {
                PathSegment::LineTo { to } => pts.push(*to),
                PathSegment::QuadTo { control, to } => {
                    pts.push(*control);
                    pts.push(*to);
                }
                PathSegment::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    pts.push(*control1);
                    pts.push(*control2);
                    pts.push(*to);
                }
                PathSegment::Arc { center, radius, .. } => {
                    pts.push(PointMm {
                        x: center.x - radius,
                        y: center.y - radius,
                    });
                    pts.push(PointMm {
                        x: center.x + radius,
                        y: center.y + radius,
                    });
                }
            }
        }
        pts
    }

    pub fn transformed(&self, xf: Affine) -> Self {
        let map = |p: PointMm| xf.apply(p);
        Self {
            start: map(self.start),
            closed: self.closed,
            segments: self
                .segments
                .iter()
                .map(|seg| match *seg {
                    PathSegment::LineTo { to } => PathSegment::LineTo { to: map(to) },
                    PathSegment::QuadTo { control, to } => PathSegment::QuadTo {
                        control: map(control),
                        to: map(to),
                    },
                    PathSegment::CubicTo {
                        control1,
                        control2,
                        to,
                    } => PathSegment::CubicTo {
                        control1: map(control1),
                        control2: map(control2),
                        to: map(to),
                    },
                    PathSegment::Arc {
                        center,
                        radius,
                        start_angle_rad,
                        sweep_rad,
                    } => {
                        let sx = (xf.a * xf.a + xf.b * xf.b).sqrt();
                        let sy = (xf.c * xf.c + xf.d * xf.d).sqrt();
                        let scale = (sx + sy) * 0.5;
                        let det = xf.a * xf.d - xf.b * xf.c;
                        let sweep = if det < 0.0 { -sweep_rad } else { sweep_rad };
                        PathSegment::Arc {
                            center: map(center),
                            radius: radius * scale,
                            start_angle_rad,
                            sweep_rad: sweep,
                        }
                    }
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathGeometry {
    pub subpaths: Vec<Subpath>,
}

impl PathGeometry {
    pub fn bounds(&self) -> Result<Bounds2D, CoreError> {
        let pts: Vec<PointMm> = self.subpaths.iter().flat_map(|s| s.points_hint()).collect();
        Bounds2D::from_points(pts)
    }

    pub fn transformed(&self, xf: Affine) -> Self {
        Self {
            subpaths: self.subpaths.iter().map(|s| s.transformed(xf)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_up_flip_moves_svg_origin() {
        let xf = Affine::svg_y_up(50.0);
        let top_left = xf.apply(PointMm { x: 0.0, y: 0.0 });
        let bottom_left = xf.apply(PointMm { x: 0.0, y: 50.0 });
        assert!((top_left.y - 50.0).abs() < 1e-9);
        assert!(bottom_left.y.abs() < 1e-9);
    }
}

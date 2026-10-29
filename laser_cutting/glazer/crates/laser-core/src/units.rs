use serde::{Deserialize, Serialize};

use crate::error::{check_finite, CoreError};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LengthUnit {
    Millimeter,
    Centimeter,
    Meter,
    Inch,
    Foot,
    Point,
    Pica,
    CssPx,
    Unknown,
}

impl LengthUnit {
    pub fn to_mm_factor(self) -> Option<f64> {
        match self {
            Self::Millimeter => Some(1.0),
            Self::Centimeter => Some(10.0),
            Self::Meter => Some(1000.0),
            Self::Inch => Some(25.4),
            Self::Foot => Some(304.8),
            Self::Point => Some(25.4 / 72.0),
            Self::Pica => Some(25.4 / 6.0),
            Self::CssPx => Some(25.4 / 96.0),
            Self::Unknown => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Millimeter => "mm",
            Self::Centimeter => "cm",
            Self::Meter => "m",
            Self::Inch => "in",
            Self::Foot => "ft",
            Self::Point => "pt",
            Self::Pica => "pc",
            Self::CssPx => "px",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UnitCertainty {
    Known,
    Assumed,
    Ambiguous,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnitMetadata {
    pub source_unit: LengthUnit,
    pub certainty: UnitCertainty,
    pub assumed_unit: Option<LengthUnit>,
    pub notes: Vec<String>,
}

impl UnitMetadata {
    pub fn known(unit: LengthUnit) -> Self {
        Self {
            source_unit: unit,
            certainty: UnitCertainty::Known,
            assumed_unit: None,
            notes: Vec::new(),
        }
    }

    pub fn assumed(detected: LengthUnit, assumed: LengthUnit, note: impl Into<String>) -> Self {
        Self {
            source_unit: detected,
            certainty: UnitCertainty::Assumed,
            assumed_unit: Some(assumed),
            notes: vec![note.into()],
        }
    }

    pub fn ambiguous(note: impl Into<String>) -> Self {
        Self {
            source_unit: LengthUnit::Unknown,
            certainty: UnitCertainty::Ambiguous,
            assumed_unit: None,
            notes: vec![note.into()],
        }
    }

    pub fn export_blocked(&self) -> bool {
        self.certainty == UnitCertainty::Ambiguous
    }

    pub fn conversion_unit(&self) -> LengthUnit {
        self.assumed_unit.unwrap_or(self.source_unit)
    }
}

pub fn to_mm(value: f64, unit: LengthUnit) -> Result<f64, CoreError> {
    let value = check_finite(value)?;
    let factor = unit
        .to_mm_factor()
        .ok_or_else(|| CoreError::Invalid("cannot convert unknown unit to millimetres".into()))?;
    check_finite(value * factor)
}

/// DXF `$INSUNITS` mapping.
pub fn dxf_insunits(code: i16) -> (LengthUnit, UnitCertainty, &'static str) {
    match code {
        0 => (
            LengthUnit::Unknown,
            UnitCertainty::Ambiguous,
            "DXF $INSUNITS is unitless",
        ),
        1 => (LengthUnit::Inch, UnitCertainty::Known, "DXF inches"),
        2 => (LengthUnit::Foot, UnitCertainty::Known, "DXF feet"),
        4 => (LengthUnit::Millimeter, UnitCertainty::Known, "DXF millimetres"),
        5 => (LengthUnit::Centimeter, UnitCertainty::Known, "DXF centimetres"),
        6 => (LengthUnit::Meter, UnitCertainty::Known, "DXF metres"),
        _ => (
            LengthUnit::Unknown,
            UnitCertainty::Ambiguous,
            "DXF $INSUNITS is unsupported or missing",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_common_units() {
        assert!((to_mm(1.0, LengthUnit::Inch).unwrap() - 25.4).abs() < 1e-9);
        assert!((to_mm(1.0, LengthUnit::Centimeter).unwrap() - 10.0).abs() < 1e-9);
        assert!((to_mm(96.0, LengthUnit::CssPx).unwrap() - 25.4).abs() < 1e-9);
        assert!((to_mm(72.0, LengthUnit::Point).unwrap() - 25.4).abs() < 1e-9);
    }

    #[test]
    fn dxf_unitless_is_ambiguous() {
        let (unit, certainty, _) = dxf_insunits(0);
        assert_eq!(unit, LengthUnit::Unknown);
        assert_eq!(certainty, UnitCertainty::Ambiguous);
    }
}

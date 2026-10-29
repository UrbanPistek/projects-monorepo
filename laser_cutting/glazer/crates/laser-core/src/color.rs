use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim().trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }
        let n = u32::from_str_radix(hex, 16).ok()?;
        Some(Self::rgb(
            ((n >> 16) & 0xff) as u8,
            ((n >> 8) & 0xff) as u8,
            (n & 0xff) as u8,
        ))
    }

    pub fn to_hex(self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Euclidean RGB distance, 0..=441.67
    pub fn rgb_distance(self, other: Self) -> f64 {
        let dr = self.r as f64 - other.r as f64;
        let dg = self.g as f64 - other.g as f64;
        let db = self.b as f64 - other.b as f64;
        (dr * dr + dg * dg + db * db).sqrt()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorPreset {
    pub name: String,
    pub color: Rgba8,
    pub suggested_use: String,
}

pub fn standard_palette() -> Vec<ColorPreset> {
    vec![
        ColorPreset {
            name: "Cut".into(),
            color: Rgba8::from_hex("#FF0000").unwrap(),
            suggested_use: "Cut".into(),
        },
        ColorPreset {
            name: "Engrave".into(),
            color: Rgba8::from_hex("#00FF00").unwrap(),
            suggested_use: "Engrave".into(),
        },
        ColorPreset {
            name: "Mark".into(),
            color: Rgba8::from_hex("#0000FF").unwrap(),
            suggested_use: "Mark/engrave".into(),
        },
        ColorPreset {
            name: "Secondary".into(),
            color: Rgba8::from_hex("#FFFF00").unwrap(),
            suggested_use: "Secondary operation".into(),
        },
        ColorPreset {
            name: "Special Magenta".into(),
            color: Rgba8::from_hex("#FF00FF").unwrap(),
            suggested_use: "Special operation".into(),
        },
        ColorPreset {
            name: "Special Cyan".into(),
            color: Rgba8::from_hex("#00FFFF").unwrap(),
            suggested_use: "Special operation".into(),
        },
        ColorPreset {
            name: "Black".into(),
            color: Rgba8::from_hex("#000000").unwrap(),
            suggested_use: "Unassigned/engrave candidate".into(),
        },
        ColorPreset {
            name: "White".into(),
            color: Rgba8::from_hex("#FFFFFF").unwrap(),
            suggested_use: "Unassigned/disabled candidate".into(),
        },
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorMatch {
    pub preset_name: String,
    pub preset_color: Rgba8,
    pub distance: f64,
}

pub fn match_color(color: Rgba8, presets: &[ColorPreset], tolerance: f64) -> Option<ColorMatch> {
    presets
        .iter()
        .map(|p| ColorMatch {
            preset_name: p.name.clone(),
            preset_color: p.color,
            distance: color.rgb_distance(p.color),
        })
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
        .filter(|m| m.distance <= tolerance)
}

/// Default Euclidean RGB tolerance. 40 covers mild anti-aliasing / rounding.
pub const DEFAULT_COLOR_TOLERANCE: f64 = 40.0;

/// AutoCAD Color Index (ACI) 1–255 to RGB. Index 0 is ByBlock, 256 is ByLayer.
pub fn aci_to_rgb(index: i16) -> Option<Rgba8> {
    match index {
        1 => Some(Rgba8::rgb(255, 0, 0)),
        2 => Some(Rgba8::rgb(255, 255, 0)),
        3 => Some(Rgba8::rgb(0, 255, 0)),
        4 => Some(Rgba8::rgb(0, 255, 255)),
        5 => Some(Rgba8::rgb(0, 0, 255)),
        6 => Some(Rgba8::rgb(255, 0, 255)),
        7 => Some(Rgba8::rgb(255, 255, 255)),
        8 => Some(Rgba8::rgb(128, 128, 128)),
        9 => Some(Rgba8::rgb(192, 192, 192)),
        250..=255 => Some(Rgba8::rgb(0, 0, 0)),
        i if (10..=249).contains(&i) => {
            // Approximate remaining ACI by a simple hue cycle; exact ACI table is large.
            let t = (i - 10) as f64 / 239.0;
            let r = (t * 255.0) as u8;
            Some(Rgba8::rgb(r, 128, 255u8.saturating_sub(r)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_red_cut_preset() {
        let pal = standard_palette();
        let m = match_color(Rgba8::rgb(254, 1, 0), &pal, DEFAULT_COLOR_TOLERANCE).unwrap();
        assert_eq!(m.preset_name, "Cut");
    }

    #[test]
    fn unmatched_stays_none() {
        let pal = standard_palette();
        assert!(match_color(Rgba8::rgb(120, 80, 40), &pal, 20.0).is_none());
    }
}

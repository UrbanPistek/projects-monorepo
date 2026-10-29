use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("non-finite coordinate or numeric value")]
    NonFinite,
    #[error("geometry extent is unreasonably large")]
    ExtentTooLarge,
    #[error("invalid value: {0}")]
    Invalid(String),
    #[error("empty geometry")]
    EmptyGeometry,
}

impl CoreError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NonFinite => "non_finite",
            Self::ExtentTooLarge => "extent_too_large",
            Self::Invalid(_) => "invalid",
            Self::EmptyGeometry => "empty_geometry",
        }
    }
}

impl Serialize for CoreError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("CoreError", 2)?;
        s.serialize_field("code", self.code())?;
        s.serialize_field("message", &self.to_string())?;
        s.end()
    }
}

pub fn check_finite(value: f64) -> Result<f64, CoreError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CoreError::NonFinite)
    }
}

pub const MAX_EXTENT_MM: f64 = 1_000_000.0;

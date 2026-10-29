pub mod color;
pub mod document;
pub mod error;
pub mod flatten;
pub mod geom;
pub mod ids;
pub mod layers;
pub mod project;
pub mod style;
pub mod units;

pub use color::{
    aci_to_rgb, match_color, standard_palette, ColorMatch, ColorPreset, Rgba8, DEFAULT_COLOR_TOLERANCE,
};
pub use document::{EntityCount, ImportReport, ImportResult, PathObject, VectorDocument};
pub use error::CoreError;
pub use flatten::{flatten_geometry, polyline_length};
pub use geom::{Affine, Bounds2D, PathGeometry, PathSegment, PointMm, Subpath};
pub use ids::{DocumentId, LayerId, ObjectId};
pub use layers::{AssignmentRule, LaserMode, OperationKind, ProcessLayer};
pub use project::{
    DisplaySettings, ExportSettings, JobSettings, MachineSettings, OriginMode, Project,
    APP_VERSION, PROJECT_SCHEMA_VERSION,
};
pub use style::{ImportWarning, SourceFormat, SourceMetadata, SourceStyle};
pub use units::{LengthUnit, UnitCertainty, UnitMetadata};

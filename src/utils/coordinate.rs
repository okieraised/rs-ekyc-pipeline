use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarkMetadata {
    pub near: FaceLandmark,
    pub mid: FaceLandmark,
    pub far: FaceLandmark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmark {
    pub left_eye: Coordinate2D,
    pub right_eye: Coordinate2D,
    pub nose: Coordinate2D,
    pub left_mouth: Coordinate2D,
    pub right_mouth: Coordinate2D,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinate2D {
    pub x: f32,
    pub y: f32,
}
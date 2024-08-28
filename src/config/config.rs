use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceDetectionConfig {
    pub model_name: String,
    pub timeout: i32,
    pub mean: f32,
    pub scale: f32,
}

impl FaceDetectionConfig {
    pub(crate) fn new() -> Self {
        FaceDetectionConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: 127.5,
            scale: 0.00784313725490196,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceIDConfig {
    pub model_name: String,
    pub timeout: i32,
    pub mean: f32,
    pub scale: f32,
    pub threshold_same_ekyc: f32,
    pub threshold_same_person: f32,
    pub imsize: (i32, i32),
}

impl FaceIDConfig {
    pub(crate) fn new() -> Self {
        FaceIDConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: 127.5,
            scale: 0.00784313725490196,
            threshold_same_ekyc: 0.3,
            threshold_same_person: 0.4,
            imsize: (112, 112),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceAttributeConfig {
    pub model_name: String,
    pub timeout: i32,
    pub mean: f32,
    pub scale: f32,
    pub threshold_face_mask: f32,
    pub imsize: i32,
}

impl FaceAttributeConfig {
    pub(crate) fn new() -> Self {
        FaceAttributeConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: 127.5,
            scale: 1.0 / 127.5,
            threshold_face_mask: 0.5,
            imsize: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceQualityConfig {
    pub model_name: String,
    pub timeout: i32,
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub threshold_cover: f32,
    pub threshold_all: f32,
    pub imsize: i32,
}

impl FaceQualityConfig {
    pub(crate) fn new() -> Self {
        FaceQualityConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: vec![123.675, 116.28, 103.53],
            scale: vec![1.0/(0.229*255.0), 1.0/(0.224*255.0), 1.0/(0.225*255.0)],
            threshold_cover: 0.5,
            threshold_all: 0.5,
            imsize: 112,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FASConfig {
    pub model_name: String,
    pub timeout: i32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub threshold: f32,
    pub imsize: (i32, i32),
}

impl FASConfig {
    pub(crate) fn new_crop() -> Self {
        FASConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            threshold: 0.58,
            imsize: (224, 224),
        }
    }

    pub(crate) fn new_full() -> Self {
        FASConfig {
            model_name: "abc".to_string(),
            timeout: 20,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            threshold: 0.48,
            imsize: (224, 224),
        }
    }
}
use std::collections::HashMap;
use anyhow::Error;
use ndarray::{array, Array2, ArrayBase, Ix2, OwnedRepr, ShapeError, stack};
use opencv::core::{Mat, MatTraitConst, Range, Vector};
use opencv::imgcodecs::{imdecode, IMREAD_COLOR, imwrite};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color};
use crate::utils::coordinate::{Coordinate2D, FaceLandmark};


pub fn convert_image_to_ndarray(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = match Mat::from_slice(im_bytes) {
        Ok(img_as_mat) => img_as_mat,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Decode the image
    let img_as_arr_bgr = match imdecode(&img_as_mat, IMREAD_COLOR) {
        Ok(img_as_arr_bgr) => img_as_arr_bgr,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    let mut img_as_arr_rgb = Mat::default();
    let _ = match cvt_color(&img_as_arr_bgr, &mut img_as_arr_rgb, COLOR_BGR2RGB, 0) {
        Ok(_) => {}
        Err(e) => return Err(Error::from(e))
    };

    Ok(img_as_arr_rgb)
}

pub fn convert_json_metadata_to_ndarray(metadata: FaceLandmark) -> Result<Array2<f32>, Error> {
    let mut result:Vec<f32> = Vec::new();
    let mut nrows = 5;
    let ncols = 2;

    result.extend_from_slice(&*vec![metadata.left_eye.x, metadata.left_eye.y]);
    result.extend_from_slice(&*vec![metadata.right_eye.x, metadata.right_eye.y]);
    result.extend_from_slice(&*vec![metadata.nose.x, metadata.nose.y]);
    result.extend_from_slice(&*vec![metadata.left_mouth.x, metadata.left_mouth.y]);
    result.extend_from_slice(&*vec![metadata.right_mouth.x, metadata.right_mouth.y]);

    let arr = match Array2::from_shape_vec((nrows, ncols), result) {
        Ok(arr) => {arr}
        Err(e) => return Err(Error::from(e))
    };

    Ok(arr)
}

pub fn convert_hashmap_metadata_to_ndarray(metadata: HashMap<&str, Coordinate2D>) -> Result<Option<Array2<f32>>, Error> {

    if metadata.is_empty() {
        return Ok(None)
    }

    let mut result = Vec::new();
    let mut nrows = 0;
    let ncols = 2;

    let ordered_meta_key = vec!["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"];

    for key in ordered_meta_key {
        let val = metadata.get(key);
        if let Some(_val) = val {
            result.extend_from_slice(&*vec![_val.x, _val.y]);
            nrows += 1;
        }
    }

    let arr = match Array2::from_shape_vec((nrows, ncols), result) {
        Ok(arr) => {arr}
        Err(e) => return Err(Error::from(e))
    };

    Ok(Some(arr))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::utils::coordinate::{Coordinate2D, FaceLandmarkMetadata};
    use crate::utils::image::{convert_image_to_ndarray, convert_hashmap_metadata_to_ndarray, convert_json_metadata_to_ndarray};

    #[test]
    fn test_convert_image_to_ndarray() {
    }

    #[test]
    fn test_convert_metadata_to_ndarray() {

        let mut metadata: HashMap<&str, Coordinate2D> = HashMap::from(
            [
                ("left_eye", Coordinate2D{x: 169.7128, y: 213.38426 }),
                ("right_eye", Coordinate2D{x: 455.29285, y: 223.66956 }),
                ("nose", Coordinate2D{x: 310.71146, y: 320.74503 }),
                ("left_mouth", Coordinate2D{x: 195.21452, y: 379.8982 }),
                ("right_mouth", Coordinate2D{x: 408.377, y: 384.25134}),
            ]
        );

        let result = match convert_hashmap_metadata_to_ndarray(metadata) {
            Ok(result) => {result}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };
        if result.is_some() {
            println!("{:?}", result);
        }
    }

    #[test]
    fn test_convert_json_metadata_json_to_ndarray() {
        let metadata = r#"{"near":{"left_eye":{"x":169.7128,"y":213.38426},"right_eye":{"x":455.29285,"y":223.66956},"nose":{"x":310.71146,"y":320.74503},"left_mouth":{"x":195.21452,"y":379.8982},"right_mouth":{"x":408.377,"y":384.25134}},"mid":{"left_eye":{"x":276.0993,"y":226.09839},"right_eye":{"x":450.5989,"y":228.72801},"nose":{"x":365.71985,"y":283.22446},"left_mouth":{"x":300.92358,"y":324.42694},"right_mouth":{"x":427.99792,"y":326.67972}},"far":{"left_eye":{"x":266.14566,"y":220.05692},"right_eye":{"x":397.149,"y":221.45383},"nose":{"x":332.13202,"y":258.6127},"left_mouth":{"x":284.05356,"y":294.186},"right_mouth":{"x":380.22375,"y":294.31165}}}"#;
        let face_metadata: FaceLandmarkMetadata = serde_json::from_str(metadata).unwrap();
        let result = match convert_json_metadata_to_ndarray(face_metadata.near) {
            Ok(result) => {result}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };
        println!("{:?}", result);

    }
}
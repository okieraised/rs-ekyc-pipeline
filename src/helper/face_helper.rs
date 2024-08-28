use anyhow::Error;
use nalgebra::Vector2;
use ndarray::{Array, Array2, ArrayBase, Axis, Dim, Ix, Ix1, IxDyn};
use opencv::calib3d::{estimate_affine_partial_2d, LMEDS};
use opencv::core::{BORDER_CONSTANT, copy_make_border, Mat, MatTraitConst, MatTraitConstManual, Point2f, Scalar, Size, Vector};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color, INTER_LINEAR, warp_affine};
use crate::modules::face_detection_client::FaceDetectionClient;

#[derive(Debug, Clone)]
pub struct FaceHelper {
    face_size: (i32, i32),
    face_template: Array2<f32>,
    fas_size: (i32, i32),
    fas_template: Array2<f32>,
    fa_size: (i32, i32),
    fa_template: Array2<f32>,
    face_det: FaceDetectionClient,
}

impl FaceHelper {

    /// new initializes new instance of face helper module.
    pub fn new(
        in_face_size: Option<(i32, i32)>, in_face_template: Option<Array2<f32>>,
        in_fas_size: Option<(i32, i32)>, in_fas_template: Option<Array2<f32>>,
        in_fa_size: Option<(i32, i32)>, in_fa_template: Option<Array2<f32>>,
        face_det: FaceDetectionClient,
    ) -> Self {

        let mut face_size: (i32, i32) = (112, 112);
        if let Some(_in_face_size) = in_face_size {
            face_size = _in_face_size;
        }

        let mut fas_size: (i32, i32) = (224, 224);
        if let Some(_in_fas_size) = in_fas_size {
            fas_size = _in_fas_size;
        }

        let mut fa_size: (i32, i32) = (128/4*3, 128);
        if let Some(_in_fa_size) = in_fa_size {
            fa_size = (_in_fa_size.0/4*3, _in_fa_size.1);
        }

        let mut face_template: Array2<f32> = Array2::<f32>::from(
            vec![
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ]
        );
        if let Some(_in_face_template) = in_face_template {
            face_template = _in_face_template;
        };

        let face_template = face_template.mapv(|x| x * (face_size.0 / 112) as f32);

        let mut fas_template: Array2<f32> = Array2::<f32>::from(
            vec![

                [74.01555, 90.46853],
                [135.68065, 90.12745],
                [105.0441, 125.539055],
                [79.71127, 161.63963],
                [130.77733, 161.35718],

            ]
        );
        if let Some(_in_fas_tempalte) = in_fas_template {
            fas_template = _in_fas_tempalte;
        };

        let mut fa_template: Array2<f32> = Array2::<f32>::from(
            vec![

                [0.34549999237060547, 0.38670000433921814],
                [0.6545000076293945, 0.38670000433921814],
                [0.5, 0.5386250019073486],
                [0.3970000147819519, 0.6596499681472778],
                [0.6030000448226929, 0.6596499681472778]

            ]
        );

        if let Some(_in_fa_template) = in_fa_template{
            fa_template = _in_fa_template;
        };

        fa_template = fa_template.mapv(|x| x * fa_size.0 as f32);

        FaceHelper {
            face_size,
            face_template,
            fas_size,
            fas_template,
            fa_size,
            fa_template,
            face_det,
        }
    }

    pub fn swap_rgb(&self, batch_imgs: &[Mat]) -> Result<Vec<Mat>, Error>{

        let mut result: Vec<Mat> = Vec::with_capacity(batch_imgs.len());
        for img in batch_imgs {
            let mut converted_img = Mat::default();
            cvt_color(img, &mut converted_img, COLOR_BGR2RGB, 0);
            result.push(converted_img);
        }
        Ok(result)

    }

    pub async fn get_face_landmarks_5(
        &self,
        batch_imgs: Vec<Mat>,
        keep_largest: Option<bool>,
        keep_center: Option<bool>,
        score_threshold: Option<f32>,
        eye_distance_threshold: Option<f32>,
        try_padding: Option<bool>,
    ) -> Result<(Vec<Vec<Array<f32, IxDyn>>>, Vec<Vec<Array<f32, Dim<[Ix; 2]>>>>), Error>{
        let padding = try_padding.unwrap_or(false);

        let batch_results = self.face_det.infer_batch(&vec![batch_imgs.clone()]).await?;

        let mut batch_bboxes: Vec<Vec<Array<f32, IxDyn>>> = Vec::with_capacity(1);
        let mut batch_landmarks: Vec<Vec<Array<f32, Dim<[Ix; 2]>>>> = Vec::with_capacity(1);

        for idx in 0..batch_results.len() {
            let results = &batch_results[idx];
            let (off_x, off_y): (f32, f32) = (0.0, 0.0);

            if padding && results.len() == 0 {
                return Ok((vec![], vec![]))
            }

            let mut filter_bboxes: Vec<Array<f32, IxDyn>> = Vec::with_capacity(results.len());
            let mut filter_landmarks: Vec<Array<f32, Dim<[Ix; 2]>>> = Vec::with_capacity(results.len());

            for (_, result) in results.iter().enumerate() {
                let (bbox, score, _, landmark) = result;

                let bbox_offset = Array::from_vec(vec![off_x, off_y, off_x, off_y])
                    .into_shape(IxDyn(&[1, 4]))?;

                let offset_bbox = bbox - bbox_offset;
                let landmark_offset = Array::from_vec(vec![off_x, off_y, off_x, off_y, off_x, off_y, off_x, off_y, off_x, off_y])
                    .into_shape(IxDyn(&[1, 10]))?;
                let offset_landmark = landmark - landmark_offset;

                let dx = landmark[0] - landmark[2];
                let dy = landmark[1] - landmark[3];

                let eye_dist = (dx.powi(2) + dy.powi(2)).sqrt();

                if eye_distance_threshold.is_some() && eye_dist < eye_distance_threshold.unwrap() {
                    continue
                }
                if score < &score_threshold.unwrap_or(0.5) {
                    continue
                }
                let filter_landmark = offset_landmark.into_shape([5, 2]).unwrap().to_owned();
                let filter_bbox = offset_bbox.axis_iter(Axis(0)).collect::<Vec<_>>()[0].to_owned();
                filter_bboxes.push(filter_bbox);
                filter_landmarks.push(filter_landmark);
            }

            if filter_bboxes.len() == 0 {
                if keep_largest.unwrap_or(false) || keep_center.unwrap_or(false) {
                    filter_bboxes = vec![];
                    filter_landmarks = vec![];
                    batch_bboxes.push(filter_bboxes);
                    batch_landmarks.push(filter_landmarks);
                }
                continue
            }


            if keep_largest.unwrap_or(false) {
                let img_size = batch_imgs[idx].size().unwrap();
                let w = img_size.width;
                let h = img_size.height;
                let (bboxes, largest_idx) = get_largest_face(filter_bboxes, h, w);
                let largest_landmarks = filter_landmarks[largest_idx].to_owned();
                filter_bboxes = vec![bboxes];
                filter_landmarks = vec![largest_landmarks];
            }

            else if  keep_center.unwrap_or(false) {
                let img_size = batch_imgs[idx].size().unwrap();
                let w = img_size.width;
                let h = img_size.height;
                let (bboxes, center_idx) = get_center_face(filter_bboxes.clone(), Some(h), Some(w), None);
                let lmk = filter_landmarks[center_idx].to_owned();
                filter_bboxes = vec![bboxes];
                filter_landmarks = vec![lmk];
            }
            batch_bboxes.push(filter_bboxes);
            batch_landmarks.push(filter_landmarks);
        }
        Ok((batch_bboxes, batch_landmarks))
    }

    pub fn align_warp_faces(&self, imgs: &Vec<Mat>, lmk: &Vec<Array2<f32>> , border_mode: Option<i32>) -> Result<(Vec<Mat>, Vec<Mat>), Error> {
        let _border_mode = border_mode.unwrap_or(BORDER_CONSTANT);
        let mut affine_matrices: Vec<Mat> = Vec::with_capacity(imgs.len());
        let mut cropped_faces: Vec<Mat> = Vec::with_capacity(imgs.len());

        for (img, landmark) in imgs.into_iter().zip(lmk) {
            let vec_lmk = array2_to_vector_of_point2f(&landmark);
            let vec_face_template = array2_to_vector_of_point2f(&self.face_template);
            let mut inliers = Mat::default();
            let affine_matrix = estimate_affine_partial_2d(
                &vec_lmk,
                &vec_face_template,
                &mut inliers,
                LMEDS,
                3.0,
                2000,
                0.99,
                10,
            )?;
            affine_matrices.push(affine_matrix.clone());

            let affine_matrix_f64 = Array2::from_shape_vec(
                (2, 3),
                affine_matrices[0].data_typed::<f64>().unwrap().to_vec()
            )?;
            let affine_matrix_f32 =  affine_matrix_f64.map(|&x| x as f32);
            let mut cropped_face = Mat::default();
            warp_affine(
                &img,
                &mut cropped_face,
                &affine_matrix,
                Size::new(self.face_size.0, self.face_size.1),
                INTER_LINEAR,
                _border_mode,
                Scalar::from((0, 0, 0)),
            )?;

            let mut rgb_img = Mat::default();
            cvt_color(&cropped_face, &mut rgb_img, COLOR_BGR2RGB, 0)?;
            cropped_faces.push(cropped_face);
        }
        Ok((cropped_faces, affine_matrices))
    }

    pub fn align_face_idcard(&self, img: Mat, lmk: Array2<f32>, bbox: Array<f32, IxDyn>, border_mode: Option<i32>) -> Result<(Mat, Mat), Error>{
        let _border_mode = border_mode.unwrap_or(BORDER_CONSTANT);
        let size: (i32, i32) = (240, 320);

        let face_template_adult: Array2<f32> = Array2::from(vec![
            [87.56117786, 140.95207892],
            [152.12076214, 140.5917773],
            [120.04617, 177.99955243],
            [93.52425322, 216.13514054],
            [146.98728108, 215.83676865],
        ]);

        let face_template_baby: Array2<f32> = Array2::from(vec![
            [89.26848429, 149.95460108],
            [150.43019571, 149.6132627],
            [120.04374, 185.05220757],
            [94.91771357, 221.18065946],
            [145.56689786, 220.89799136],
        ]);

        let reshaped = bbox.into_shape((2, 2))?;
        let center = match reshaped.mean_axis(Axis(0)) {
            None => {
                return Err(Error::msg("invalid bbox dimensions"))
            }
            Some(center) => {center}
        };

        /// Calculate vectors
        /// Calculate the cross product (determinant in 2D)
        /// Calculate the norm of vec_lmk
        /// Calculate the final result
        let lmk0 = Vector2::new(lmk[[0, 0]], lmk[[0, 1]]);
        let lmk1 = Vector2::new(lmk[[1, 0]], lmk[[1, 1]]);
        let lmk2 = Vector2::new(lmk[[2, 0]], lmk[[2, 1]]);

        let vec_lmk = lmk0 - lmk1;
        let vec_center = Vector2::new(center[0], center[1]) - lmk1;
        let cross_product = vec_lmk.x * vec_center.y - vec_lmk.y * vec_center.x;
        let norm_vec_lmk = vec_lmk.norm();
        let d_cbox_eyes = cross_product / norm_vec_lmk;

        let vec_lmk = lmk0 - lmk1;
        let vec_center = lmk2 - lmk1;
        let cross_product = vec_lmk.x * vec_center.y - vec_lmk.y * vec_center.x;
        let norm_vec_lmk = vec_lmk.norm();
        let d_nose_eyes = cross_product / norm_vec_lmk;

        let ratio_adult: f32 = 0.565;
        let ratio_baby: f32 = 0.306;

        let face_ratio = f32::min(f32::max(d_cbox_eyes / d_nose_eyes, ratio_baby), ratio_adult);

        let face_template: Array2<f32> = (ratio_adult * &face_template_baby - ratio_baby * &face_template_adult
            + face_ratio * (&face_template_adult - &face_template_baby))
            / (ratio_adult - ratio_baby);

        let vec_lmk = array2_to_vector_of_point2f(&lmk);
        let vec_face_template = array2_to_vector_of_point2f(&face_template);
        let mut inliers = Mat::default();
        let affine_matrix = estimate_affine_partial_2d(
            &vec_lmk,
            &vec_face_template,
            &mut inliers,
            LMEDS,
            3.0,
            2000,
            0.99,
            10,
        )?;

        let affine_matrix_f64 = Array2::from_shape_vec(
            (2, 3),
            affine_matrix.data_typed::<f64>().unwrap().to_vec()
        )?;
        let affine_matrix_f32 =  affine_matrix_f64.map(|&x| x as f32);

        let mut cropped_img = Mat::default();
        warp_affine(&img, &mut cropped_img, &affine_matrix, Size::new(size.0, size.1), INTER_LINEAR, _border_mode, Scalar::from((0, 0, 0)))?;

        let mut rgb_img = Mat::default();
        cvt_color(&cropped_img, &mut rgb_img, COLOR_BGR2RGB, 0)?;

        Ok((rgb_img, affine_matrix))
    }


    pub fn align_fas_faces(&self, imgs: &Vec<Mat>, lmk: &Vec<Array2<f32>>, border_mode: Option<i32>) -> Result<(Vec<Mat>, Vec<Mat>), Error> {
        let _border_mode = border_mode.unwrap_or(BORDER_CONSTANT);
        let mut affine_matrices: Vec<Mat> = Vec::with_capacity(imgs.len());
        let mut cropped_faces: Vec<Mat> = Vec::with_capacity(imgs.len());
        for (img, landmark) in imgs.into_iter().zip(lmk) {
            let vec_lmk = array2_to_vector_of_point2f(&landmark);
            let vec_fas_template = array2_to_vector_of_point2f(&self.fas_template);
            let mut inliers = Mat::default();
            let affine_matrix = estimate_affine_partial_2d(
                &vec_lmk,
                &vec_fas_template,
                &mut inliers,
                LMEDS,
                3.0,
                2000,
                0.99,
                10,
            )?;
            affine_matrices.push(affine_matrix.clone());

            let affine_matrix_f64 = Array2::from_shape_vec(
                (2, 3),
                affine_matrices[0].data_typed::<f64>().unwrap().to_vec()
            )?;
            let affine_matrix_f32 =  affine_matrix_f64.map(|&x| x as f32);
            let mut cropped_face = Mat::default();
            warp_affine(
                &img,
                &mut cropped_face,
                &affine_matrix,
                Size::new(self.fas_size.0, self.fas_size.1),
                INTER_LINEAR,
                _border_mode,
                Scalar::from((0, 0, 0)),
            )?;

            let mut rgb_img = Mat::default();
            cvt_color(&cropped_face, &mut rgb_img, COLOR_BGR2RGB, 0)?;
            cropped_faces.push(cropped_face);
        }
        Ok((cropped_faces, affine_matrices))
    }
}

fn array2_to_vector_of_point2f(array: &Array2<f32>) -> Vector<Point2f> {
    let mut vec = Vector::new();
    for i in 0..array.nrows() {
        let point = Point2f::new(array[[i, 0]], array[[i, 1]]);
        vec.push(point);
    }
    vec
}

pub fn get_largest_face(det_faces: Vec<Array<f32, IxDyn>>, h: i32, w: i32) -> (Array<f32, IxDyn>, usize){
    fn get_location(val: f32, length: f32) -> f32 {
        if val < 0.0 {
            return 0.0
        }
        else if val > length {
            return length
        }
        else {
            return val
        }
    }

    let mut face_areas: Vec<f32> = Vec::with_capacity(det_faces.len());
    for det_face in &det_faces {
        let left = get_location(det_face[0], w as f32);
        let right = get_location(det_face[2], w as f32);
        let top = get_location(det_face[1], w as f32);
        let bottom = get_location(det_face[3], w as f32);
        let face_area = (right - left) * (bottom - top);
        face_areas.push(face_area);
    }
    let largest_idx = face_areas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();

    (det_faces[largest_idx].to_owned(), largest_idx)
}

pub fn get_center_face(det_faces: Vec<Array<f32, IxDyn>>, h: Option<i32>, w: Option<i32>, center: Option<(i32, i32)>) -> (Array<f32, IxDyn>, usize) {
    let _h = h.unwrap_or(0);
    let _w = w.unwrap_or(0);
    let _center = center.unwrap_or((_w / 2, _h /2));

    let arr_center = Array::from_vec(vec![_center.0, _center.1]).to_owned();

    let mut center_dist: Vec<f32> = Vec::with_capacity(det_faces.len());
    for det_face in &det_faces {
        let face_center: Array<f32, Ix1> = ArrayBase::from_vec(vec![(det_face[0] + det_face[2]) / 2.0, (det_face[1] + det_face[3]) / 2.0]);
        let dx = face_center[0] - arr_center[0] as f32;
        let dy = face_center[1] - arr_center[1] as f32;
        let dist = (dx.powi(2) + dy.powi(2)).sqrt();
        center_dist.push(dist);
    }
    let center_idx = center_dist
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();

    (det_faces[center_idx].to_owned(), center_idx)

}

pub fn pad_image(img: &Mat, ratio: Option<f32>) -> Result<(Mat, (i32, i32)), Error>{
    let img_shape = img.size()?;
    let h = img_shape.height;
    let w = img_shape.width;
    let off_x = (ratio.unwrap_or(0.5) * w as f32) as i32;
    let off_y = (ratio.unwrap_or(0.5) * h as f32) as i32;

    let mut bordered_img = Mat::default();
    copy_make_border(
        &img,
        &mut bordered_img,
        off_y,
        off_y,
        off_x,
        off_x,
        BORDER_CONSTANT,
        Default::default(),
    )?;

    Ok((bordered_img, (off_x, off_y)))
}


#[cfg(test)]
mod tests {
}

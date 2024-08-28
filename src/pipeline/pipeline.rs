use anyhow::Error;
use ndarray::{Array, Array2, Dim, Ix, IxDyn, s};
use opencv::core::{Mat, MatTraitConst};
use serde::{Deserialize, Serialize};
use crate::helper::face_helper::{FaceHelper, get_center_face};
use crate::modules::face_anti_spoofing::FaceFASClient;
use crate::modules::face_id_client::FaceIDClient;
use crate::modules::face_quality_client::FaceQualityClient;

#[derive(Debug, Clone)]
pub struct EKYCPipeline {
    face_id: FaceIDClient,
    face_quality: FaceQualityClient,
    face_helper: FaceHelper,
    fas_crop: FaceFASClient,
    fas_full: FaceFASClient,
}

impl EKYCPipeline {

    /// new initializes new instance of the pipeline
    pub fn new(
        face_id_client: FaceIDClient,
        face_quality_client: FaceQualityClient,
        face_helper_client: FaceHelper,
        face_fas_crop_client: FaceFASClient,
        face_fas_full_client: FaceFASClient,
    ) -> Self {
        EKYCPipeline {
            face_id: face_id_client,
            face_quality: face_quality_client,
            face_helper: face_helper_client,
            fas_crop: face_fas_crop_client,
            fas_full: face_fas_full_client,
        }
    }

    /// similarity_score calculates the cosine similarity
    ///
    /// # Arguments
    /// * `a` - &Vec<f32>
    /// * `b` - &Vec<f32>
    ///
    /// # Returns
    /// * `f32`
    pub fn similarity_score(&self, a: &Vec<f32>, b: &Vec<f32>)  -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
        let norm_a = a.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|b| b.powi(2)).sum::<f32>().sqrt();
        let cosine = dot_product / (norm_a * norm_b);
        cosine
    }

    /// extract_embedding returns the vector representation of the opencv matrices.
    ///
    /// # Arguments
    /// * `raw_input_tensors` - A vector of opencv matrix vectors
    ///
    /// # Returns
    /// * `Result<Array<f32, IxDyn>, Error>`
    async fn extract_embedding(&self, raw_input_tensors: Vec<Vec<Mat>>) -> Result<Array<f32, IxDyn>, Error> {
        let result = self.face_id.infer_batch(&raw_input_tensors).await?;
        Ok(result)
    }

    /// get_face_vector extracts the input image and returns its float32 vectorized representation.
    ///
    /// # Arguments
    /// * `img` - OpenCV matrix
    /// * `landmark` - Option<Array2<f32>>
    ///
    /// # Returns
    /// * `Result<Array<f32, IxDyn>, Error>`
    pub async fn get_face_vector(&self, img: Mat, landmark: Option<Array2<f32>>) ->  Result<Array<f32, IxDyn>, Error> {

        let mut lmk: Vec<Array<f32, Dim<[Ix; 2]>>> = Vec::with_capacity(1);
        if landmark.is_none() {
            let (_, landmarks) = self.face_helper.get_face_landmarks_5(
                vec![img.clone()],
                None,
                Some(true),
                None,
                None,
                None,
            ).await?;
            lmk = landmarks[0].to_owned();
        }

        let (cropped_faces, _) = self.face_helper.align_warp_faces(&vec![img], &lmk, None)?;
        self.extract_embedding(vec![cropped_faces]).await
    }

    /// get_face_landmarks_5 extracts the face landmarks from the input images.
    ///
    /// # Arguments
    /// * `batch_imgs` - Vector of OpenCV matrices
    ///
    /// # Returns
    /// * `Vec<Vec<Array<f32, Dim<[Ix; 2]>>>>`
    pub async fn get_face_landmarks_5(&self, batch_imgs: Vec<Mat>) -> Result<Vec<Vec<Array<f32, Dim<[Ix; 2]>>>>, Error>{
        let (_, batch_landmarks) = self.face_helper.get_face_landmarks_5(batch_imgs, None, Some(true), None, None, Some(true)).await?;
        Ok(batch_landmarks)
    }

    /// get_face_from_selfie crop the face image from the input image.
    ///
    /// # Arguments
    /// * `img` - OpenCV matrix
    ///
    /// # Returns
    /// * `Mat`
    pub async fn get_face_from_selfie(&self, img: Mat) -> Result<Mat, Error> {
        let (bboxes, batch_landmarks) = self.face_helper.get_face_landmarks_5(
            vec![img.clone()],
            Some(true),
            None,
            None,
            None,
            None,
        ).await?;

        let boxes = bboxes[0].to_owned();
        let landmarks = batch_landmarks[0].to_owned();
        let lmk = landmarks[0].to_owned().into_shape([5,2]).unwrap();

        let (face_img, _) = self.face_helper.align_face_idcard(img, lmk, boxes[0].to_owned(), None)?;
        Ok(face_img)
    }

    /// get_face_from_id_card crop the face image from the id card.
    ///
    /// # Arguments
    /// * `img` - OpenCV matrix
    ///
    /// # Returns
    /// * `Mat`
    pub async fn get_face_from_id_card(&self, img: Mat) -> Result<Mat, Error>{
        let (bboxes, batch_landmarks) = self.face_helper.get_face_landmarks_5(
            vec![img.clone()],
            None,
            None,
            None,
            None,
            None,
        ).await?;
        if bboxes.len() == 0 {
            return Ok(Mat::default())
        }

        let boxes = bboxes[0].to_owned();
        let landmarks = batch_landmarks[0].to_owned();
        let img_shape = img.clone().size().unwrap();
        let (bbox, center_idx) = get_center_face(boxes, None, None, Some((img_shape.width/5, img_shape.height/2)));
        let lmk = landmarks[center_idx].to_owned().into_shape([5,2]).unwrap();
        let (face_img, _) = self.face_helper.align_face_idcard(img, lmk, bbox, None)?;

        Ok(face_img)
    }

    /// get_face_quality checks the quality of the input images.
    ///
    /// The number of input images and the input landmarks must be the same
    /// Returns a tuple of `(f32, bool)` where the first element is
    /// the full face liveness score, the second element is the face mask decision.
    ///
    /// # Arguments
    /// * `imgs` - A vector of opencv Matrices
    /// * `lmks` - A vector of two-dimensional arrays of face landmarks
    ///
    /// # Returns
    /// * `(f32, bool)`
    pub async fn get_face_quality(&self, imgs: Vec<Mat>, lmks: Vec<Array2<f32>>) -> Result<(f32, bool), Error> {

        let (cropped_faces, _) = self.face_helper.align_warp_faces(&imgs, &lmks, None)?;
        let score_cover = self.face_quality.infer_batch(&vec![cropped_faces]).await?;
        let face_mask_score: f32 = score_cover
            .iter()
            .filter_map(|array| array.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()))
            .max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(-1.0);

        let facemask: bool = face_mask_score > self.face_quality.threshold_cover;

        Ok((face_mask_score, facemask))
    }

    /// check_liveness checks the person liveness from the input images.
    ///
    /// The number of input images and the input landmarks must be the same
    /// Returns a tuple of `(f32, f32, bool)` where the first element is
    /// the full face liveness score, the second element is the cropped face liveness score
    /// and the third element is the liveness decision.
    ///
    /// # Arguments
    /// * `imgs` - A vector of opencv Matrices
    /// * `lmks` - A vector of two-dimensional arrays of face landmarks
    ///
    /// # Returns
    /// * `(f32, f32, bool)`
    pub async fn check_liveness(&self, imgs: Vec<Mat>, lmks: Vec<Array2<f32>>) -> Result<(f32, f32, bool), Error> {
        if imgs.len() != lmks.len() || imgs.len() != 3 {
            return Err(Error::msg("the number of images and landmarks must be 3"))
        }

        let (cropped_faces, _) = self.face_helper.align_fas_faces(&imgs, &lmks, None)?;
        let liveness_score_crop = self.fas_crop.infer_single(&cropped_faces).await?;
        let liveness_score_full = self.fas_full.infer_single(&imgs).await?;
        let is_liveness = liveness_score_crop >= self.fas_crop.threshold && liveness_score_full >= self.fas_full.threshold;

        println!("{liveness_score_crop}");
        println!("{liveness_score_full}");
        println!("{is_liveness}");

        Ok((liveness_score_full, liveness_score_crop, is_liveness))
    }

    /// check_liveness_passive  checks the person liveness from the input images.
    ///
    /// The number of input images and the input landmarks must be the same
    /// Returns a tuple of `(f32, f32, bool)` where the first element is
    /// the full face liveness score, the second element is the cropped face liveness score
    /// and the third element is the liveness decision.
    ///
    /// # Arguments
    /// * `full_imgs` - A vector of opencv Matrices
    /// * `crop_imgs` - A vector of opencv Matrices
    ///
    /// # Returns
    /// * `(f32, f32, bool)`
    pub async fn check_liveness_passive(&self, full_imgs: Vec<Mat>, crop_imgs: Vec<Mat>) -> Result<(f32, f32, bool), Error> {
        if full_imgs.len() != crop_imgs.len() || full_imgs.len() != 3 {
            return Err(Error::msg("the number of images must be 3"))
        }

        let liveness_score_crop = self.fas_crop.infer_single(&crop_imgs).await?;
        let liveness_score_full = self.fas_full.infer_single(&full_imgs).await?;
        let is_liveness = liveness_score_crop >= self.fas_crop.threshold && liveness_score_full >= self.fas_full.threshold;

        println!("{liveness_score_crop}");
        println!("{liveness_score_full}");
        println!("{is_liveness}");

        Ok((liveness_score_full, liveness_score_crop, is_liveness))
    }


    /// check_same_person verifies if the input images belong to the same person.
    ///
    /// The number of input images and the input landmarks must be the same
    /// Returns a tuple of `(f32, f32, bool)` where the first element is
    /// the similarity score between the 1st and 2nd input images, the second element
    /// is the similarity score between the 2nd and 3rd input images
    /// and the third element is the same person decision.
    ///
    /// # Arguments
    /// * `imgs` - A vector of opencv Matrices
    /// * `lmks` - A vector of two-dimensional arrays of face landmarks
    ///
    /// # Returns
    /// * `(f32, f32, bool)`
    pub async fn check_same_person(&self, imgs: Vec<Mat>, lmks: Vec<Array2<f32>>) -> Result<(f32, f32, bool), Error> {
        if imgs.len() != lmks.len() || imgs.len() != 3 {
            return Err(Error::msg("the number of images and landmarks must be 3"))
        }
        let (cropped_faces, _) = self.face_helper.align_warp_faces(&imgs, &lmks, None)?;
        let result_vectors = self.extract_embedding(vec![cropped_faces]).await?;
        let v_f = result_vectors.slice(s![0, ..]).to_vec();
        let v_m = result_vectors.slice(s![1, ..]).to_vec();
        let v_n = result_vectors.slice(s![2, ..]).to_vec();
        let score_fm = self.similarity_score(&v_f, &v_m);
        let score_mn = self.similarity_score(&v_m, &v_n);
        let is_same_person = score_fm >= self.face_id.threshold_same_person && score_mn >= self.face_id.threshold_same_person;

        println!("score_fm: {:?}", score_fm);
        println!("score_mn: {:?}", score_mn);
        println!("is_same_person: {:?}", is_same_person);
        Ok((score_fm, score_mn, is_same_person))
    }

    /// person_card_verify checks if the person face image matches with the face on the personal document
    ///
    /// # Arguments
    /// * `id_card` - OpenCV matrix of id card image
    /// * `i_f` - OpenCV matrix of person face
    /// * `in_lmk_f` - person face landmark
    ///
    /// # Returns
    /// * `(f32, bool)`
    pub async fn person_card_verify(&self, id_card: Mat, i_f: Mat, in_lmk_f: Option<Array2<f32>>) -> Result<(f32, bool), Error> {
        let (_, card_lmks) = self.face_helper.get_face_landmarks_5(
            vec![id_card.clone()],
            Some(true),
            None,
            None,
            None,
            None,
        ).await?;

        let mut lmk_f: Array2<f32> = Array2::zeros((5, 2));
        if card_lmks.is_empty() {
            return Err(Error::msg("cannot detect any face in card image"))
        }

        let card_lmk = card_lmks[0][0].to_owned();

        if in_lmk_f.is_none() {
            let (_, lmks) = self.face_helper.get_face_landmarks_5(
                vec![i_f.clone()],
                Some(true),
                None,
                None,
                None,
                None,
            ).await?;
            if lmks.is_empty() {
                return Err(Error::msg("cannot detect any landmark in face image"))
            }
            lmk_f = lmks[0][0].to_owned();
        } else {
            lmk_f = in_lmk_f.unwrap();
        }

        let (cropped_faces, _) = self.face_helper.align_warp_faces(&vec![id_card, i_f], &vec![card_lmk, lmk_f], None)?;
        let extracted_vectors = self.extract_embedding(vec![cropped_faces]).await?;
        let v_card = extracted_vectors.slice(s![0,..]).to_vec();;
        let v_f = extracted_vectors.slice(s![1,..]).to_vec();
        let similarity_score = self.similarity_score(&v_card, &v_f);
        let is_same_person = similarity_score >= self.face_id.threshold_same_ekyc;

        println!("{similarity_score}");
        println!("{is_same_person}");

        Ok((similarity_score, is_same_person))
    }
}


#[cfg(test)]
mod tests {
    use crate::config::config::{FaceDetectionConfig, FaceIDConfig, FaceQualityConfig, FASConfig};
    use crate::helper::face_helper::FaceHelper;
    use crate::modules::face_anti_spoofing::FaceFASClient;
    use crate::modules::face_detection_client::FaceDetectionClient;
    use crate::modules::face_id_client::FaceIDClient;
    use crate::modules::face_quality_client::FaceQualityClient;
    use crate::pipeline::pipeline::EKYCPipeline;
    use crate::triton_client::client::triton::ModelConfigRequest;
    use crate::triton_client::client::TritonInferenceClient;
    use crate::utils::coordinate::FaceLandmarkMetadata;
    use crate::utils::image::{convert_image_to_ndarray, convert_json_metadata_to_ndarray};

    #[tokio::test]
    async fn test_ekyc_pipeline_face_quality() {

    }


}
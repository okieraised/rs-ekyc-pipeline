use std::cmp::max;
use anyhow::Error;
use ndarray::{Array, Array3, Array4, ArrayBase, Axis, concatenate, IxDyn, OwnedRepr, s};
use opencv::core::{Mat, MatTraitConst, Rect, Scalar, Size, Vec3b};
use opencv::imgproc;
use opencv::imgproc::resize;
use crate::config::config::FaceDetectionConfig;
use crate::triton_client::client::triton::{InferTensorContents, ModelConfigResponse, ModelInferRequest};
use crate::triton_client::client::triton::model_infer_request::InferInputTensor;
use crate::triton_client::client::TritonInferenceClient;
use crate::utils::utils::{u8_to_f32_vec, u8_to_i32_vec};

#[derive(Debug, Clone)]
pub(crate) struct FaceDetectionClient {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: String,
    timeout: i32,
    mean: f32,
    scale: f32,
}

impl FaceDetectionClient {
    pub fn new(triton_infer_client: TritonInferenceClient,
               triton_model_config: ModelConfigResponse,
               config: FaceDetectionConfig
    ) -> Self {
        FaceDetectionClient {
            triton_infer_client,
            triton_model_config,
            model_name: config.model_name,
            timeout: config.timeout,
            mean: config.mean,
            scale: config.scale,
        }
    }

    fn preprocess(&self) {

    }

    pub fn preprocess_batch(&self, raw_input_tensors: &Vec<Vec<Mat>>) -> Result<(Vec<Array4<f32>>, Vec<(i32, i32)>), Error> {
        let inputs = &raw_input_tensors[0];
        let mut outputs: Vec<Array4<f32>> = Vec::with_capacity(1);
        let mut sizes: Vec<(i32, i32)> = Vec::with_capacity(2);

        let model_config = match self.triton_model_config.clone().config {
            None => {
                return Err(Error::msg("face_detection_client - face detection model config is empty"))
            }
            Some(model_config) => {
                model_config
            }
        };

        let input_shape = &model_config.input[0].dims;

        for input in inputs {
            let img_h = input.rows();
            let img_w = input.cols();
            let im_ratio = img_w as f32 / img_h as f32;
            sizes.push((img_w, img_h));

            let model_ratio = input_shape[2] as f32 / input_shape[1] as f32;

            let (new_width, new_height) = if im_ratio > model_ratio {
                let new_width = input_shape[2] as i32;
                let new_height = (new_width as f32 / im_ratio) as i32;
                (new_width, new_height)
            } else {
                let new_height = input_shape[1] as i32;
                let new_width = (new_height as f32 * im_ratio) as i32;
                (new_width, new_height)
            };

            let mut img_resized = Mat::default();
            resize(
                input,
                &mut img_resized,
                Size::new(new_width, new_height),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            let mut img_scaled = Mat::new_rows_cols_with_default(
                input_shape[1] as i32,
                input_shape[2] as i32,
                opencv::core::CV_8UC3,
                Scalar::all(0.0)
            )?;

            let mut roi = Mat::roi_mut(
                &mut img_scaled,
                Rect::new(0, 0, new_width, new_height),
            )?;

            img_resized.copy_to(&mut roi)?;
            let mut im_tensor = Array3::<f32>::zeros((input_shape[1] as usize, input_shape[2] as usize, input_shape[0] as usize));

            // Convert the image to float and normalize it
            for i in 0..3 {
                for y in 0..input_shape[1] as usize {
                    for x in 0..input_shape[2] as usize {
                        let pixel_value = img_scaled.at_2d::<Vec3b>(y as i32, x as i32).unwrap()[i];
                        im_tensor[[y, x, i]] = (pixel_value as f32 - self.mean) * self.scale;
                    }
                }
            }

            let transposed_tensors = im_tensor.permuted_axes([2, 0, 1]);
            outputs.push(transposed_tensors.insert_axis(Axis(0)));
        }
        Ok((outputs, sizes))
    }

    fn postprocess(&self) {

    }

    fn postprocess_batch(&self, outputs: Vec<Array<f32, IxDyn>>,  sizes: Vec<(i32, i32)>) -> Vec<Vec<(Array<f32, IxDyn>, f32, f32, Array<f32, IxDyn>)>> {
        let mut result: Vec<Vec<(Array<f32, IxDyn>, f32, f32, Array<f32, IxDyn>)>> = Vec::with_capacity(1);

        for b in 0..outputs[0].dim()[0] {
            let num_dets = &outputs[0].axis_iter(Axis(b)).collect::<Vec<_>>()[0];
            let boxes = &outputs[1].axis_iter(Axis(b)).collect::<Vec<_>>()[0];
            let scores = &outputs[2].axis_iter(Axis(b)).collect::<Vec<_>>()[0];
            let classes = &outputs[3].axis_iter(Axis(b)).collect::<Vec<_>>()[0];
            let landmarks = &outputs[4].axis_iter(Axis(b)).collect::<Vec<_>>()[0];
            let scale = max(sizes[b].0, sizes[b].1);

            let mut res: Vec<(Array<f32, IxDyn>, f32, f32, Array<f32, IxDyn>)> = Vec::with_capacity(num_dets.len());

            for i in 0..num_dets.len() {
                let score = scores[i];
                let class_id = classes[i];
                let face_box = &boxes.to_owned().axis_iter(Axis(i)).collect::<Vec<_>>()[0].to_owned() * scale as f32;
                let landmark = &landmarks.to_owned().axis_iter(Axis(i)).collect::<Vec<_>>()[0].to_owned() * scale as f32;
                res.push((face_box, score, class_id, landmark));
            }
            result.push(res);
        }
        result
    }

    pub async fn infer(&self) {

    }

    pub async fn infer_batch(&self, raw_input_tensors: &Vec<Vec<Mat>>) -> Result<Vec<Vec<(Array<f32, IxDyn>, f32, f32, Array<f32, IxDyn>)>>, Error> {
        let max_batch_size = self.triton_model_config.config.clone().unwrap().max_batch_size.to_owned();

        let (input_tensors, sizes) = self.preprocess_batch(raw_input_tensors)?;

        let batch_size = &input_tensors[0].dim().0;

        let model_config = match &self.triton_model_config.config {
            None => {
                return Err(Error::msg("face_detection_client - face detection model config is empty"))
            }
            Some(model_config) => {model_config}
        };

        let input_cfgs =  &model_config.input;
        let output_cfgs = &model_config.output;

        let mut outputs: Vec<Vec< ArrayBase<OwnedRepr<f32>, IxDyn>>> = vec![Vec::with_capacity(1); output_cfgs.len()];

        for idx in (0..input_tensors.len()).step_by(1usize) {

            let mut input_placeholders = Vec::<InferInputTensor>::with_capacity(input_cfgs.len());
            let mut model_request = ModelInferRequest{
                model_name: self.model_name.to_owned(),
                model_version: "".to_string(),
                id: "".to_string(),
                parameters: Default::default(),
                inputs: vec![],
                outputs: Default::default(),
                raw_input_contents: vec![],
            };

            for (_, input_cfg) in input_cfgs.iter().enumerate() {
                let mut sub_tensor: Vec<f32> = input_tensors[idx].clone().into_iter().collect();
                let model_input = InferInputTensor {
                    name: input_cfg.name.to_string(),
                    datatype: input_cfg.data_type().as_str_name()[5..].to_uppercase(),
                    shape: input_tensors[idx].shape().iter().map(|&x| x as i64).collect(),
                    parameters: Default::default(),
                    contents: Option::from(InferTensorContents {
                        bool_contents: vec![],
                        int_contents: vec![],
                        int64_contents: vec![],
                        uint_contents: vec![],
                        uint64_contents: vec![],
                        fp32_contents: sub_tensor,
                        fp64_contents: vec![],
                        bytes_contents: vec![],
                    }),
                };
                input_placeholders.push(model_input)
            }
            model_request.inputs = input_placeholders;
            let mut sub_result = self.triton_infer_client.model_infer(model_request).await?;

            for (oidx, output) in sub_result.outputs.iter().enumerate() {
                let c = output.to_owned().shape.len();
                let mut dimensions: Vec<usize> =  Vec::with_capacity(c);
                for dim in &output.shape {
                    dimensions.push(*dim as usize);
                }
                let u8_array: &[u8] = &sub_result.raw_output_contents[oidx];
                let mut f_vec: Vec<f32> = vec![];
                match output.datatype.as_str() {
                    "INT32" => {
                        f_vec = u8_to_i32_vec(u8_array).iter().map(|&x| x as f32).collect();
                    }
                    "FP32" => {
                        f_vec = u8_to_f32_vec(u8_array);
                    }
                    _ => {}
                };

                let f_arr = Array::from_shape_vec(dimensions, f_vec)?;
                outputs[oidx].push(f_arr);
            }
        }

        let concatenated_outputs: Vec<_> = outputs.into_iter()
            .map(|output| {
                concatenate(
                    Axis(0),
                    &output.iter().map(|array| array.view()).collect::<Vec<_>>()
                ).expect("Concatenation failed")
            })
            .collect();

        Ok(self.postprocess_batch(concatenated_outputs, sizes))
    }


}

#[cfg(test)]
mod tests {
}


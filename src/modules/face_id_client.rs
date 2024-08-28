use anyhow::Error;
use ndarray::{Array, Array3, Array4, ArrayBase, Axis, concatenate, IxDyn, OwnedRepr};
use opencv::core::{Mat, MatTraitConst, Size, Vec3b};
use opencv::imgproc;
use opencv::imgproc::resize;
use crate::config::config::FaceIDConfig;
use crate::triton_client::client::triton::{InferTensorContents, ModelConfigResponse, ModelInferRequest};
use crate::triton_client::client::triton::model_infer_request::InferInputTensor;
use crate::triton_client::client::TritonInferenceClient;
use crate::utils::utils::u8_to_f32_vec;

#[derive(Clone, Debug)]
pub(crate) struct FaceIDClient {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    pub model_name: String,
    pub timeout: i32,
    pub mean: f32,
    pub scale: f32,
    pub threshold_same_ekyc: f32,
    pub threshold_same_person: f32,
    pub imsize: (i32, i32),
}

impl FaceIDClient {
    pub fn new(triton_infer_client: TritonInferenceClient,
               triton_model_config: ModelConfigResponse,
               config: FaceIDConfig
    ) -> Self {
        FaceIDClient {
            triton_infer_client,
            triton_model_config,
            model_name: config.model_name,
            timeout: config.timeout,
            mean: config.mean,
            scale: config.scale,
            threshold_same_ekyc: config.threshold_same_ekyc,
            threshold_same_person: config.threshold_same_person,
            imsize: config.imsize,
        }
    }

    async fn preprocess(&self) {

    }

    fn preprocess_batch(&self, raw_input_tensors: &Vec<Vec<Mat>>) -> Result<(Vec<Array4<f32>>, Vec<(i32, i32)>), Error> {
        let inputs = raw_input_tensors[0].to_owned();
        let mut outputs: Vec<Array4<f32>> = Vec::with_capacity(1);
        let mut sizes: Vec<(i32, i32)> = Vec::with_capacity(raw_input_tensors.len());

        for input in inputs {
            let mut img_resized = Mat::default();
            resize(
                &input,
                &mut img_resized,
                Size::new(self.imsize.0, self.imsize.1),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            let img_shape = img_resized.size()?;
            sizes.push((img_shape.width, img_shape.height));

            let mut im_tensor = Array3::<f32>::zeros((img_shape.width as usize, img_shape.height as usize, 3usize));

            // Convert the image to float and normalize it
            for i in 0..3 {
                for y in 0..img_shape.width as usize {
                    for x in 0..img_shape.height as usize {
                        let pixel_value = img_resized.at_2d::<Vec3b>(y as i32, x as i32).unwrap()[i];
                        im_tensor[[y, x, i]] = (pixel_value as f32 - self.mean) * self.scale;
                    }
                }
            }
            let transposed_tensors = im_tensor.permuted_axes([2, 0, 1]);
            outputs.push(transposed_tensors.insert_axis(Axis(0)));
        }
        Ok((outputs, sizes))

    }

    async fn postprocess(&self) {

    }

    fn postprocess_batch(&self, outputs: Vec<Array<f32, IxDyn>>) -> Array<f32, IxDyn> {
        outputs[0].to_owned()
    }

    pub async fn infer_batch(&self, raw_input_tensors: &Vec<Vec<Mat>>) -> Result<Array<f32, IxDyn>, Error> {
        let (input_tensors, sizes) = self.preprocess_batch(raw_input_tensors)?;
        let model_config = match &self.triton_model_config.config {
            None => {
                return Err(Error::msg("face_id_client - face detection model config is empty"))
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
                f_vec = u8_to_f32_vec(u8_array);

                let f_arr =Array::from_shape_vec(dimensions, f_vec)?;
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

        Ok(self.postprocess_batch(concatenated_outputs))
    }
}
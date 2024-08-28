use anyhow::Error;
use ndarray::Array3;
use opencv::core::{Mat, MatTraitConst, Size, Vec3b};
use opencv::imgproc;
use opencv::imgproc::resize;
use crate::config::config::FASConfig;
use crate::triton_client::client::{TritonInferenceClient};
use crate::triton_client::client::triton::{InferTensorContents, ModelConfigResponse, ModelInferRequest};
use crate::triton_client::client::triton::model_infer_request::InferInputTensor;
use crate::utils::utils::u8_to_f32_vec;


#[derive(Clone, Debug)]
pub(crate) struct FaceFASClient {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    pub model_name: String,
    pub timeout: i32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub threshold: f32,
    pub imsize: (i32, i32),
}

impl FaceFASClient {
    pub fn new(triton_infer_client: TritonInferenceClient,
               triton_model_config: ModelConfigResponse,
               config: FASConfig) -> Self {
        FaceFASClient {
            triton_infer_client,
            triton_model_config,
            model_name: config.model_name,
            timeout: config.timeout,
            mean: config.mean,
            std: config.std,
            threshold: config.threshold,
            imsize: config.imsize,
        }
    }

    fn preprocess(&self, img: &Mat) -> Result<Array3<f32>, Error>{
        let mut img_resized = Mat::default();
        resize(
            &img,
            &mut img_resized,
            Size::new(self.imsize.0, self.imsize.1),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let img_shape = img_resized.size()?;

        let mut im_tensor = Array3::<f32>::zeros((img_shape.width as usize, img_shape.height as usize, 3usize));

        // Convert the image to float and normalize it
        for i in 0..3 {
            for y in 0..img_shape.width as usize {
                for x in 0..img_shape.height as usize {
                    let pixel_value = img_resized.at_2d::<Vec3b>(y as i32, x as i32).unwrap()[i];
                    im_tensor[[y, x, i]] = ((pixel_value as f32 / 255.0) - self.mean[i]) / self.std[i];
                }
            }
        }
        let transposed_tensors = im_tensor.permuted_axes([2, 0, 1]);

        Ok(transposed_tensors)

    }

    pub async fn infer_single(&self, imgs: &Vec<Mat>) -> Result<f32, Error> {
        if imgs.len() != 3 {
            return Err(Error::msg("the number of input images must be 3"))
        }

        let i_f = self.preprocess(&imgs[0])?;
        let i_m = self.preprocess(&imgs[1])?;
        let i_n = self.preprocess(&imgs[2])?;

        let model_config = match &self.triton_model_config.config {
            None => {
                return Err(Error::msg("face_fas_client - face fas model config is empty"))
            }
            Some(model_config) => {model_config}
        };

        let input_cfgs =  &model_config.input;
        let output_cfgs = &model_config.output;

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

        let i_f_flatten = i_f.into_iter().collect();
        let i_m_flatten = i_m.into_iter().collect();
        let i_n_flatten = i_n.into_iter().collect();

        let sub_tensor : Vec<Vec<f32>> = vec![i_f_flatten, i_m_flatten, i_n_flatten];

        for (i, input_cfg) in input_cfgs.iter().enumerate() {
            let model_input = InferInputTensor {
                name: input_cfg.name.to_string(),
                datatype: input_cfg.data_type().as_str_name()[5..].to_uppercase(),
                shape: input_cfg.clone().dims,
                parameters: Default::default(),
                contents: Option::from(InferTensorContents {
                    bool_contents: vec![],
                    int_contents: vec![],
                    int64_contents: vec![],
                    uint_contents: vec![],
                    uint64_contents: vec![],
                    fp32_contents: sub_tensor[i].clone(),
                    fp64_contents: vec![],
                    bytes_contents: vec![],
                }),
            };
            input_placeholders.push(model_input)
        }
        model_request.inputs = input_placeholders;
        let mut sub_result = self.triton_infer_client.model_infer(model_request).await?;

        let mut score: f32 = 0.0;
        for (oidx, output) in sub_result.outputs.iter().enumerate() {
            let c = output.to_owned().shape.len();
            let mut dimensions: Vec<usize> =  Vec::with_capacity(c);
            for dim in &output.shape {
                dimensions.push(*dim as usize);
            }
            let u8_array: &[u8] = &sub_result.raw_output_contents[oidx];
            let mut f_vec: Vec<f32> = vec![];

            f_vec = u8_to_f32_vec(u8_array);
            for (idx, val) in f_vec.iter().enumerate() {
                if idx == 1 {
                    score = val.to_owned();
                }
            }
        }
        Ok(score)
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_face_antispoofing_client() {

    }
}


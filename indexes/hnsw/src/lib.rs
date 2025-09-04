mod builder;
mod index;
mod kind;

pub use builder::*;
pub use index::*;
pub use kind::*;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Euclidean;

impl space::Metric<Vec<f32>> for Euclidean {
    type Unit = u32;

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits()
    }
}

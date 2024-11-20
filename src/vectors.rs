pub struct VectorsOps;

impl VectorsOps {
    pub fn weighted_average(
        e1: &Vec<f32>, // Embedding of the paragraph
        w1: f32,       // Weight for the paragraph
        e2: &Vec<f32>, // Embedding of the heavier sentence
        w2: f32,
        normalize_result: bool, // Weight for the heavier sentence
    ) -> Vec<f32> {
        let total_weight = w1 + w2;
        // Compute the weighted average
        let weighted_avg: Vec<f32> = e1
            .iter()
            .zip(e2.iter())
            .map(|(&x1, &x2)| (w1 * x1 + w2 * x2) / total_weight)
            .collect();
        // Normalize the weighted average embedding
        if normalize_result {
            Self::normalize(&weighted_avg)
        } else {
            weighted_avg
        }
    }

    fn normalize(v: &Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Avoid division by zero
        if norm == 0.0 {
            return v.clone();
        }
        v.iter().map(|x| x / norm).collect()
    }

    fn cosine_similarity(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
        v1.iter().zip(v2.iter()).map(|(&x1, &x2)| x1 * x2).sum()
    }

    fn find_closest_embedding<'a>(
        query: &Vec<f32>,
        embeddings: &'a [Vec<f32>],
    ) -> Option<&'a Vec<f32>> {
        embeddings.iter().max_by(|a, b| {
            Self::cosine_similarity(query, a)
                .partial_cmp(&Self::cosine_similarity(query, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

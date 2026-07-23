use std::collections::HashMap;

use crate::skill_graph::graph_store::SkillGraphStore;

/// Link prediction result.
#[derive(Debug, Clone)]
pub struct LinkPrediction {
    pub source_iri: String,
    pub target_iri: String,
    pub score: f32,
    pub existing_link: bool,
}

/// Similarity and link prediction engine using node embeddings.
pub struct SimilarityEngine;

impl SimilarityEngine {
    /// Compute cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Predict missing links for a skill based on embedding similarity.
    ///
    /// For each candidate skill (not already linked), computes cosine similarity
    /// between their embeddings. Returns top-K candidates sorted by score.
    ///
    /// `embeddings`: map of skill_iri → embedding vector (e.g., from NeighborhoodAggregator).
    /// `store`: skill graph store for checking existing links.
    /// `source_iri`: the skill to find link candidates for.
    /// `top_k`: number of candidates to return.
    pub fn predict_links(
        embeddings: &HashMap<String, Vec<f32>>,
        store: &SkillGraphStore,
        source_iri: &str,
        top_k: usize,
    ) -> Vec<LinkPrediction> {
        let source_emb = match embeddings.get(source_iri) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let skill = match store.get_skill(source_iri) {
            Some(s) => s,
            None => return Vec::new(),
        };

        // Existing links
        let existing_targets: Vec<&str> =
            skill.links.iter().map(|l| l.target_iri.as_str()).collect();

        let mut scored: Vec<LinkPrediction> = embeddings
            .iter()
            .filter(|(iri, _)| iri.as_str() != source_iri)
            .map(|(iri, emb)| {
                let existing = existing_targets.contains(&iri.as_str());
                let score = Self::cosine_similarity(source_emb, emb);
                LinkPrediction {
                    source_iri: source_iri.to_string(),
                    target_iri: iri.clone(),
                    score,
                    existing_link: existing,
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// Find skills most similar to a query embedding.
    pub fn find_similar(
        embeddings: &HashMap<String, Vec<f32>>,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = embeddings
            .iter()
            .map(|(iri, emb)| {
                let sim = Self::cosine_similarity(query_embedding, emb);
                (iri.clone(), sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = SimilarityEngine::cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let sim = SimilarityEngine::cosine_similarity(&a, &c);
        assert!((sim - 1.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        let sim = SimilarityEngine::cosine_similarity(&a, &d);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_predict_links() {
        let store = SkillGraphStore::new();
        let mut embeddings = HashMap::new();
        embeddings.insert("iri://skills/a".to_string(), vec![1.0, 0.0, 0.0]);
        embeddings.insert("iri://skills/b".to_string(), vec![0.95, 0.1, 0.0]);
        embeddings.insert("iri://skills/c".to_string(), vec![0.0, 1.0, 0.0]);

        // Register source skill so predict_links can look up existing links
        let node_a = crate::skill_graph::types::SkillGraphNode::new(
            "iri://skills/a",
            "Skill A",
            "Test skill",
        );
        store.register_skill(node_a).unwrap();

        let result = SimilarityEngine::predict_links(&embeddings, &store, "iri://skills/a", 2);
        assert_eq!(result.len(), 2);
        // b is most similar to a
        assert_eq!(result[0].target_iri, "iri://skills/b");
        assert!(!result[0].existing_link);
    }

    #[test]
    fn test_find_similar() {
        let mut embeddings = HashMap::new();
        embeddings.insert("a".to_string(), vec![1.0, 0.0]);
        embeddings.insert("b".to_string(), vec![0.9, 0.1]);
        embeddings.insert("c".to_string(), vec![0.0, 1.0]);

        let query = vec![1.0, 0.0];
        let result = SimilarityEngine::find_similar(&embeddings, &query, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "a");
    }
}

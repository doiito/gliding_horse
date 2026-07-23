//! End-to-end integration test for HyperspaceEngine.
//!
//! Tests the full lifecycle: insert → search → delete → crash recovery.

use std::path::Path;

use hyperspace_engine::filter::JsonLdFilter;
use hyperspace_engine::hnsw::HnswConfig;
use hyperspace_engine::hyper_vector::{EmbeddingVector, MetricKind};
use hyperspace_engine::metric::CosineMetric;
use hyperspace_engine::wal::WalSyncMode;
use hyperspace_engine::HyperspaceEngine;
use hyperspace_engine::HyperspaceEngineImpl;
use serde_json::json;

fn v(coords: Vec<f64>) -> EmbeddingVector {
    EmbeddingVector::new_unchecked(coords, MetricKind::Cosine)
}

fn setup(dir: &Path) -> HyperspaceEngineImpl {
    HyperspaceEngineImpl::open(
        dir,
        WalSyncMode::Strict,
        4,
        Box::new(CosineMetric),
        HnswConfig::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert(
        "vec:0",
        v(vec![1.0, 0.0, 0.0, 0.0]),
        json!({"label": "x_axis", "@type": ["axis"]}),
    )
    .await
    .unwrap();
    eng.insert(
        "vec:1",
        v(vec![0.0, 1.0, 0.0, 0.0]),
        json!({"label": "y_axis", "@type": ["axis"]}),
    )
    .await
    .unwrap();
    eng.insert(
        "vec:2",
        v(vec![0.0, 0.0, 1.0, 0.0]),
        json!({"label": "z_axis", "@type": ["axis"]}),
    )
    .await
    .unwrap();

    assert_eq!(eng.count().await.unwrap(), 3);

    let results = eng
        .search(&v(vec![0.99, 0.01, 0.0, 0.0]), 3, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].iri, "vec:0",
        "x_axis should be nearest to itself"
    );
    assert!(
        results[0].score.abs() < 0.1,
        "score to x_axis should be small"
    );

    eng.delete("vec:1").await.unwrap();
    assert_eq!(eng.count().await.unwrap(), 2);

    let results = eng
        .search(&v(vec![1.0, 1.0, 1.0, 0.0]), 3, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 2, "should return only 2 after delete");
}

#[tokio::test]
async fn test_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();

    {
        let eng = setup(dir.path());
        eng.insert("r:0", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"id": "first"}))
            .await
            .unwrap();
        eng.insert("r:1", v(vec![0.0, 1.0, 0.0, 0.0]), json!({"id": "second"}))
            .await
            .unwrap();
        eng.insert("r:2", v(vec![0.5, 0.5, 0.0, 0.0]), json!({"id": "third"}))
            .await
            .unwrap();
        eng.checkpoint().await.unwrap();
    }

    {
        let eng = setup(dir.path());
        assert_eq!(eng.count().await.unwrap(), 3, "all vectors recovered");

        let results = eng
            .search(&v(vec![1.0, 0.01, 0.0, 0.0]), 3, &[])
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].iri, "r:0", "first should be nearest to x_axis");
    }
}

#[tokio::test]
async fn test_upsert_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert(
        "up:10",
        v(vec![1.0, 0.0, 0.0, 0.0]),
        json!({"label": "original"}),
    )
    .await
    .unwrap();
    eng.upsert(
        "up:10",
        v(vec![0.0, 1.0, 0.0, 0.0]),
        json!({"label": "updated"}),
    )
    .await
    .unwrap();

    assert_eq!(eng.count().await.unwrap(), 1);

    let results = eng
        .search(&v(vec![0.0, 1.0, 0.0, 0.0]), 5, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].iri, "up:10");
}

#[tokio::test]
async fn test_search_empty_engine() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());
    let results = eng
        .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 5, &[])
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_delete_and_reinsert() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert("d:0", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"n": 1}))
        .await
        .unwrap();
    eng.delete("d:0").await.unwrap();
    assert_eq!(eng.count().await.unwrap(), 0);

    eng.insert("d:0", v(vec![0.0, 1.0, 0.0, 0.0]), json!({"n": 2}))
        .await
        .unwrap();
    assert_eq!(eng.count().await.unwrap(), 1);

    let results = eng
        .search(&v(vec![0.0, 1.0, 0.0, 0.0]), 5, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].iri, "d:0");
}

#[tokio::test]
async fn test_100_insert_search_delete() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    let n = 100u32;
    for i in 0..n {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n as f64);
        let vector = EmbeddingVector::new_unchecked(
            vec![angle.cos(), angle.sin(), 0.0, 0.0],
            MetricKind::Cosine,
        );
        let iri = format!("circ:{i}");
        eng.insert(&iri, vector, json!({"idx": i})).await.unwrap();
    }
    assert_eq!(eng.count().await.unwrap(), n as u64);

    let query = v(vec![1.0, 0.0, 0.0, 0.0]);
    let results = eng.search(&query, 5, &[]).await.unwrap();
    assert_eq!(results.len(), 5);
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score - 1e-6,
            "scores must be sorted descending"
        );
    }
    let mut ids: Vec<u32> = results.iter().map(|r| r.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), results.len(), "no duplicate IDs in results");

    let results = eng.search(&query, 1000, &[]).await.unwrap();
    assert!(results.len() >= 10, "search should find at least 10 nodes");
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score - 1e-6,
            "scores must be sorted descending"
        );
    }

    for i in 10..20u32 {
        let iri = format!("circ:{i}");
        eng.delete(&iri).await.unwrap();
    }
    assert_eq!(eng.count().await.unwrap(), (n - 10) as u64);

    let results = eng.search(&query, 100, &[]).await.unwrap();
    for hit in &results {
        assert!(
            hit.id < 11 || hit.id > 20,
            "deleted vector {} should not appear (ids 11..=20 were deleted)",
            hit.id
        );
    }
}

/// Deterministic mutation/recovery pressure guard. This is deliberately not a
/// throughput benchmark: it verifies that repeated upsert/delete operations
/// keep the vector store, HNSW, metadata filter index and persistent payload
/// view in agreement across a checkpoint/reopen boundary.
#[tokio::test]
async fn test_mutation_churn_preserves_index_metadata_and_recovery_consistency() {
    let dir = tempfile::tempdir().unwrap();
    const COUNT: u32 = 64;
    let deleted = |index: u32| index.is_multiple_of(5);
    let updated = |index: u32| index.is_multiple_of(3) && !deleted(index);

    {
        let eng = setup(dir.path());
        for index in 0..COUNT {
            let axis = (index % 4) as usize;
            let mut coords = vec![0.05; 4];
            coords[axis] = 1.0;
            eng.insert(
                &format!("churn:{index}"),
                v(coords),
                json!({"@type": ["Original"], "revision": "v1", "index": index}),
            )
            .await
            .unwrap();
        }
        for index in 0..COUNT {
            if index % 3 == 0 {
                let axis = ((index + 1) % 4) as usize;
                let mut coords = vec![0.05; 4];
                coords[axis] = 1.0;
                eng.upsert(
                    &format!("churn:{index}"),
                    v(coords),
                    json!({"@type": ["Updated"], "revision": "v2", "index": index}),
                )
                .await
                .unwrap();
            }
            if deleted(index) {
                eng.delete(&format!("churn:{index}")).await.unwrap();
            }
        }
        eng.checkpoint().await.unwrap();
    }

    let eng = setup(dir.path());
    let expected_count = (0..COUNT).filter(|index| !deleted(*index)).count() as u64;
    assert_eq!(eng.count().await.unwrap(), expected_count);

    let all = eng.list(0, COUNT as usize).await.unwrap();
    assert_eq!(all.len() as u64, expected_count);
    for hit in &all {
        let index = hit
            .iri
            .strip_prefix("churn:")
            .unwrap()
            .parse::<u32>()
            .unwrap();
        assert!(
            !deleted(index),
            "deleted IRI must not remain listed: {}",
            hit.iri
        );
        assert_eq!(
            hit.payload.as_ref().expect("listed hit payload")["index"],
            index
        );
    }

    let query = v(vec![1.0, 0.0, 0.0, 0.0]);
    let original_hits = eng
        .search(
            &query,
            COUNT as usize,
            &[JsonLdFilter::Type("Original".to_string())],
        )
        .await
        .unwrap();
    assert_eq!(
        original_hits.len(),
        (0..COUNT)
            .filter(|index| !deleted(*index) && !updated(*index))
            .count()
    );
    assert!(original_hits.iter().all(|hit| {
        hit.payload
            .as_ref()
            .is_some_and(|payload| payload["revision"] == "v1")
    }));

    let updated_hits = eng
        .search(
            &query,
            COUNT as usize,
            &[JsonLdFilter::Type("Updated".to_string())],
        )
        .await
        .unwrap();
    assert_eq!(
        updated_hits.len(),
        (0..COUNT).filter(|index| updated(*index)).count()
    );
    assert!(updated_hits.iter().all(|hit| {
        hit.payload
            .as_ref()
            .is_some_and(|payload| payload["revision"] == "v2")
    }));
}

/// Deterministic correctness guard, not an ANN performance benchmark. For a
/// small orthogonal fixture, an exact stored vector must retrieve its own IRI
/// at rank one. This catches basic HNSW/registry regressions that aggregate
/// count or ordering assertions do not expose.
#[tokio::test]
async fn test_small_fixture_exact_query_recall_at_one() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());
    // `setup` intentionally opens a 4-dimensional engine; use the complete
    // orthogonal basis for that configured dimension rather than bypassing the
    // engine's dimension validation.
    const DIMENSIONS: usize = 4;

    for index in 0..DIMENSIONS {
        let mut coords = vec![0.0; DIMENSIONS];
        coords[index] = 1.0;
        eng.insert(
            &format!("basis:{index}"),
            v(coords),
            json!({"fixture": "orthogonal", "index": index}),
        )
        .await
        .unwrap();
    }

    let mut correct = 0usize;
    for index in 0..DIMENSIONS {
        let mut coords = vec![0.0; DIMENSIONS];
        coords[index] = 1.0;
        let hit = eng.search(&v(coords), 1, &[]).await.unwrap();
        assert_eq!(hit.len(), 1);
        if hit[0].iri == format!("basis:{index}") {
            correct += 1;
        }
    }
    assert_eq!(correct, DIMENSIONS, "small-fixture recall@1 must be 100%");
}

#[tokio::test]
async fn test_get_vector_and_resolve_iri() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert("vec:a", v(vec![0.5, 0.5, 0.0, 0.0]), json!({"label": "A"}))
        .await
        .unwrap();
    eng.insert("vec:b", v(vec![0.1, 0.9, 0.0, 0.0]), json!({"label": "B"}))
        .await
        .unwrap();

    let id_a = eng.resolve_iri("vec:a").await.unwrap().unwrap();
    let id_b = eng.resolve_iri("vec:b").await.unwrap().unwrap();
    assert_eq!(id_a, 1, "first IRI gets id 1");
    assert_eq!(id_b, 2, "second IRI gets id 2");

    let vec = eng.get_vector("vec:a").await.unwrap().unwrap();
    assert!((vec.coords[0] - 0.5).abs() < 1e-10);
    assert!((vec.coords[1] - 0.5).abs() < 1e-10);

    let nonexistent = eng.resolve_iri("vec:unknown").await.unwrap();
    assert!(nonexistent.is_none());

    let no_vec = eng.get_vector("vec:unknown").await.unwrap();
    assert!(no_vec.is_none());
}

#[tokio::test]
async fn test_get_payload_details() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert(
        "p:0",
        v(vec![1.0, 0.0, 0.0, 0.0]),
        json!({"@type": ["Test"], "value": 42}),
    )
    .await
    .unwrap();

    let payload = eng.get_payload("p:0").await.unwrap().unwrap();
    assert_eq!(payload["value"], 42);
    assert_eq!(payload["@type"][0], "Test");

    let none = eng.get_payload("p:missing").await.unwrap();
    assert!(none.is_none());
}

#[tokio::test]
async fn test_list_with_pagination() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    for i in 0..10u32 {
        let iri = format!("l:{}", i);
        eng.insert(
            &iri,
            v(vec![1.0 - i as f64 * 0.1, 0.0, 0.0, 0.0]),
            json!({"idx": i}),
        )
        .await
        .unwrap();
    }

    let page1 = eng.list(0, 3).await.unwrap();
    assert_eq!(page1.len(), 3);

    let page2 = eng.list(3, 3).await.unwrap();
    assert_eq!(page2.len(), 3);

    let all = eng.list(0, 100).await.unwrap();
    assert_eq!(all.len(), 10);

    for hit in &all {
        assert!(!hit.iri.is_empty(), "each hit should have an IRI");
    }
}

#[tokio::test]
async fn test_filter_search() {
    use hyperspace_engine::filter::JsonLdFilter;

    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    eng.insert(
        "f:0",
        v(vec![1.0, 0.0, 0.0, 0.0]),
        json!({"@type": ["cat"], "age": 3}),
    )
    .await
    .unwrap();
    eng.insert(
        "f:1",
        v(vec![0.8, 0.2, 0.0, 0.0]),
        json!({"@type": ["dog"], "age": 5}),
    )
    .await
    .unwrap();
    eng.insert(
        "f:2",
        v(vec![0.6, 0.4, 0.0, 0.0]),
        json!({"@type": ["cat"], "age": 1}),
    )
    .await
    .unwrap();
    eng.insert(
        "f:3",
        v(vec![0.0, 1.0, 0.0, 0.0]),
        json!({"@type": ["bird"], "age": 2}),
    )
    .await
    .unwrap();

    let cat_filter = JsonLdFilter::Type("cat".to_string());
    let results = eng
        .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 10, &[cat_filter])
        .await
        .unwrap();
    assert_eq!(results.len(), 2, "should find both cats");
    for hit in &results {
        assert!(hit.iri == "f:0" || hit.iri == "f:2");
    }

    let age_filter = JsonLdFilter::Range {
        key: "age".to_string(),
        gte: Some(2.0),
        lte: Some(5.0),
    };
    let results = eng
        .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 10, &[age_filter])
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        3,
        "should find f:0(age=3), f:1(age=5), f:3(age=2)"
    );
}

#[tokio::test]
async fn test_upsert_with_resolve() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    let id1 = eng
        .upsert(
            "up:t",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({"label": "first"}),
        )
        .await
        .unwrap();
    let resolved = eng.resolve_iri("up:t").await.unwrap().unwrap();
    assert_eq!(resolved, id1);

    let id2 = eng
        .upsert(
            "up:t",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({"label": "second"}),
        )
        .await
        .unwrap();
    assert_eq!(id2, id1, "upsert reuses the same ID");

    let vec = eng.get_vector("up:t").await.unwrap().unwrap();
    assert!(
        (vec.coords[0] - 0.0).abs() < 1e-10,
        "vector should be updated"
    );
    assert!((vec.coords[1] - 1.0).abs() < 1e-10);
}

#[tokio::test]
async fn test_checkpoint_then_crash_with_filters() {
    let dir = tempfile::tempdir().unwrap();

    {
        let eng = setup(dir.path());
        eng.insert(
            "cr:0",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({"@type": ["X"], "val": 10}),
        )
        .await
        .unwrap();
        eng.insert(
            "cr:1",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({"@type": ["Y"], "val": 20}),
        )
        .await
        .unwrap();
        eng.checkpoint().await.unwrap();
    }

    {
        let eng = setup(dir.path());
        assert_eq!(eng.count().await.unwrap(), 2);

        let resolved = eng.resolve_iri("cr:0").await.unwrap();
        assert!(resolved.is_some(), "IRI should survive crash recovery");

        use hyperspace_engine::filter::JsonLdFilter;
        let y_filter = JsonLdFilter::Type("Y".to_string());
        let results = eng
            .search(&v(vec![0.0, 1.0, 0.0, 0.0]), 5, &[y_filter])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }
}

#[tokio::test]
async fn test_vacuum_with_multiple_ops() {
    let dir = tempfile::tempdir().unwrap();
    let eng = setup(dir.path());

    for i in 0..10u32 {
        let iri = format!("v:{}", i);
        eng.insert(
            &iri,
            v(vec![1.0 - i as f64 * 0.1, 0.0, 0.0, 0.0]),
            json!({"idx": i}),
        )
        .await
        .unwrap();
    }
    assert_eq!(eng.count().await.unwrap(), 10);

    eng.delete("v:0").await.unwrap();
    eng.delete("v:5").await.unwrap();
    eng.delete("v:9").await.unwrap();
    assert_eq!(eng.count().await.unwrap(), 7);

    eng.vacuum().await.unwrap();

    assert_eq!(eng.count().await.unwrap(), 7);
    let results = eng
        .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 10, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 7);
    for hit in &results {
        assert_ne!(hit.id, 1, "id 1 (v:0) was deleted");
        assert_ne!(hit.id, 6, "id 6 (v:5) was deleted");
        assert_ne!(hit.id, 10, "id 10 (v:9) was deleted");
    }
}

//! Reproducible release-mode probe for the local performance targets listed in README.md.
//!
//! Run with:
//!   cargo run --release --example readme_performance

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use glidinghorse::memory::l0_store::{L0Entry, L0Store, MesiState};
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::CoreConfig;
use hyperspace_engine::{CosineMetric, EmbeddingVector, HnswConfig, IncrementalHNSW, MetricKind};

fn average(elapsed: Duration, operations: usize) -> Duration {
    Duration::from_secs_f64(elapsed.as_secs_f64() / operations as f64)
}

fn status(actual: Duration, target: Duration) -> &'static str {
    if actual <= target {
        "PASS"
    } else {
        "FAIL"
    }
}

fn deterministic_vector(index: usize, dimensions: usize) -> EmbeddingVector {
    let coordinates = (0..dimensions)
        .map(|dimension| {
            let value = ((index * 31 + dimension * 17 + 1) % 997) as f64 / 997.0;
            value - 0.5
        })
        .collect::<Vec<_>>();
    EmbeddingVector::new_unchecked(coordinates, MetricKind::Cosine)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CoreConfig::default();
    let temp_root = std::env::temp_dir().join(format!(
        "glidinghorse-readme-performance-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&temp_root)?;

    // L0 redb read: one warm record, repeated point lookup.
    let l0 = L0Store::new(temp_root.join("l0").to_string_lossy().as_ref())?;
    let now = Utc::now();
    l0.store_entry(&L0Entry {
        iri: "iri://performance/l0".to_string(),
        content: "warm read payload".to_string(),
        importance: 0.5,
        access_count: 0,
        created_at: now,
        last_accessed: now,
        tags: vec!["performance".to_string()],
        metadata: serde_json::Map::new(),
        mesi_state: MesiState::Shared,
        content_hash: String::new(),
        named_graph: None,
        jsonld_context: None,
        jsonld_types: Vec::new(),
    })?;
    let l0_iterations = 10_000;
    let started = Instant::now();
    for _ in 0..l0_iterations {
        black_box(l0.retrieve("iri://performance/l0")?).ok_or("missing L0 probe")?;
    }
    let l0_average = average(started.elapsed(), l0_iterations);

    // L2 write, including the deferred Oxigraph synchronization barrier. The
    // public write_node call alone measures queueing rather than an Oxigraph write.
    let l2 = Blackboard::new()?;
    let l2_iterations = 100;
    let started = Instant::now();
    for index in 0..l2_iterations {
        let iri = format!("iri://task/performance-l2/node/{index}");
        let node = serde_json::json!({
            "@id": iri,
            "@type": "PerformanceNode",
            "summary": format!("node {index}")
        });
        l2.write_node(&iri, &node.to_string(), &config)?;
    }
    l2.flush_oxigraph();
    let l2_average = average(started.elapsed(), l2_iterations);

    // Cold L3 projections: unique task keys prevent materialized-cache hits.
    let projection_bb = Arc::new(Blackboard::new()?);
    let projection = ProjectionEngine::new(projection_bb.clone(), 5_000);
    let projection_iterations = 12;
    for task in 0..projection_iterations {
        for node_index in 0..20 {
            let iri = format!("iri://task/performance-projection-{task}/node/{node_index}");
            let node = serde_json::json!({
                "@id": iri,
                "@type": "TestNode",
                "summary": format!("projection node {node_index}"),
                "data": "x".repeat(50)
            });
            projection_bb.write_node(&iri, &node.to_string(), &config)?;
        }
    }
    projection_bb.flush_oxigraph();
    let started = Instant::now();
    for task in 0..projection_iterations {
        let task_iri = format!("iri://task/performance-projection-{task}");
        black_box(
            projection
                .project(&task_iri, "summary_only", HashMap::new())
                .await?,
        );
    }
    let projection_average = average(started.elapsed(), projection_iterations);

    // HNSW search over exactly 10K vectors. Index construction is intentionally
    // outside the search timer because README specifies search latency.
    let mut hnsw = IncrementalHNSW::new(Box::new(CosineMetric), HnswConfig::default());
    for index in 0..10_000 {
        hnsw.insert(index as u32, deterministic_vector(index, 32));
    }
    let query = deterministic_vector(10_001, 32);
    let hnsw_iterations = 1_000;
    let started = Instant::now();
    for _ in 0..hnsw_iterations {
        black_box(hnsw.search(&query, 10));
    }
    let hnsw_average = average(started.elapsed(), hnsw_iterations);

    // README calls this “Poincaré Embedding”; the concrete operation measured
    // here is creation and alpha precomputation of one validated 4D vector.
    let poincare_iterations = 100_000;
    let started = Instant::now();
    for index in 0..poincare_iterations {
        let delta = (index % 10) as f64 * 0.0001;
        black_box(EmbeddingVector::new(
            vec![0.1 + delta, 0.2, 0.05, 0.01],
            MetricKind::Poincare,
        )?);
    }
    let poincare_average = average(started.elapsed(), poincare_iterations);

    println!("README performance targets (release mode)");
    println!("operation\tactual\ttarget\tstatus");
    println!(
        "L2 durable node write\t{:?}\t2ms\t{}",
        l2_average,
        status(l2_average, Duration::from_millis(2))
    );
    println!(
        "L3 cold projection\t{:?}\t15ms\t{}",
        projection_average,
        status(projection_average, Duration::from_millis(15))
    );
    println!(
        "L0 redb KV read\t{:?}\t1ms\t{}",
        l0_average,
        status(l0_average, Duration::from_millis(1))
    );
    println!(
        "HNSW search (10K)\t{:?}\t1ms\t{}",
        hnsw_average,
        status(hnsw_average, Duration::from_millis(1))
    );
    println!(
        "Poincare 4D vector\t{:?}\t50us\t{}",
        poincare_average,
        status(poincare_average, Duration::from_micros(50))
    );

    drop(l0);
    std::fs::remove_dir_all(&temp_root)?;
    Ok(())
}

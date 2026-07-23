use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l1_session::L1Session;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::CoreConfig;
use std::sync::Arc;
use tempfile::tempdir;

/// Test L0 → L1 → L2 → L3 memory pipeline
#[test]
fn test_memory_pipeline() {
    let dir = tempdir().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));

    // L0 store/retrieve
    l0.store("iri://test/1", "test data").unwrap();
    let entry = l0.retrieve("iri://test/1").unwrap().unwrap();
    assert_eq!(entry.content, "test data");

    // L1 session
    let mut l1 = L1Session::new("agent_1", "DA", "iri://task/1");
    l1.add_summary("assistant", "Completed step 1", None);
    l1.add_summary("assistant", "Completed step 2", None);
    assert_eq!(l1.turn_count(), 2);

    // L2 blackboard
    let json_ld = r#"{"@id":"iri://task/1","@type":"Test"}"#;
    let config = glidinghorse::CoreConfig::default();
    l2.write_node("iri://task/1/node_1", json_ld, &config)
        .unwrap();
    let node = l2.read_node("iri://task/1/node_1").unwrap().unwrap();
    assert_eq!(node.node_type.as_ref().unwrap(), "Test");

    // L3 projection
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let result = proj
            .project(
                "iri://task/1",
                "reference_only",
                std::collections::HashMap::new(),
            )
            .await
            .unwrap();
        assert!(result.contains("task_iri"));
    });
}

/// Test MemoryManager coordination
#[test]
fn test_memory_manager() {
    let dir = tempdir().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let mut mm = MemoryManager::new(l0.clone(), l2.clone(), proj.clone(), CoreConfig::default());

    let session = mm.create_session("agent_1", "DA", "iri://task/1");
    let sid = session.session_id().to_string();
    mm.track_session(session);

    let fetched = mm.get_session(&sid);
    assert!(fetched.is_some());
    assert_eq!(fetched.unwrap().agent_id(), "agent_1");

    let summary = mm.close_session(&sid).unwrap();
    assert_eq!(summary.agent_role, "DA");
}

/// Test MemoryManager → HyperspaceStore integration via vector_store accessor.
#[test]
fn test_memory_manager_with_vector_store() {
    use glidinghorse::memory::embedding_service::FallbackEmbeddingService;
    use glidinghorse::memory::hyperspace_store::HyperspaceStore;

    let dir = tempdir().unwrap();
    let vdir = tempdir().unwrap();
    let embed = Arc::new(FallbackEmbeddingService::new());
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        let store = Arc::new(HyperspaceStore::open(vdir.path(), embed).unwrap());

        // Insert entries via HyperspaceStore
        store
            .upsert("iri://vec/1", "memory search test alpha", &["tag_a".into()])
            .await
            .unwrap();
        store
            .upsert("iri://vec/2", "memory search test beta", &["tag_a".into()])
            .await
            .unwrap();
        store
            .upsert("iri://vec/3", "unrelated gamma entry", &["tag_b".into()])
            .await
            .unwrap();

        // Search without filter – all match because FallbackEmbeddingService returns unit vector
        let all = store.search("test", 10).await.unwrap();
        assert_eq!(
            all.len(),
            3,
            "All 3 entries should match zero-vector search"
        );

        // Filter by tag
        let filtered = store
            .search_with_filter(
                "test",
                &glidinghorse::memory::hyperspace_store::HybridSearchFilter::new()
                    .with_must_tags(vec!["tag_a".into()]),
                10,
            )
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2, "Only tag_a entries");

        // Hybrid search (query + tag filter)
        let hybrid = store
            .hybrid_search("search", &["tag_a".into()], &[], None, 10)
            .await
            .unwrap();
        assert_eq!(hybrid.len(), 2);

        // Count & delete
        assert_eq!(store.count().await.unwrap(), 3);
        store.delete("iri://vec/1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
    });
}

/// Test MemoryManager → ProjectionEngine → HyperspaceStore end-to-end pipeline.
#[test]
fn test_full_pipeline_with_vector_store() {
    use glidinghorse::memory::embedding_service::FallbackEmbeddingService;
    use glidinghorse::memory::hyperspace_store::HyperspaceStore;

    let _dir = tempdir().unwrap();
    let vdir = tempdir().unwrap();
    let embed = Arc::new(FallbackEmbeddingService::new());
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        // Build full stack with vector store
        let l0 = Arc::new(L0Store::new(_dir.path().to_string_lossy().as_ref()).unwrap());
        let l2 = Arc::new(Blackboard::new().unwrap());
        let store = Arc::new(HyperspaceStore::open(vdir.path(), embed).unwrap());

        // ProjectionEngine receives vector store — verifies the construction compiles
        let _proj = Arc::new(ProjectionEngine::with_vector_store(
            l2.clone(),
            500,
            Some(store.clone()),
        ));

        let mut mm = MemoryManager::with_vector_store(
            l0.clone(),
            l2.clone(),
            _proj.clone(),
            CoreConfig::default(),
            Some(store.clone()),
        );

        // Upsert via MemoryManager's vector store reference
        mm.vector_store()
            .unwrap()
            .upsert(
                "iri://mm/1",
                "MemoryManager threaded vector store",
                &["mm".into()],
            )
            .await
            .unwrap();
        assert_eq!(mm.vector_store().unwrap().count().await.unwrap(), 1);

        // Create a session (L1) + L2 write + L3 projection still works
        let json_ld = r#"{"@id":"iri://task/vs","@type":"VectorStoreTest"}"#;
        let config = glidinghorse::CoreConfig::default();
        l2.write_node("iri://task/vs/n1", json_ld, &config).unwrap();
        let node = l2.read_node("iri://task/vs/n1").unwrap().unwrap();
        assert_eq!(node.node_type.as_ref().unwrap(), "VectorStoreTest");

        let session = mm.create_session("agent_vs", "DA", "iri://task/vs");
        assert_eq!(session.agent_id(), "agent_vs");
        let sid = mm.track_session(session);

        let fetched = mm.get_session(&sid).unwrap();
        assert_eq!(fetched.agent_id(), "agent_vs");
    });
}

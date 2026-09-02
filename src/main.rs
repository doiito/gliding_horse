use glidinghorse::api::grpc::server::seapp::se_kernel_service_server::SeKernelServiceServer;
use glidinghorse::api::grpc::server::AgentOSService;
use glidinghorse::config::settings::Settings;
use glidinghorse::utils::init_logging;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::load().unwrap_or_else(|e| {
        eprintln!("Warning: Failed to load config ({}), using defaults", e);
        Settings::default()
    });

    let _logging_guard = init_logging(&settings.logging);

    if let Err(e) = settings.validate() {
        eprintln!("Configuration error: {}", e);
        eprintln!("Please set AGENT_OS_GATEWAY_API_KEY or configure config.yaml");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&settings.output.directory)?;
    std::fs::create_dir_all(&settings.memory.l0.path)?;

    let addr: std::net::SocketAddr = settings.api.grpc_addr.parse()?;
    let http_addr: std::net::SocketAddr = settings.api.http_addr.parse()?;
    let auth_token = settings.api.auth_token.clone();
    let agent_os_service =
        AgentOSService::new(settings).map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    // async initialize BatchAgent system (register agents, start triggers)
    agent_os_service.init_batch_system().await;

    // mount existing axum HTTP/SSE routes (build_router) alongside gRPC, sharing runtime state
    let http_router = agent_os_service.build_http_router();
    tokio::spawn(async move {
        match tokio::net::TcpListener::bind(http_addr).await {
            Ok(listener) => {
                tracing::info!("Agent OS HTTP/SSE server starting on {}", http_addr);
                if let Err(e) = axum::serve(listener, http_router).await {
                    tracing::error!("HTTP server error: {}", e);
                }
            }
            Err(e) => tracing::error!("Failed to bind HTTP server on {}: {}", http_addr, e),
        }
    });

    tracing::info!("Agent OS gRPC server starting on {}", addr);

    if let Some(expected_token) = auth_token.filter(|token| !token.trim().is_empty()) {
        let interceptor = move |request: tonic::Request<()>| {
            let header = request
                .metadata()
                .get("authorization")
                .and_then(|value| value.to_str().ok());
            if glidinghorse::api::auth::valid_bearer_header(header, &expected_token) {
                Ok(request)
            } else {
                Err(tonic::Status::unauthenticated(
                    "missing or invalid bearer token",
                ))
            }
        };
        tonic::transport::Server::builder()
            .add_service(SeKernelServiceServer::with_interceptor(
                agent_os_service,
                interceptor,
            ))
            .serve(addr)
            .await?;
    } else {
        tonic::transport::Server::builder()
            .add_service(SeKernelServiceServer::new(agent_os_service))
            .serve(addr)
            .await?;
    }

    Ok(())
}

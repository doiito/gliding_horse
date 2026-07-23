use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustyline::error::ReadlineError;
use rustyline::{Config, DefaultEditor, EditMode};
use tokio::sync::broadcast;

use crate::config::CliConfig;
use crate::engine::CodeCliEngine;
use crate::render::{self, StreamRenderer};

pub async fn run_repl(mut config: CliConfig) -> anyhow::Result<()> {
    render::banner();

    let rl_config = Config::builder().edit_mode(EditMode::Emacs).build();
    let mut rl = DefaultEditor::with_config(rl_config)?;

    let mut engine = CodeCliEngine::new(config.clone())?;

    loop {
        render::prompt();
        let readline = rl.readline("");
        match readline {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(&line);

                if line.starts_with('/') {
                    handle_slash_command(&line, &mut config, &mut engine)?;
                } else {
                    let input = collect_input(&mut rl, line)?;
                    render::user_input(&input);

                    let mut renderer = StreamRenderer::new();
                    renderer.show_task_start(&engine.workspace());

                    let mut receiver = engine.subscribe();
                    let done = Arc::new(AtomicBool::new(false));
                    let done_clone = done.clone();

                    let event_handle = tokio::spawn(async move {
                        while !done_clone.load(Ordering::SeqCst) {
                            match receiver.try_recv() {
                                Ok(event) => {
                                    renderer.handle_event(&event);
                                }
                                Err(broadcast::error::TryRecvError::Empty) => {
                                    tokio::time::sleep(tokio::time::Duration::from_millis(50))
                                        .await;
                                }
                                Err(broadcast::error::TryRecvError::Closed) => break,
                                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                            }
                        }
                        while let Ok(event) = receiver.try_recv() {
                            renderer.handle_event(&event);
                        }
                        renderer
                    });

                    let result = engine.process_task(&input).await;
                    done.store(true, Ordering::SeqCst);

                    let mut renderer = event_handle.await?;

                    match result {
                        Ok((_task_iri, task_result)) => {
                            renderer.show_task_result(
                                &task_result.status,
                                &task_result.summary,
                                task_result.turn_count,
                                task_result.tool_call_count,
                                &engine.workspace(),
                            );
                        }
                        Err(e) => {
                            render::error(&format!("任务执行失败: {}", e));
                        }
                    }
                    renderer.finish();
                }
            }
            Err(ReadlineError::Interrupted) => {
                render::info("按 Ctrl+D 或输入 /exit 退出");
            }
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                render::error(&format!("输入错误: {}", err));
                break;
            }
        }
    }

    render::info("再见！");
    Ok(())
}

fn collect_input(rl: &mut DefaultEditor, first_line: String) -> anyhow::Result<String> {
    let mut input = first_line;
    while input.ends_with('\\') {
        input.pop();
        let continuation = rl.readline("  > ")?;
        input.push('\n');
        input.push_str(&continuation);
    }
    Ok(input)
}

fn handle_slash_command(
    line: &str,
    config: &mut CliConfig,
    engine: &mut CodeCliEngine,
) -> anyhow::Result<bool> {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0];

    match cmd {
        "/exit" | "/quit" => {
            render::info("再见！");
            std::process::exit(0);
        }
        "/help" => {
            render::help_message();
            Ok(true)
        }
        "/model" => {
            if parts.len() > 1 {
                let new_model = parts[1].trim().to_string();
                let supported = [
                    "deepseek-v4-flash",
                    "deepseek-v4-pro",
                    "deepseek-chat",
                    "deepseek-reasoner",
                ];
                if supported.contains(&new_model.as_str()) {
                    config.model = new_model.clone();
                    engine.rebuild_with_model(new_model.clone())?;
                    render::success(&format!("模型已切换为: {}", new_model));
                } else {
                    render::error(&format!(
                        "不支持的模型: {}. 支持: {:?}",
                        new_model, supported
                    ));
                }
            } else {
                render::info(&format!("当前模型: {}", engine.model()));
                render::info("用法: /model <deepseek-v4-flash|deepseek-v4-pro>");
            }
            Ok(true)
        }
        "/clear" => {
            engine.rebuild()?;
            render::success("对话历史已清空");
            Ok(true)
        }
        "/stats" => {
            render::info(&format!("当前模型: {}", engine.model()));
            render::info(&format!("工作目录: {}", engine.workspace()));
            render::info(&format!("最大迭代次数: {}", engine.max_iterations()));
            Ok(true)
        }
        _ => {
            render::error(&format!("未知命令: {}", cmd));
            render::info("输入 /help 查看可用命令");
            Ok(true)
        }
    }
}

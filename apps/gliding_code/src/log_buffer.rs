use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A shared buffer that captures formatted tracing output line-by-line.
///
/// Lines are appended to a bounded in-memory queue. When `mirror_to_stderr`
/// is enabled (set for single-shot / one-shot mode), every line is ALSO
/// written to stderr in real time so a running task shows live progress —
/// rather than withholding it until [`LogBuffer::drain`] is called at the
/// very end of the task. This was the root cause of "glidingcode looks stuck
/// / generates nothing" for long-running tasks: with buffering-only, an
/// agent that is actively downloading assets or writing thousands of files
/// appears completely still because zero bytes ever reach the terminal while
/// it works.
pub struct LogBuffer {
    buffer: Arc<Mutex<VecDeque<String>>>,
    mirror_to_stderr: Arc<AtomicBool>,
}

impl LogBuffer {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            mirror_to_stderr: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Enable/disable real-time mirroring of every log line to stderr.
    /// When enabled, log lines are visible in the terminal as the task
    /// progresses instead of only after the task completes.
    pub fn set_mirror_to_stderr(&self, enabled: bool) {
        self.mirror_to_stderr.store(enabled, Ordering::SeqCst);
    }

    /// True if live mirroring to stderr is currently enabled.
    pub fn mirrors_to_stderr(&self) -> bool {
        self.mirror_to_stderr.load(Ordering::SeqCst)
    }

    pub fn drain(&self) -> Vec<String> {
        let mut buf = self.buffer.lock().expect("LogBuffer Mutex poisoned");
        buf.drain(..).collect()
    }
}

/// Internal helper: flush a line to stderr faithfully (line-buffered).
fn write_line_to_stderr(line: &str) {
    use std::io::Write as _;
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(line.as_bytes());
    let _ = err.write_all(b"\n");
    let _ = err.flush();
}

/// A wrapper around `Arc<LogBuffer>` that implements `MakeWriter`.
/// This is needed because Rust's orphan rule prevents implementing
/// a foreign trait (MakeWriter) for a foreign type (Arc).
pub struct SharedLogBuffer(pub Arc<LogBuffer>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedLogBuffer {
    type Writer = LogBufferWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogBufferWriter {
            buffer: self.0.buffer.clone(),
            mirror_to_stderr: self.0.mirror_to_stderr.clone(),
        }
    }
}

pub struct LogBufferWriter {
    buffer: Arc<Mutex<VecDeque<String>>>,
    mirror_to_stderr: Arc<AtomicBool>,
}

impl Write for LogBufferWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let s = String::from_utf8_lossy(buf).to_string();
        let mirror = self.mirror_to_stderr.load(Ordering::SeqCst);
        let mut lines = Vec::with_capacity(8);
        let mut buffer = self.buffer.lock().expect("LogBuffer Mutex poisoned");
        for line in s.split_inclusive('\n') {
            let trimmed = line.trim_end_matches('\n');
            if !trimmed.is_empty() {
                buffer.push_back(trimmed.to_string());
                if mirror {
                    lines.push(trimmed.to_string());
                }
            }
        }
        while buffer.len() > 2000 {
            buffer.pop_front();
        }
        drop(buffer);
        if mirror {
            // Write mirrored lines outside the buffer lock to avoid holding
            // it during slow terminal IO.
            for l in &lines {
                write_line_to_stderr(l);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

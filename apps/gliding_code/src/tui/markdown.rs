use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Replace bare fences with a plain-text language so the renderer does not
/// attempt syntax detection for an empty language name.
fn default_code_lang(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut in_fence = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_fence && trimmed.starts_with("```") {
            let rest = trimmed[3..].trim();
            if rest.is_empty() {
                out.push_str("```txt\n");
            } else {
                out.push_str(line);
                out.push('\n');
            }
            in_fence = !trimmed.ends_with("```") || rest.is_empty();
        } else if in_fence && trimmed == "```" {
            out.push_str(line);
            out.push('\n');
            in_fence = false;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

pub(super) fn markdown_to_owned_lines(content: &str) -> Vec<Line<'static>> {
    let prepared = default_code_lang(content);
    let text = tui_markdown::from_str(&prepared);
    text.lines
        .into_iter()
        .map(|line| {
            Line::from(
                line.spans
                    .into_iter()
                    .map(|span| {
                        Span::styled(
                            span.content.into_owned(),
                            convert_markdown_style(span.style),
                        )
                    })
                    .collect::<Vec<Span<'static>>>(),
            )
        })
        .collect()
}

fn convert_markdown_color(color: ratatui_core::style::Color) -> Color {
    use ratatui_core::style::Color as MarkdownColor;

    match color {
        MarkdownColor::Reset => Color::Reset,
        MarkdownColor::Black => Color::Black,
        MarkdownColor::Red => Color::Red,
        MarkdownColor::Green => Color::Green,
        MarkdownColor::Yellow => Color::Yellow,
        MarkdownColor::Blue => Color::Blue,
        MarkdownColor::Magenta => Color::Magenta,
        MarkdownColor::Cyan => Color::Cyan,
        MarkdownColor::Gray => Color::Gray,
        MarkdownColor::DarkGray => Color::DarkGray,
        MarkdownColor::LightRed => Color::LightRed,
        MarkdownColor::LightGreen => Color::LightGreen,
        MarkdownColor::LightYellow => Color::LightYellow,
        MarkdownColor::LightBlue => Color::LightBlue,
        MarkdownColor::LightMagenta => Color::LightMagenta,
        MarkdownColor::LightCyan => Color::LightCyan,
        MarkdownColor::White => Color::White,
        MarkdownColor::Rgb(red, green, blue) => Color::Rgb(red, green, blue),
        MarkdownColor::Indexed(index) => Color::Indexed(index),
    }
}

fn convert_markdown_modifiers(source: ratatui_core::style::Modifier) -> Modifier {
    use ratatui_core::style::Modifier as MarkdownModifier;

    [
        (MarkdownModifier::BOLD, Modifier::BOLD),
        (MarkdownModifier::DIM, Modifier::DIM),
        (MarkdownModifier::ITALIC, Modifier::ITALIC),
        (MarkdownModifier::UNDERLINED, Modifier::UNDERLINED),
        (MarkdownModifier::SLOW_BLINK, Modifier::SLOW_BLINK),
        (MarkdownModifier::RAPID_BLINK, Modifier::RAPID_BLINK),
        (MarkdownModifier::REVERSED, Modifier::REVERSED),
        (MarkdownModifier::HIDDEN, Modifier::HIDDEN),
        (MarkdownModifier::CROSSED_OUT, Modifier::CROSSED_OUT),
    ]
    .into_iter()
    .filter_map(|(from, to)| source.contains(from).then_some(to))
    .fold(Modifier::empty(), |combined, modifier| combined | modifier)
}

pub(super) fn convert_markdown_style(source: ratatui_core::style::Style) -> Style {
    Style {
        fg: source.fg.map(convert_markdown_color),
        bg: source.bg.map(convert_markdown_color),
        add_modifier: convert_markdown_modifiers(source.add_modifier),
        sub_modifier: convert_markdown_modifiers(source.sub_modifier),
    }
}

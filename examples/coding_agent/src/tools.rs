use autoagents::llm::{ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{tool, ToolInput};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct FileSearchArgs {
    #[input(description = "Glob pattern to search for files (e.g., '*.rs', 'src/**/*.txt')")]
    pattern: String,
    #[input(description = "Base directory to search in")]
    base_dir: String,
}

#[tool(
    name = "FileSearchTool",
    description = "Search for files matching a glob pattern in the project",
    input = FileSearchArgs,
)]
pub fn search_files(args: FileSearchArgs) -> Result<String, ToolCallError> {
    println!("🔍 Searching for files matching: {}", args.pattern);

    let base_path = Path::new(&args.base_dir);
    if !base_path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("Directory {} does not exist", args.base_dir).into(),
        ));
    }

    let mut matches = Vec::new();
    let pattern = glob::Pattern::new(&args.pattern)
        .map_err(|e| ToolCallError::RuntimeError(format!("Invalid pattern: {}", e).into()))?;

    for entry in WalkDir::new(&args.base_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(&args.base_dir).unwrap_or(path);
            if pattern.matches_path(relative_path) {
                matches.push(relative_path.display().to_string());
            }
        }
    }

    if matches.is_empty() {
        Ok("No files found matching the pattern.".to_string())
    } else {
        Ok(format!(
            "Found {} files:\n{}",
            matches.len(),
            matches.join("\n")
        ))
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct GrepArgs {
    #[input(description = "Regular expression pattern to search for")]
    pattern: String,
    #[input(description = "File glob pattern to search in (e.g., '*.rs')")]
    file_pattern: String,
    #[input(description = "Base directory to search in")]
    base_dir: String,
}

#[tool(
    name = "GrepTool",
    description = "Search for content in files using regex patterns",
    input = GrepArgs,
)]
pub fn grep_files(args: GrepArgs) -> Result<String, ToolCallError> {
    println!("🔎 Grepping for: {} in {}", args.pattern, args.file_pattern);

    let regex = Regex::new(&args.pattern)
        .map_err(|e| ToolCallError::RuntimeError(format!("Invalid regex: {}", e).into()))?;

    let base_path = Path::new(&args.base_dir);
    if !base_path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("Directory {} does not exist", args.base_dir).into(),
        ));
    }

    let file_pattern = glob::Pattern::new(&args.file_pattern)
        .map_err(|e| ToolCallError::RuntimeError(format!("Invalid file pattern: {}", e).into()))?;

    let mut results = Vec::new();
    let max_results = 50;

    for entry in WalkDir::new(&args.base_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if results.len() >= max_results {
            break;
        }

        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(&args.base_dir).unwrap_or(path);
            if file_pattern.matches_path(relative_path) {
                if let Ok(content) = fs::read_to_string(path) {
                    for (line_num, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            results.push(format!(
                                "{}:{}: {}",
                                relative_path.display(),
                                line_num + 1,
                                line.trim()
                            ));
                            if results.len() >= max_results {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if results.is_empty() {
        Ok("No matches found.".to_string())
    } else {
        Ok(format!(
            "Found {} matches (showing up to {}):\n{}",
            results.len(),
            max_results,
            results.join("\n")
        ))
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ReadFileArgs {
    #[input(description = "Path to the file to read")]
    file_path: String,
}

#[tool(
    name = "ReadFileTool",
    description = "Read contents of a file, optionally specifying line range",
    input = ReadFileArgs,
)]
pub fn read_file(args: ReadFileArgs) -> Result<String, ToolCallError> {
    println!("📖 Reading file: {}", args.file_path);

    let path = Path::new(&args.file_path);
    if !path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("File {} does not exist", args.file_path).into(),
        ));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| ToolCallError::RuntimeError(format!("Failed to read file: {}", e).into()))?;

    Ok(format!("File: {}\n{}", args.file_path, content))
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WriteFileArgs {
    #[input(description = "Path to the file to write")]
    file_path: String,
    #[input(description = "Content to write to the file")]
    content: String,
}

#[tool(
    name = "WriteFileTool",
    description = "Write content to a file (create or overwrite)",
    input = WriteFileArgs,
)]
pub fn write_file(args: WriteFileArgs) -> Result<String, ToolCallError> {
    println!("✍️  Writing to file: {}", args.file_path);

    let path = Path::new(&args.file_path);

    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            ToolCallError::RuntimeError(format!("Failed to create directories: {}", e).into())
        })?;
    }

    fs::write(path, &args.content)
        .map_err(|e| ToolCallError::RuntimeError(format!("Failed to write file: {}", e).into()))?;

    Ok(format!(
        "Successfully wrote file: {} ({} bytes)",
        args.file_path,
        args.content.len()
    ))
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct DeleteFileArgs {
    #[input(description = "Path to the file to delete")]
    file_path: String,
}

#[tool(
    name = "DeleteFileTool",
    description = "Delete a file from the filesystem",
    input = DeleteFileArgs,
)]
pub fn delete_file(args: DeleteFileArgs) -> Result<String, ToolCallError> {
    println!("🗑️  Deleting file: {}", args.file_path);

    let path = Path::new(&args.file_path);
    if !path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("File {} does not exist", args.file_path).into(),
        ));
    }

    fs::remove_file(path)
        .map_err(|e| ToolCallError::RuntimeError(format!("Failed to delete file: {}", e).into()))?;

    Ok(format!("Successfully deleted file: {}", args.file_path))
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ListDirectoryArgs {
    #[input(description = "Path to the directory to list")]
    dir_path: String,
}

#[tool(
    name = "ListDirectoryTool",
    description = "List contents of a directory",
    input = ListDirectoryArgs,
)]
pub fn list_directory(args: ListDirectoryArgs) -> Result<String, ToolCallError> {
    println!("📁 Listing directory: {}", args.dir_path);

    let path = Path::new(&args.dir_path);
    if !path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("Directory {} does not exist", args.dir_path).into(),
        ));
    }

    if !path.is_dir() {
        return Err(ToolCallError::RuntimeError(
            format!("{} is not a directory", args.dir_path).into(),
        ));
    }

    let mut entries = Vec::new();

    for entry in fs::read_dir(path).map_err(|e| {
        ToolCallError::RuntimeError(format!("Failed to read directory: {}", e).into())
    })? {
        if let Ok(entry) = entry {
            let file_name = entry.file_name().to_string_lossy().to_string();

            if file_name.starts_with('.') {
                continue;
            }

            let metadata = entry.metadata().ok();
            let file_type = if metadata.as_ref().map_or(false, |m| m.is_dir()) {
                "[DIR]"
            } else {
                "[FILE]"
            };

            entries.push(format!("{} {}", file_type, file_name));
        }
    }

    entries.sort();

    if entries.is_empty() {
        Ok("Directory is empty.".to_string())
    } else {
        Ok(format!(
            "Directory {} contains {} items:\n{}",
            args.dir_path,
            entries.len(),
            entries.join("\n")
        ))
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AnalyzeCodeArgs {
    #[input(description = "Path to the file or directory to analyze")]
    path: String,
    #[input(description = "Type of analysis: 'structure', 'complexity', 'dependencies'")]
    analysis_type: String,
}

#[tool(
    name = "AnalyzeCodeTool",
    description = "Analyze code structure, complexity, or dependencies",
    input = AnalyzeCodeArgs,
)]
pub fn analyze_code(args: AnalyzeCodeArgs) -> Result<String, ToolCallError> {
    println!("🔬 Analyzing code: {} ({})", args.path, args.analysis_type);

    let path = Path::new(&args.path);
    if !path.exists() {
        return Err(ToolCallError::RuntimeError(
            format!("Path {} does not exist", args.path).into(),
        ));
    }

    match args.analysis_type.as_str() {
        "structure" => analyze_structure(path),
        "complexity" => analyze_complexity(path),
        "dependencies" => analyze_dependencies(path),
        _ => Err(ToolCallError::RuntimeError(
            "Invalid analysis type. Choose 'structure', 'complexity', or 'dependencies'".into(),
        )),
    }
}

fn analyze_structure(path: &Path) -> Result<String, ToolCallError> {
    let mut file_count = 0;
    let mut dir_count = 0;
    let mut total_lines = 0;
    let mut extensions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    if path.is_file() {
        file_count = 1;
        if let Ok(content) = fs::read_to_string(path) {
            total_lines = content.lines().count();
        }
        if let Some(ext) = path.extension() {
            *extensions
                .entry(ext.to_string_lossy().to_string())
                .or_insert(0) += 1;
        }
    } else {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                file_count += 1;
                if let Ok(content) = fs::read_to_string(entry_path) {
                    total_lines += content.lines().count();
                }
                if let Some(ext) = entry_path.extension() {
                    *extensions
                        .entry(ext.to_string_lossy().to_string())
                        .or_insert(0) += 1;
                }
            } else if entry_path.is_dir() {
                dir_count += 1;
            }
        }
    }

    let mut ext_summary = String::new();
    for (ext, count) in extensions.iter() {
        ext_summary.push_str(&format!("\n  .{}: {} files", ext, count));
    }

    Ok(format!(
        "Code Structure Analysis:\n\
        - Files: {}\n\
        - Directories: {}\n\
        - Total lines: {}\n\
        - File types:{}",
        file_count, dir_count, total_lines, ext_summary
    ))
}

fn analyze_complexity(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified complexity analysis
    Ok(
        "Complexity analysis: This is a placeholder. In a real implementation, \
        this would calculate cyclomatic complexity, function lengths, and other metrics."
            .to_string(),
    )
}

fn analyze_dependencies(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified dependency analysis
    Ok(
        "Dependency analysis: This is a placeholder. In a real implementation, \
        this would parse import statements and analyze module dependencies."
            .to_string(),
    )
}

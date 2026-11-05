//! Builds the tests once, then runs the test executables directly, a few at a time.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

const HELP: &str = "\
Builds the tests once, then runs the test executables directly, a few at a time.

usage: cargo run -p xtask -- test [OPTIONS] [FILTER...] [-- TEST_ARGS...]

  FILTER         run only targets whose name contains it, such as lighting_suite.path,
                 orr_cook. or doc
  --skip FILTER  leave out targets whose name contains it; repeatable
  --fast         leave out the slow targets
  --jobs N       targets to run at once (default 3)
  --list         print the selected targets and stop
  TEST_ARGS      passed to every test executable

ORR_TEST_WRAPPER is a command, split on spaces, that every test executable runs under.
It is the place for a time limit: the runner sets none. Doc tests run without it.

Each target's output goes to target/test-logs/<target>.log. Exits 0 when every selected
target passes, 1 when one fails and 2 when the tests can't be built or run.
";

/// More GPU runs at once have coincided with device resets, and gain little: the driver's
/// pipeline compiles dominate and already use many threads.
const DEFAULT_JOBS: usize = 3;

/// Targets that take much longer than the rest, longest first. They start first, so the
/// run doesn't end on one of them, and `--fast` leaves them out.
const SLOW: &[&str] = &[
    "lighting_suite.linked",
    "orr_examples.screenshots",
    "lighting_suite.dielectric",
    "shader_link.gpu",
    "lighting_suite.media",
    "lighting_suite.subsurface",
    "lighting_suite.world_cache",
    "atmosphere_suite.rendered_medium",
    "lighting_suite.path",
    "lighting_suite.alpha",
    "lighting_suite.guides",
    "orr_atmosphere.reference",
    "scene_suite.renderer",
];

/// What the workspace pass and the doc tests build.
const WORKSPACE: &[&str] = &[
    "--workspace",
    "--features",
    "lighting_math/gpu,openpbr_ref/gpu",
];

/// One `cargo test --no-run`.
struct Pass {
    args: &'static [&'static str],
    /// The one target this pass is for. Without it the pass builds every selected target.
    only: Option<&'static str>,
}

/// Features the workspace build leaves off are turned on here. The workspace pass enables
/// the ones that only add tests; features that replace behavior get a pass of their own,
/// whose target takes the place of the workspace pass's build. Every pass selects the whole
/// workspace, so dependencies resolve to the same features and build once.
const PASSES: &[Pass] = &[
    Pass {
        args: WORKSPACE,
        only: None,
    },
    Pass {
        args: &[
            "--workspace",
            "--features",
            "lighting_math/gpu,openpbr_ref/gpu,orr_dlss/mock",
            "--test",
            "mock",
        ],
        only: Some("orr_dlss.mock"),
    },
];

/// The name of the doc tests' job. `cargo test` runs them, for the whole workspace.
const DOC: &str = "doc";

struct Options {
    filters: Vec<String>,
    skips: Vec<String>,
    fast: bool,
    jobs: usize,
    list: bool,
    test_args: Vec<String>,
}

impl Options {
    fn parse(args: Vec<String>) -> Result<Option<Self>, String> {
        let mut options = Self {
            filters: Vec::new(),
            skips: Vec::new(),
            fast: false,
            jobs: DEFAULT_JOBS,
            list: false,
            test_args: Vec::new(),
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(None),
                "--fast" => options.fast = true,
                "--list" => options.list = true,
                "--skip" => options
                    .skips
                    .push(args.next().ok_or("--skip needs a filter")?),
                "--jobs" => {
                    options.jobs = args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .filter(|&n| n > 0)
                        .ok_or("--jobs needs a positive number")?;
                }
                "--" => options.test_args.extend(args.by_ref()),
                flag if flag.starts_with('-') => return Err(format!("unknown option {flag}")),
                _ => options.filters.push(arg),
            }
        }
        Ok(Some(options))
    }

    fn selects(&self, name: &str) -> bool {
        (self.filters.is_empty() || self.filters.iter().any(|f| name.contains(f.as_str())))
            && !self.skips.iter().any(|f| name.contains(f.as_str()))
            && !(self.fast && SLOW.contains(&name))
    }

    fn narrowed(&self) -> bool {
        self.fast || !self.filters.is_empty() || !self.skips.is_empty()
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Lib,
    Bin,
    Test,
    Example,
    Bench,
    BuildScript,
}

/// A target `cargo test` runs.
struct Target {
    name: String,
    package: String,
    kind: Kind,
    /// The name cargo selects it by.
    cargo_name: String,
    dir: PathBuf,
    required_features: Vec<String>,
    exe: Option<PathBuf>,
}

struct Workspace {
    root: PathBuf,
    logs: PathBuf,
    /// Package names by package id.
    packages: BTreeMap<String, String>,
    /// Every feature each package declares.
    features: BTreeMap<String, BTreeSet<String>>,
    targets: Vec<Target>,
    /// Library, binary and test targets marked `test = false`.
    untested: Vec<String>,
    /// Examples and benches, which `cargo test` doesn't run.
    not_tests: usize,
}

impl Workspace {
    fn selected<'a>(&'a self, options: &'a Options) -> impl Iterator<Item = &'a Target> {
        self.targets
            .iter()
            .filter(move |t| options.selects(&t.name))
    }
}

fn kind_of(target: &Value) -> Kind {
    match target["kind"][0].as_str() {
        Some("test") => Kind::Test,
        Some("bin") => Kind::Bin,
        Some("example") => Kind::Example,
        Some("bench") => Kind::Bench,
        Some("custom-build") => Kind::BuildScript,
        _ => Kind::Lib,
    }
}

fn target_name(package: &str, target: &Value) -> String {
    let name = target["name"].as_str().unwrap_or_default();
    match kind_of(target) {
        Kind::Lib => format!("{package}.lib"),
        Kind::Bin => format!("{package}.bin.{name}"),
        Kind::Test => format!("{package}.{name}"),
        Kind::Example | Kind::Bench | Kind::BuildScript => format!("{package}.other.{name}"),
    }
}

fn strings(value: &Value) -> Vec<String> {
    let items = value.as_array().map_or(&[][..], Vec::as_slice);
    items
        .iter()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect()
}

fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn workspace() -> Result<Workspace, String> {
    let output = cargo()
        .args(["metadata", "--no-deps", "--locked", "--format-version=1"])
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo metadata: {}", output.status));
    }
    let meta: Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("cargo metadata: {e}"))?;
    let mut ws = Workspace {
        root: meta["workspace_root"].as_str().unwrap_or(".").into(),
        logs: Path::new(meta["target_directory"].as_str().unwrap_or("target")).join("test-logs"),
        packages: BTreeMap::new(),
        features: BTreeMap::new(),
        targets: Vec::new(),
        untested: Vec::new(),
        not_tests: 0,
    };
    for package in meta["packages"].as_array().into_iter().flatten() {
        let name = package["name"].as_str().unwrap_or_default();
        let id = package["id"].as_str().unwrap_or_default();
        ws.packages.insert(id.to_owned(), name.to_owned());
        let declared = package["features"].as_object().into_iter().flatten();
        ws.features.insert(
            name.to_owned(),
            declared
                .map(|(feature, _)| feature.clone())
                .filter(|feature| feature != "default")
                .collect(),
        );
        let manifest = Path::new(package["manifest_path"].as_str().unwrap_or_default());
        let dir = manifest.parent().unwrap_or(manifest);
        for target in package["targets"].as_array().into_iter().flatten() {
            let kind = kind_of(target);
            if target["test"].as_bool() == Some(true) {
                ws.targets.push(Target {
                    name: target_name(name, target),
                    package: name.to_owned(),
                    kind,
                    cargo_name: target["name"].as_str().unwrap_or_default().to_owned(),
                    dir: dir.to_owned(),
                    required_features: strings(&target["required-features"]),
                    exe: None,
                });
            } else {
                match kind {
                    Kind::Lib | Kind::Bin | Kind::Test => {
                        ws.untested.push(target_name(name, target));
                    }
                    Kind::Example | Kind::Bench => ws.not_tests += 1,
                    Kind::BuildScript => {}
                }
            }
        }
    }
    Ok(ws)
}

/// Cargo's target selection for the selected targets.
fn selection(targets: &[&Target]) -> Vec<String> {
    let mut args = BTreeSet::new();
    for target in targets {
        let flag = match target.kind {
            Kind::Lib => {
                args.insert(vec!["--lib".to_owned()]);
                continue;
            }
            Kind::Bin => "--bin",
            Kind::Test => "--test",
            Kind::Example => "--example",
            Kind::Bench => "--bench",
            Kind::BuildScript => continue,
        };
        args.insert(vec![flag.to_owned(), target.cargo_name.clone()]);
    }
    args.into_iter().flatten().collect()
}

/// Builds the selected targets, fills in their executables and returns the features each
/// package's tests built with.
fn build(
    ws: &mut Workspace,
    options: &Options,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let shared: Vec<&Target> = ws
        .selected(options)
        .filter(|t| PASSES.iter().all(|p| p.only != Some(t.name.as_str())))
        .collect();
    // With every target selected cargo chooses, so the build also covers what `cargo test`
    // builds without running, such as examples.
    let narrow = (ws.selected(options).count() < ws.targets.len()).then(|| selection(&shared));
    let mut exes = BTreeMap::new();
    let mut features: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for pass in PASSES {
        if pass
            .only
            .map_or(shared.is_empty(), |only| !options.selects(only))
        {
            continue;
        }
        let mut command = cargo();
        command
            .args(["test", "--no-run", "--locked", "--message-format=json"])
            .args(pass.args)
            .current_dir(&ws.root)
            .stdout(Stdio::piped());
        if pass.only.is_none()
            && let Some(narrow) = &narrow
        {
            command.args(narrow);
        }
        let mut child = command.spawn().map_err(|e| format!("cargo test: {e}"))?;
        let stdout = child.stdout.take().ok_or("cargo test: no stdout")?;
        for line in BufReader::new(stdout).lines() {
            let line = line.map_err(|e| format!("cargo test: {e}"))?;
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            match message["reason"].as_str() {
                Some("compiler-message") => {
                    eprint!("{}", message["message"]["rendered"].as_str().unwrap_or(""));
                }
                Some("compiler-artifact") if message["profile"]["test"] == true => {
                    let package = message["package_id"]
                        .as_str()
                        .and_then(|id| ws.packages.get(id));
                    let (Some(package), Some(exe)) = (package, message["executable"].as_str())
                    else {
                        continue;
                    };
                    let name = target_name(package, &message["target"]);
                    if pass.only.is_none_or(|only| only == name) {
                        exes.insert(name, PathBuf::from(exe));
                        features
                            .entry(package.clone())
                            .or_default()
                            .extend(strings(&message["features"]));
                    }
                }
                _ => {}
            }
        }
        let status = child.wait().map_err(|e| format!("cargo test: {e}"))?;
        if !status.success() {
            return Err(format!("the test build failed: {status}"));
        }
    }
    for target in &mut ws.targets {
        target.exe = exes.remove(&target.name);
    }
    Ok(features)
}

struct Job {
    name: String,
    command: Command,
}

/// Each executable runs as under `cargo test`: in its package's directory, with
/// `CARGO_MANIFEST_DIR` and `CARGO_PKG_NAME` set and the dependency directory on `PATH`.
fn test_job(target: &Target, exe: &Path, wrapper: &[String], test_args: &[String]) -> Job {
    let mut command = match wrapper {
        [program, args @ ..] => {
            let mut command = Command::new(program);
            command.args(args).arg(exe);
            command
        }
        [] => Command::new(exe),
    };
    let path = env::var_os("PATH").unwrap_or_default();
    let paths = exe
        .parent()
        .map(Path::to_owned)
        .into_iter()
        .chain(env::split_paths(&path));
    command
        .args(test_args)
        .current_dir(&target.dir)
        .env("CARGO_MANIFEST_DIR", &target.dir)
        .env("CARGO_PKG_NAME", &target.package)
        .env(
            "PATH",
            env::join_paths(paths).unwrap_or_else(|_| OsString::from(&path)),
        );
    Job {
        name: target.name.clone(),
        command,
    }
}

/// Doc tests go through cargo without the wrapper: rustdoc compiles them as they run, which
/// no `--no-run` build can do first, and none of them touches the GPU.
fn doc_job(ws: &Workspace, test_args: &[String]) -> Job {
    let mut command = cargo();
    command
        .args(["test", "--doc", "--locked", "--no-fail-fast"])
        .args(WORKSPACE)
        .current_dir(&ws.root);
    if !test_args.is_empty() {
        command.arg("--").args(test_args);
    }
    Job {
        name: DOC.to_owned(),
        command,
    }
}

/// Slow targets first, longest first, then the rest by name.
fn schedule(jobs: &mut [Job]) {
    let rank = |job: &Job| {
        SLOW.iter()
            .position(|name| *name == job.name)
            .unwrap_or(SLOW.len())
    };
    jobs.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.name.cmp(&b.name)));
}

#[derive(Default, PartialEq, Debug)]
struct LogSummary {
    passed: u64,
    failed: u64,
    ignored: u64,
    failed_tests: Vec<String>,
}

/// Adds up libtest's result lines. The doc job prints one per package.
fn read_log(text: &str) -> LogSummary {
    let mut summary = LogSummary::default();
    for line in text.lines() {
        if let Some(result) = line.strip_prefix("test result: ") {
            for part in result.split(['.', ';']) {
                let mut words = part.split_whitespace();
                let count = words.next().and_then(|n| n.parse::<u64>().ok());
                match (count, words.next()) {
                    (Some(n), Some("passed")) => summary.passed += n,
                    (Some(n), Some("failed")) => summary.failed += n,
                    (Some(n), Some("ignored")) => summary.ignored += n,
                    _ => {}
                }
            }
        } else if let Some(name) = line
            .strip_prefix("test ")
            .and_then(|l| l.strip_suffix(" ... FAILED"))
        {
            summary.failed_tests.push(name.to_owned());
        }
    }
    summary
}

struct Outcome {
    name: String,
    log: PathBuf,
    status: Result<ExitStatus, String>,
    time: Duration,
    summary: LogSummary,
}

impl Outcome {
    fn ok(&self) -> bool {
        self.status.as_ref().is_ok_and(ExitStatus::success)
    }

    fn describe_status(&self) -> String {
        match &self.status {
            Ok(status) => status.to_string(),
            Err(e) => e.clone(),
        }
    }
}

fn run_job(mut job: Job, logs: &Path) -> Outcome {
    let log = logs.join(format!("{}.log", job.name));
    let start = Instant::now();
    let status = File::create(&log)
        .and_then(|out| Ok((out.try_clone()?, out)))
        .and_then(|(stdout, stderr)| {
            job.command
                .stdin(Stdio::null())
                .stdout(stdout)
                .stderr(stderr)
                .status()
        })
        .map_err(|e| e.to_string());
    let time = start.elapsed();
    let text = fs::read(&log).unwrap_or_default();
    Outcome {
        name: job.name,
        log,
        status,
        time,
        summary: read_log(&String::from_utf8_lossy(&text)),
    }
}

fn run_jobs(jobs: Vec<Job>, threads: usize, logs: &Path) -> Vec<Outcome> {
    let total = jobs.len();
    let queue = Mutex::new(VecDeque::from(jobs));
    let outcomes = Mutex::new(Vec::with_capacity(total));
    let next = || queue.lock().unwrap().pop_front();
    thread::scope(|scope| {
        for _ in 0..threads.min(total) {
            scope.spawn(|| {
                while let Some(job) = next() {
                    let outcome = run_job(job, logs);
                    let mut outcomes = outcomes.lock().unwrap();
                    println!(
                        "[{:>3}/{total}] {:>7.1} s  {:<6} {}",
                        outcomes.len() + 1,
                        outcome.time.as_secs_f64(),
                        if outcome.ok() { "ok" } else { "FAILED" },
                        outcome.name,
                    );
                    outcomes.push(outcome);
                }
            });
        }
    });
    outcomes.into_inner().unwrap()
}

/// What the run leaves untested, as lines for the summary.
fn skipped(
    ws: &Workspace,
    options: &Options,
    built_features: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut left_out = usize::from(!options.selects(DOC));
    for target in &ws.targets {
        if !options.selects(&target.name) {
            left_out += 1;
        } else if target.exe.is_none() {
            lines.push(if target.required_features.is_empty() {
                format!("{}: cargo built no executable for it", target.name)
            } else {
                format!(
                    "{}: needs features {}",
                    target.name,
                    target.required_features.join(", ")
                )
            });
        }
    }
    if left_out > 0 {
        lines.push(format!("{left_out} targets left out by the options"));
    }
    for name in &ws.untested {
        lines.push(format!("{name}: test = false in its manifest"));
    }
    if !options.narrowed() {
        let off: Vec<String> = ws
            .features
            .iter()
            .flat_map(|(package, declared)| {
                let on = built_features.get(package);
                declared
                    .iter()
                    .filter(move |f| on.is_none_or(|on| !on.contains(*f)))
                    .map(move |f| format!("{package}/{f}"))
            })
            .collect();
        if !off.is_empty() {
            lines.push(format!("features no test built with: {}", off.join(", ")));
        }
    }
    if ws.not_tests > 0 {
        lines.push(format!(
            "{} examples and benches, which cargo test doesn't run",
            ws.not_tests
        ));
    }
    lines
}

fn report(outcomes: &mut [Outcome], skipped: &[String], build: Duration, wall: Duration) {
    outcomes.sort_by_key(|outcome| Reverse(outcome.time));
    println!("\nslowest:");
    for outcome in outcomes.iter().take(10) {
        println!("  {:>7.1} s  {}", outcome.time.as_secs_f64(), outcome.name);
    }
    if !skipped.is_empty() {
        println!("\nnot run:");
        for line in skipped {
            println!("  {line}");
        }
    }
    let failed: Vec<&Outcome> = outcomes.iter().filter(|o| !o.ok()).collect();
    if !failed.is_empty() {
        println!("\nfailed:");
        for outcome in &failed {
            println!(
                "  {}: {}, log {}",
                outcome.name,
                outcome.describe_status(),
                outcome.log.display()
            );
            for test in &outcome.summary.failed_tests {
                println!("    {test}");
            }
        }
    }
    let count = |f: fn(&LogSummary) -> u64| outcomes.iter().map(|o| f(&o.summary)).sum::<u64>();
    let total: Duration = outcomes.iter().map(|o| o.time).sum();
    println!(
        "\n{} targets: {} ok, {} failed. Tests: {} passed, {} failed, {} ignored.",
        outcomes.len(),
        outcomes.len() - failed.len(),
        failed.len(),
        count(|s| s.passed),
        count(|s| s.failed),
        count(|s| s.ignored),
    );
    println!(
        "Build {:.0} s, run {:.0} s, {:.0} s summed over targets.",
        build.as_secs_f64(),
        wall.as_secs_f64(),
        total.as_secs_f64(),
    );
}

pub(crate) fn run(args: Vec<String>) -> ExitCode {
    match try_run(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn try_run(args: Vec<String>) -> Result<ExitCode, String> {
    let Some(options) = Options::parse(args)? else {
        print!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    };
    let mut ws = workspace()?;
    let doc = options.selects(DOC);
    if options.list {
        let names = ws.selected(&options).map(|t| t.name.as_str());
        for name in names.chain(doc.then_some(DOC)) {
            println!("{name}");
        }
        return Ok(ExitCode::SUCCESS);
    }
    if !doc && ws.selected(&options).next().is_none() {
        return Err("no target matches".to_owned());
    }

    let start = Instant::now();
    let features = build(&mut ws, &options)?;
    let build_time = start.elapsed();

    let wrapper = env::var("ORR_TEST_WRAPPER").unwrap_or_default();
    let wrapper: Vec<String> = wrapper.split_whitespace().map(str::to_owned).collect();
    let mut jobs: Vec<Job> = ws
        .selected(&options)
        .filter_map(|t| {
            let exe = t.exe.as_ref()?;
            Some(test_job(t, exe, &wrapper, &options.test_args))
        })
        .collect();
    if doc {
        jobs.push(doc_job(&ws, &options.test_args));
    }
    schedule(&mut jobs);

    fs::create_dir_all(&ws.logs).map_err(|e| format!("{}: {e}", ws.logs.display()))?;
    let start = Instant::now();
    let mut outcomes = run_jobs(jobs, options.jobs, &ws.logs);
    let wall = start.elapsed();

    let not_run = skipped(&ws, &options, &features);
    report(&mut outcomes, &not_run, build_time, wall);
    let unbuilt = ws
        .selected(&options)
        .any(|t| t.exe.is_none() && t.required_features.is_empty());
    let not_started = outcomes.iter().any(|o| o.status.is_err());
    Ok(if unbuilt || not_started || outcomes.is_empty() {
        ExitCode::from(2)
    } else if outcomes.iter().all(Outcome::ok) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_add_up_across_result_lines() {
        let log = "\
running 3 tests
test a::passes ... ok
test a::breaks ... FAILED
test b::skipped ... ignored, needs a device

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 2 filtered out; finished in 0.50s

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
";
        assert_eq!(
            read_log(log),
            LogSummary {
                passed: 5,
                failed: 1,
                ignored: 1,
                failed_tests: vec!["a::breaks".to_owned()],
            }
        );
    }
}

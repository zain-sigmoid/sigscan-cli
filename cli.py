from __future__ import annotations
import logging
import traceback
from pathlib import Path
from importlib import metadata
from termcolor import colored, cprint
from typing import List, Optional, Set
from dataclasses import is_dataclass, asdict
from utils.logs_service.logger import AppLogger
import argparse, asyncio, json, os, sys, tempfile

AppLogger.init(
    level=logging.INFO,
    log_to_file=True,
)
from core.interfaces import analyzer_registry
from utils.prod_shift import Extract
from core.models import AnalysisConfiguration, SeverityLevel
from core.engine import (
    UnifiedAnalysisEngine as Engine,
)

from analyzers.robustness_analyzer import RobustnessAnalyzer
from analyzers.pii_analyzer import PIIAnalyzer
from analyzers.testability_analyzer import TestabilityAnalyzer
from analyzers.observability_analyzer import ObservabilityAnalyzer
from analyzers.readability_analyzer import ReadabilityAnalyzer
from analyzers.injection_analyzer import InjectionAnalyzer
from analyzers.maintainability_analyzer import MaintainabilityAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.compliance_analyzer import ComplianceAnalyzer
from analyzers.secrets_analyzer import HardcodedSecretsAnalyzer

logger = AppLogger.get_logger(__name__)
LOG_FILE = Path(__file__).resolve().parent / "logs" / "sigscan.log"


def _get_version() -> str:
    """Return the CLI version from installed metadata or pyproject."""
    try:
        return "sigscan version " + metadata.version("sigscan")
    except metadata.PackageNotFoundError:
        try:
            import tomllib  # type: ignore
        except ModuleNotFoundError:
            import tomli as tomllib

        if tomllib:
            try:
                data = tomllib.loads(Path("pyproject.toml").read_text())
                return "sigscan version " + str(
                    data.get("project", {}).get("version", "unknown")
                )
            except Exception:
                pass

    return "unknown"


def _read_log_file(path: Path) -> str:
    if not path.exists():
        logger.error(f"Log file not found at {path}")
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error(f"Unable to read log file {path}: {exc}")
        return False


def _set_quiet_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for handler in root_logger.handlers:
        handler.setLevel(logging.WARNING)


def initialize_analyzers() -> None:
    """Register all analyzers in the global registry."""
    analyzer_registry.register(HardcodedSecretsAnalyzer())
    analyzer_registry.register(RobustnessAnalyzer())
    analyzer_registry.register(PIIAnalyzer())
    analyzer_registry.register(TestabilityAnalyzer())
    analyzer_registry.register(ObservabilityAnalyzer())
    analyzer_registry.register(ReadabilityAnalyzer())
    analyzer_registry.register(InjectionAnalyzer())
    analyzer_registry.register(MaintainabilityAnalyzer())
    analyzer_registry.register(PerformanceAnalyzer())
    analyzer_registry.register(ComplianceAnalyzer())


def write_json_file(path: str, data: dict, *, compact: bool) -> None:
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=out_dir, encoding="utf-8"
    ) as tf:
        tmp = tf.name
        if compact:
            json.dump(
                data,
                tf,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
        else:
            json.dump(data, tf, ensure_ascii=False, indent=2, default=_json_default)
            tf.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    """Fallback converter for non-serializable types."""
    import datetime, pathlib, enum, dataclasses

    if isinstance(o, datetime.datetime):
        return o.isoformat()
    if isinstance(o, datetime.date):
        return o.isoformat()
    if isinstance(o, (pathlib.Path,)):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, enum.Enum):
        return o.value
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)

    # Try model_dump, to_dict, dict, json (Pydantic, etc.)
    for attr in ("to_dict", "dict", "model_dump", "json"):
        if hasattr(o, attr) and callable(getattr(o, attr)):
            try:
                v = getattr(o, attr)()
                if isinstance(v, str):
                    return json.loads(v)
                return v
            except Exception:
                pass

    return str(o)


def collect_code_files(target_path: str, *, exts: set[str] | None = None) -> list[str]:
    """Return a list of code files under target_path using Extract's filters."""
    exts = exts or set(Extract.CODE_EXTS)
    root = Path(target_path).resolve()

    # Let Extract decide the best project root (handles wrapper dirs)
    project_root = Extract.find_best_project_root(root, exts)
    count = Extract.count_code_files(project_root, exts)

    files: list[str] = []
    # Manual stack walk so we can prune hidden/excluded dirs eagerly
    stack: list[Path] = [project_root]
    while stack:
        d = stack.pop()
        try:
            for entry in d.iterdir():
                if entry.is_dir():
                    if not Extract.is_hidden_dir(entry):
                        stack.append(entry)
                else:
                    if Extract.is_code_file(entry, exts):
                        files.append(str(entry))
        except PermissionError:
            # ignore unreadable dirs
            continue
    return files, count


def eprint(msg: str, *, end: str = "\n"):
    print(msg, file=sys.stderr, end=end, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # First parse only --list-analyzers to detect it early
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--list-analyzers", action="store_true")
    known, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(
        prog="sigscan",
        description="Run signature scanning/analysis over a path with a configurable setup.",
    )
    p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to analyze. By default scan the current folder from terminal",
    )
    p.add_argument(
        "-a",
        "--analyzer",
        action="append",
        default=[],
        help="Enable only these analyzers (repeatable, by name).",
    )
    p.add_argument(
        "--all-analyzers", action="store_true", help="Enable all available analyzers."
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Does Parallel Processing for faster execution",
    )
    p.add_argument(
        "--include-low-confidence",
        action="store_true",
        help="Includes findings with Low Confidence",
    )
    p.add_argument(
        "--timeout", type=int, default=900, help="Waiting time in sec, default 900"
    )
    p.add_argument(
        "--max-findings",
        type=int,
        default=1000,
        help="Finding threshold for individual analyzer, default 1000",
    )
    p.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        help="Write JSON result to FILE (no stdout on success).",
    )
    p.add_argument("--compact", action="store_true", help="Minified JSON.")
    p.add_argument(
        "--no-progress", action="store_true", help="Hide Analyzer progress information"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Use for checking error i.e. traceback",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings and errors logs, hides info logs.",
    )
    p.add_argument(
        "--logs",
        action="store_true",
        help="Print the previous saved logs from log file.",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Show the sigscan CLI version and exit.",
    )
    p.add_argument(
        "--list-analyzers",
        action="store_true",
        help="List available analyzers and exit.",
    )

    # only enforce -o/--out if we are not listing analyzers
    args = p.parse_args(argv)
    if not (args.list_analyzers or args.logs or args.version) and not args.out:
        p.error("the following arguments are required: -o/--out")
    return args


class TqdmProgress:
    """Simple textual progress showing completed analyzers out of total."""

    def __init__(self, show: bool, desc: str = "Analyzing"):
        self.show = show
        self.total = 0
        self.completed = 0
        self.current_stage = desc
        self.total_known = False

    def __call__(self, increment=1, stage=None, total_analyzers=None):
        if total_analyzers is not None and not self.total_known:
            self.total = max(1, int(total_analyzers))
            self.total_known = True

        if stage:
            if "finished" in stage:
                color = "green"
            elif "running" in stage:
                color = "yellow"
            else:
                color = "cyan"
            self.current_stage = colored(f"[{stage}]", color, attrs=["bold"])

        if increment:
            next_count = self.completed + increment
            self.completed = min(next_count, self.total or next_count)

        if self.show:
            total_display = self.total if self.total else "?"
            print(
                f"{self.current_stage} {self.completed}/{total_display} analyzers completed",
                flush=True,
            )

    def close(self):
        if self.show:
            print()


# return report
async def _run_async(engine, cfg, show_progress: bool):
    tprog = TqdmProgress(show_progress, desc=colored("[starting]", "cyan"))
    try:
        report = await engine.analyze(cfg, progress_cb=tprog)
    except asyncio.CancelledError:
        logger.info("Analysis cancelled by user.")
        raise
    finally:
        tprog.close()
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.version:
        print(_get_version())
        return 0

    if args.logs:
        logs = _read_log_file(LOG_FILE)
        if logs:
            print(logs)
        return 0

    if args.quiet:
        _set_quiet_logging()

    exts = set(Extract.CODE_EXTS)
    initialize_analyzers()
    if args.list_analyzers:
        for name in analyzer_registry.list_analyzer_names():
            print(name)
        return 0
    target_path = os.path.abspath(args.path)
    reader_files, count = collect_code_files(target_path)
    logger.info(f"Total Files for Analysis : {count}")
    if args.quiet:
        cprint(f"Total Files for Analysis : {count}", "green")
    if not reader_files:
        logger.warning("⚠️ No Python files found after filtering.")
        return 0

    if not os.path.exists(target_path):
        logger.error(f"Error: path not found: {target_path}")
        return 2

    if not args.verbose:
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # Build configuration equivalent to Streamlit
    enabled_analyzers: Set[str] = set(args.analyzer or [])
    if args.all_analyzers:
        try:
            if Engine and hasattr(Engine, "available_analyzers"):
                enabled_analyzers = set(Engine.available_analyzers())  # type: ignore[attr-defined]
        except Exception:
            # Fallback: keep whatever user passed
            pass

    cfg = AnalysisConfiguration(
        target_path=target_path,
        enabled_analyzers=enabled_analyzers,
        severity_threshold=SeverityLevel.INFO,  # match Streamlit: capture all severities
        parallel_execution=bool(args.parallel),
        include_low_confidence=bool(args.include_low_confidence),
        timeout_seconds=int(args.timeout),
        max_findings_per_analyzer=int(args.max_findings),
        files=reader_files,
    )

    # Get engine (mirror what your Streamlit app builds)
    # If your app constructs engine via a factory, import and call it here.
    try:
        if Engine is None:
            raise ImportError("Engine import path not set. Update cli.py imports.")
        engine = Engine()  # adjust if needs params
    except Exception as e:
        logger.error(f"Error: cannot construct analysis engine: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    # Run analysis (async)
    try:
        if args.quiet and args.no_progress:
            cprint("Running", "green")
        report = asyncio.run(
            _run_async(engine, cfg, show_progress=not args.no_progress)
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted.")
        return 130
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    # Serialize
    try:
        if hasattr(report, "to_dict"):
            payload = report.to_dict()  # type: ignore
        elif is_dataclass(report):
            payload = asdict(report)  # type: ignore
        else:
            # Try common attributes; otherwise convert generically
            payload = (
                report.model_dump()
                if hasattr(report, "model_dump")
                else (
                    report.dict()
                    if hasattr(report, "dict")
                    else (
                        json.loads(report.json())
                        if hasattr(report, "json")
                        else (
                            report.__dict__
                            if hasattr(report, "__dict__")
                            else {"result": report}
                        )
                    )
                )
            )
    except Exception:
        # last-resort generic conversion
        payload = {"result": report}

    # Write JSON file
    try:
        write_json_file(args.out, payload, compact=args.compact)
        logger.info(f"✅ Findings saved to {args.out}")
        if args.quiet:
            cprint(f"✅ Findings saved to {args.out}", "green")
    except Exception as e:
        logger.error(
            f"❌ Failed to write output file '{args.out}': {e}, use verbose to check"
        )
        if args.verbose:
            traceback.print_exc()
        return 1

    # Success: no stdout output as requested
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

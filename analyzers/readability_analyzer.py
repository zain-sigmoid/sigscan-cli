# Flake8: noqa: E501
"""
Readability Analyzer for evaluating code readability and style.
Analyzes naming conventions, documentation, formatting, and code clarity.
"""

import os
import subprocess
import json
import logging
import statistics
import asyncio
import traceback
from collections import Counter
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from utils.logs_service.logger import AppLogger
from core.interfaces import QualityAnalyzer
from core.file_utils import find_python_files
from core.models import (
    AnalysisConfiguration,
    AnalysisResult,
    AnalysisMetrics,
    UnifiedFinding,
    FindingCategory,
    SeverityLevel,
    ComplexityLevel,
    CodeLocation,
)

logger = AppLogger.get_logger(__name__)


class ReadabilityAnalyzer(QualityAnalyzer):
    """
    Analyzer for evaluating code readability through Pylint and custom checks.
    """

    def __init__(self):
        super().__init__("readability", "1.0.0")
        self.quality_categories = ["naming", "documentation", "formatting", "clarity"]
        self.scores = []
        self.overall_details = Counter()
        self._initialize_readability_patterns()

    def get_supported_file_types(self) -> List[str]:
        """Return supported file types."""
        return [".py"]

    def get_quality_categories(self) -> List[str]:
        """Get quality categories this analyzer covers."""
        return self.quality_categories

    def get_quality_metrics(self) -> List[str]:
        """Get quality metrics this analyzer can provide."""
        return [
            "readability_score",
            "naming_issues_count",
            "documentation_issues_count",
            "formatting_issues_count",
            "clarity_issues_count",
            "total_readability_issues",
            "readability_coverage_percentage",
        ]

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this analyzer."""
        return {
            "enable_pylint": True,
            "pylint_timeout": 60,
            "focus_on_readability": True,
            "include_naming_conventions": True,
            "include_documentation_checks": True,
            "include_formatting_checks": True,
            "minimum_readability_score": 75.0,
            "exclude_test_files": False,
        }

    def _get_issues_to_report(self):
        issues = [
            "invalid-name",
            "bad-classmethod-argument",
            "bad-mcs-classmethod-argument",
            "missing-module-docstring",
            "missing-class-docstring",
            "missing-function-docstring",
            "bad-indentation",
            # "mixed-indentation",
            "trailing-whitespace",
            "missing-final-newline",
            "line-too-long",
            "unused-import",
            "unused-variable",
            # "redefined-outer-name",
            "too-many-locals",
            "too-many-arguments",
        ]
        return issues

    async def analyze(self, config: AnalysisConfiguration) -> AnalysisResult:
        """
        Perform readability analysis on the target files.

        Args:
            config: Analysis configuration

        Returns:
            Analysis result with findings and metrics
        """
        findings = []
        error_count = 0
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(
                f"Starting readability analysis of {os.path.basename(config.target_path)}"
            )

            # Find Python files
            # python_files = self._find_python_files(config.target_path)
            if getattr(config, "files", None):
                # Use the explicit file list passed from CLI
                python_files = config.files
            else:
                # Fallback: discover files automatically
                python_files = self._find_python_files(config.target_path)

            if not python_files:
                logger.warning(
                    f"No Python files found in {os.path.basename(config.target_path)}"
                )
                return self._create_empty_result()

            logger.info(f"Found {len(python_files)} Python files to analyze")

            # Get analyzer configuration
            analyzer_config = config.analyzer_configs.get(
                self.name, self.get_default_config()
            )
            active = await self._check_pylint_status()
            if not active:
                logger.error(
                    "Aborting Readability Ananlysis, pylint not found in the environment"
                )
                return

            # Perform readability analysis
            analysis_results = await self._perform_readability_analysis(
                python_files, analyzer_config
            )
            clubbed_findings = await self._club_analysis_results(analysis_results)
            # Generate findings based on analysis
            findings = await self._generate_findings(
                clubbed_findings, config.target_path, analyzer_config
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            metrics = AnalysisMetrics(
                analyzer_name=self.name,
                execution_time_seconds=execution_time,
                files_analyzed=len(python_files),
                findings_count=len(findings),
                error_count=error_count,
                success=True,
            )

            logger.info(
                f"Readability analysis completed: {len(findings)} findings in {execution_time:.2f}s"
            )

            return AnalysisResult(
                findings=findings,
                metrics=metrics,
                metadata={
                    "python_files_count": len(python_files),
                    "readability_score": analysis_results.get(
                        "overall_readability_score", 0.0
                    ),
                    "total_readability_issues": analysis_results.get("total_issues", 0),
                    "issues_by_category": analysis_results.get(
                        "issues_by_category", {}
                    ),
                },
            )

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Readability analysis failed: {str(e)}")
            error_count += 1
            execution_time = asyncio.get_event_loop().time() - start_time

            metrics = AnalysisMetrics(
                analyzer_name=self.name,
                execution_time_seconds=execution_time,
                files_analyzed=0,
                findings_count=0,
                error_count=error_count,
                success=False,
                error_message=str(e),
            )

            return AnalysisResult(
                findings=[], metrics=metrics, metadata={"error": str(e)}
            )

    async def _check_pylint_status(
        self,
    ) -> bool:
        #  Check if pylint is available
        try:
            proc = await asyncio.create_subprocess_exec(
                "pylint",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                logger.warning("Pylint not found. Install with: pip install pylint")
                return False
            return True
        except FileNotFoundError:
            logger.warning("Pylint not found. Install with: pip install pylint")
            return False

    def _initialize_readability_patterns(self):
        """Initialize readability issue patterns and mappings."""
        # Pylint message IDs focused on readability
        self.readability_issue_mapping = {
            ## error
            "syntax-error": {
                "category": "formatting",
                "title": "Parsing failed",
                "severity": SeverityLevel.HIGH,
                "complexity": ComplexityLevel.COMPLEX,
            },
            # Naming conventions
            "invalid-name": {
                "category": "naming",
                "title": "Invalid Naming Convention",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "bad-classmethod-argument": {
                "category": "naming",
                "title": "Bad Classmethod Argument Name",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "bad-mcs-classmethod-argument": {
                "category": "naming",
                "title": "Bad Metaclass Argument Name",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            # Documentation
            "missing-module-docstring": {
                "category": "documentation",
                "title": "Missing Module Documentation",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "missing-class-docstring": {
                "category": "documentation",
                "title": "Missing Class Documentation",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "missing-function-docstring": {
                "category": "documentation",
                "title": "Missing Function Documentation",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            # Formatting and style
            "bad-indentation": {
                "category": "formatting",
                "title": "Bad Indentation",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "mixed-indentation": {
                "category": "formatting",
                "title": "Mixed Indentation",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "trailing-whitespace": {
                "category": "formatting",
                "title": "Trailing Whitespace",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.TRIVIAL,
            },
            "missing-final-newline": {
                "category": "formatting",
                "title": "Missing Final Newline",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.TRIVIAL,
            },
            "line-too-long": {
                "category": "formatting",
                "title": "Line Too Long",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.SIMPLE,
            },
            # Code clarity
            "unused-import": {
                "category": "clarity",
                "title": "Unused Import",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.TRIVIAL,
            },
            "unused-variable": {
                "category": "clarity",
                "title": "Unused Variable",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.SIMPLE,
            },
            "redefined-outer-name": {
                "category": "clarity",
                "title": "Redefined Outer Name",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.MODERATE,
            },
            "too-many-locals": {
                "category": "clarity",
                "title": "Too Many Local Variables",
                "severity": SeverityLevel.MEDIUM,
                "complexity": ComplexityLevel.MODERATE,
            },
            "too-many-arguments": {
                "category": "clarity",
                "title": "Too Many Function Arguments",
                "severity": SeverityLevel.LOW,
                "complexity": ComplexityLevel.MODERATE,
            },
        }

    def get_issue_detail(self, symbol: str) -> str:
        """Return detail text for a pylint issue symbol."""
        ISSUE_DETAILS = {
            "missing-module-docstring": (
                "The module is missing a top-level docstring. "
                "Add a descriptive docstring at the top of the file to explain its purpose."
            ),
            "invalid-name": (
                "The name does not conform to naming conventions. "
                "Constants should use UPPER_CASE, variables and functions should use snake_case."
            ),
            "line-too-long": (
                "The line exceeds the configured maximum length. "
                "Break it into multiple lines or use line continuation for readability."
            ),
            "redefined-outer-name": (
                "A variable or function redefines a name from an outer scope. "
            ),
            "too-many-locals": (
                "The function defines too many local variables, reducing readability. "
            ),
            "unused-import": ("An import is declared but never used. "),
            "unused-variable": ("A variable is assigned but never used. "),
            "missing-function-docstring": (
                "The function or method is missing a docstring. "
                "Add a descriptive docstring that explains what the function does, "
                "its parameters, and its return value."
            ),
            "bad-indentation": (
                "The code is not indented according to Python's indentation rules. "
                "Ensure that blocks are indented with consistent spaces (usually 4 spaces per level) "
                "and that indentation aligns properly with the surrounding code."
            ),
            "missing-final-newline": (
                "The file does not end with a newline character. "
            ),
            "too-many-arguments": (
                "The function or method has too many parameters above recommended, which makes it hard to use and maintain."
            ),
            "missing-class-docstring": "The class is missing a docstring",
            "trailing-whitespace": "The line has extra spaces or tabs at the end.",
            "bad-classmethod-argument": "The first argument of a `@classmethod` is not named `cls`. By convention and readability standards, class methods should always use `cls` as their first parameter",
            "bad-mcs-classmethod-argument": "The first argument of a metaclass `@classmethod` is not named `mcs`. By convention, metaclass methods should always use `mcs` as their first parameter.",
            "syntax-error": "The underlying cause is an invalid Python grammar — missing punctuation, incorrect indentation, misplaced parentheses/brackets, etc",
        }
        return ISSUE_DETAILS.get(symbol, f"No details available for symbol: {symbol}")

    def _find_python_files(self, path: str) -> List[str]:
        """Find all Python files under the given path, excluding virtual environments."""
        return find_python_files(path, exclude_test_files=False)

    async def _perform_readability_analysis(
        self, python_files: List[str], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive readability analysis."""

        all_issues = []
        issues_by_category = {
            "naming": 0,
            "documentation": 0,
            "formatting": 0,
            "clarity": 0,
        }

        if config.get("enable_pylint", True):
            for file_path in python_files:
                # Skip test files if configured
                if config.get("exclude_test_files", False) and self._is_test_file(
                    file_path
                ):
                    continue

                pylint_issues = await self._run_pylint_analysis(file_path, config)
                spf = "/".join(file_path.split("/")[-2:])
                for issue in pylint_issues:
                    issue["file_path"] = spf or file_path
                    all_issues.append(issue)

                    # Count by category
                    category = issue.get("category", "clarity")
                    if category in issues_by_category:
                        issues_by_category[category] += 1

        # Calculate overall readability score
        total_issues = len(all_issues)
        total_files = len(python_files)

        # # Simple scoring algorithm: fewer issues = higher score
        # if total_files > 1:
        #     issues_per_file = total_issues / total_files
        #     # Scale: 0 issues = 100%, 10+ issues per file = 0%
        #     overall_score = max(0, min(100, 100 - (issues_per_file * 10)))
        # else:
        #     overall_score = 100 / total_issues
        overall_score = round(statistics.mean(self.scores), 2) * 10
        overall_result = dict(self.overall_details)
        total = sum(overall_result.values()) or 1
        overall_result["total_issues"] = total
        return {
            "all_issues": all_issues,
            "total_issues": total_issues,
            "issues_by_category": issues_by_category,
            "overall_readability_score": overall_score,
            "overall_readability_details": overall_result,
            "files_analyzed": total_files,
        }

    async def _run_pylint_analysis(
        self, file_path: str, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run Pylint analysis on a single file."""
        try:
            timeout = config.get("pylint_timeout", 60)

            # Run pylint with JSON output
            proc = await asyncio.create_subprocess_exec(
                "pylint",
                os.fspath(file_path),
                "-f",
                "json2",
                "--disable=all",
                "--enable=" + ",".join(self._get_issues_to_report()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output_text = (stdout or b"").decode()
            if not output_text and stderr:
                output_text = stderr.decode()

            if output_text:
                try:
                    pylint_output = json.loads(output_text)
                    pylint_stats = pylint_output.get("statistics", {})
                    self.scores.append(pylint_stats.get("score", 0))
                    msg_counts = pylint_stats.get("messageTypeCount", {})
                    self.overall_details.update(msg_counts)
                    return self._process_pylint_output(
                        pylint_output.get("messages"), file_path
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse Pylint JSON output for {file_path}: {e}"
                    )
                    return []
            else:
                # No issues found
                return []

        except subprocess.TimeoutExpired:
            logger.warning(f"Pylint analysis timed out for {file_path}")
            return []
        except FileNotFoundError:
            logger.warning("Pylint not found. Install with: pip install pylint")
            return []
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Error running Pylint on {file_path}: {str(e)}")
            return []

    def _process_pylint_output(
        self, pylint_output: List[Dict], file_path: str
    ) -> List[Dict[str, Any]]:
        """Process Pylint JSON output into our format."""
        processed_issues = []

        for issue in pylint_output:
            symbol = issue.get("symbol", "")
            message_id = issue.get("message-id", "")
            obj = issue.get("obj", "")

            # Map to our readability categories
            issue_info = self.readability_issue_mapping.get(
                symbol,
                {
                    "category": "clarity",
                    "title": f"Code Quality Issue: {symbol}",
                    "severity": SeverityLevel.LOW,
                    "complexity": ComplexityLevel.SIMPLE,
                },
            )
            spf = "/".join(file_path.split("/")[-2:])
            processed_issue = {
                "symbol": symbol,
                "message_id": message_id,
                "object": obj,
                "message": issue.get("message", ""),
                "line_number": issue.get("line", 0),
                "column": issue.get("column", 0),
                "category": issue_info["category"],
                "title": issue_info["title"],
                "severity": issue_info["severity"],
                "complexity": issue_info["complexity"],
                "file_path": spf or file_path,
                "details": self.get_issue_detail(symbol),
            }

            processed_issues.append(processed_issue)
        return processed_issues

    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file."""
        filename = os.path.basename(file_path).lower()
        return (
            filename.startswith("test_")
            or filename.endswith("_test.py")
            or "test" in filename
            or "/test" in file_path.lower()
        )

    def _get_object_mapping(self, symbol: str) -> str:
        OBJECT_SYMBOLS = {
            "missing-function-docstring": "Function",
            "too-many-arguments": "Function",
            "missing-class-docstring": "Class",
            "unused-variable": "In",
            "invalid-name": "In",
        }

        return OBJECT_SYMBOLS.get(symbol, "In")

    async def _club_analysis_results(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Club selected symbols per file into a single finding each, adding a `clubbed`
        dict with 'lines' and 'messages'. Others are returned unchanged.
        Updates total_issues and issues_by_category.
        """

        src_issues: List[Dict[str, Any]] = analysis_results.get("all_issues", [])

        # Symbols to club (normalized, lowercase)
        CLUB_SYMBOLS = {
            "line-too-long",
            "missing-class-docstring",
            "missing-function-docstring",
            "invalid-name",
            "too-many-arguments",
            "too-many-locals",
            "unused-variable",
            "unused-import",
            "trailing-whitespace",
            "bad-indentation",
        }

        # Fallback normalization via title → symbol (covers tools that set only title)
        TITLE_TO_SYMBOL = {
            "invalid name": "invalid-name",
            "line too long": "line-too-long",
            "missing class docstring": "missing-class-docstring",
            "missing function docstring": "missing-function-docstring",
            "invalid naming convention": "invalid-naming-convention",
            "too many function arguments": "too-many-arguments",
            "too many local variables": "too-many-locals",
            "unused variable": "unused-variable",
            "unused import": "unused-import",
            "trailing whitespace": "trailing-whitespace",
            "bad indentation": "bad-indentation",
        }

        CLUB_SYMBOLS_ONE = {"missing-module-docstring", "missing-final-newline"}

        # Keyed by (file_path, normalized_symbol)
        base_for_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        acc_for_key = defaultdict(lambda: {"lines": [], "messages": []})

        out: List[Dict[str, Any]] = []

        def normalize_symbol(issue: Dict[str, Any]) -> str:
            s = (issue.get("symbol") or "").strip().lower()
            if s:
                return s
            # fallback via title
            t = (issue.get("title") or "").strip().lower()
            return TITLE_TO_SYMBOL.get(t, "")

        for issue in src_issues:
            file_path = issue.get("file_path")
            sym = normalize_symbol(issue)
            # Only club our chosen symbols when file_path is present
            if file_path and sym in CLUB_SYMBOLS:
                key = (file_path, sym)

                # store first occurrence as representative base (shallow copy)
                if key not in base_for_key:
                    base_for_key[key] = dict(issue)

                # accumulate line numbers and messages
                ln = issue.get("line_number")
                if isinstance(ln, int):
                    acc_for_key[key]["lines"].append(ln)
                msg = issue.get("message")
                obj = issue.get("object")
                if msg and obj:
                    f_msg = f"{self._get_object_mapping(sym)} `{obj}` | {msg}"
                    acc_for_key[key]["messages"].append(f_msg)
                elif msg:
                    acc_for_key[key]["messages"].append(msg)
            elif file_path and sym in CLUB_SYMBOLS_ONE:
                key = (sym, "one")
                if key not in base_for_key:
                    base_for_key[key] = dict(issue)
                ln = issue.get("line_number")
                acc_for_key[key]["lines"].append(ln)
                acc_for_key[key]["messages"].append(file_path)
            else:
                # pass-through for non-clubbed or missing file_path
                out.append(issue)

        # Emit one merged finding per (file, symbol)
        for key, base in base_for_key.items():
            merged = dict(base)  # keep original fields (details stays as-is)
            collected = acc_for_key[key]
            if key[0] in CLUB_SYMBOLS_ONE:
                merged["file_path"] = (
                    merged["file_path"].split("/")[0]
                    if "/" in merged["file_path"]
                    else os.path.basename(merged["file_path"])
                )
            merged["message"] = merged["title"]
            merged["clubbed"] = {
                "lines": sorted(
                    list(x for x in collected["lines"] if isinstance(x, int))
                ),
                "messages": collected["messages"],
            }
            merged["line_number"] = ""
            out.append(merged)

        # --- Update summary ---
        new_total = len(out)
        by_cat: Dict[str, int] = defaultdict(int)
        for f in out:
            by_cat[f.get("category", "uncategorized")] += 1

        updated = dict(analysis_results)
        updated["all_issues"] = out
        updated["total_issues"] = new_total
        updated["issues_by_category"] = dict(by_cat)
        # keep overall_readability_score and files_analyzed unchanged
        return updated

    async def _generate_findings(
        self, analysis_results: Dict[str, Any], target_path: str, config: Dict[str, Any]
    ) -> List[UnifiedFinding]:
        """Generate findings based on readability analysis results."""
        findings = []

        # Check overall readability score
        overall_score = analysis_results["overall_readability_score"]
        overall_details = analysis_results["overall_readability_details"]
        minimum_threshold = config.get("minimum_readability_score", 75.0)
        target_path = str(target_path)
        spf = "/".join(target_path.split("/")[-2:])

        if overall_score < minimum_threshold:
            severity = (
                SeverityLevel.HIGH if overall_score < 50 else SeverityLevel.MEDIUM
            )
            finding = UnifiedFinding(
                title="Poor Code Readability",
                description=f"Code readability score is {overall_score:.1f}%, below recommended {minimum_threshold}%",
                details=overall_details,
                category=FindingCategory.QUALITY,
                severity=severity,
                confidence_score=0.8,
                location=CodeLocation(file_path=spf or target_path),
                rule_id="LOW_READABILITY_SCORE",
                remediation_guidance=f"Improve code readability to reach {minimum_threshold}% score",
                remediation_complexity=ComplexityLevel.MODERATE,
                source_analyzer=self.name,
                tags={"readability", "code_quality", "maintainability"},
                extra_data={
                    "readability_score": overall_score,
                    "minimum_threshold": minimum_threshold,
                    "issues_by_category": analysis_results["issues_by_category"],
                },
            )
            findings.append(finding)

        # Generate findings for individual readability issues
        for issue in analysis_results["all_issues"]:
            if issue["title"] == "Code Quality Issue: useless-option-value":
                continue
            finding = UnifiedFinding(
                title=issue["title"],
                description=issue["message"],
                details=issue["details"],
                clubbed=issue.get("clubbed", None),
                category=FindingCategory.QUALITY,
                severity=issue["severity"],
                confidence_score=0.8,
                location=CodeLocation(
                    file_path=issue["file_path"],
                    line_number=issue["line_number"],
                    column=issue["column"],
                ),
                rule_id=issue["symbol"],
                remediation_guidance=self._get_remediation_guidance(issue["symbol"]),
                remediation_complexity=issue["complexity"],
                source_analyzer=self.name,
                tags={"readability", issue["category"], "pylint"},
                extra_data={
                    "pylint_message_id": issue["message_id"],
                    "readability_category": issue["category"],
                },
            )
            findings.append(finding)

        return findings

    def _get_remediation_guidance(self, symbol: str) -> str:
        """Get specific remediation guidance for pylint symbols."""
        remediation_mapping = {
            "invalid-name": "Use descriptive names following Python naming conventions (snake_case for variables/functions, PascalCase for classes)",
            "missing-module-docstring": "Add a module-level docstring explaining the purpose of this module",
            "missing-class-docstring": "Add a class docstring explaining the purpose and usage of this class",
            "missing-function-docstring": "Add a function docstring explaining parameters, return value, and purpose",
            "bad-indentation": "Fix indentation to use consistent spacing (4 spaces per level in Python)",
            "line-too-long": "Break long lines into multiple lines (recommended max 88-100 characters)",
            "unused-import": "Remove unused imports to reduce clutter",
            "unused-variable": "Remove unused variables or prefix with underscore if intentionally unused",
            "too-many-arguments": "Reduce function arguments by grouping related parameters into objects or using *args/**kwargs",
            "too-many-locals": "Break function into smaller functions or use helper classes to reduce complexity or reducing variables.",
            "redefined-outer-name": "Rename a variable or function to avoid shadowing.",
            "missing-final-newline": (
                "Add a blank newline at the end of the file to follow POSIX standards "
                "and ensure consistent behavior across tools."
            ),
            "trailing-whitespace": "The line has extra spaces or tabs at the end. Remove them to improve cleanliness.",
            "bad-classmethod-argument": "Rename the first parameter of the class method to `cls` to follow Python conventions",
            "bad-mcs-classmethod-argument": "Rename the first parameter of the metaclass method to `mcs` to comply with Python conventions and improve clarity.",
            "syntax-error": "Check the filename, line number & caret pointer in the output",
        }

        return remediation_mapping.get(
            symbol, "Follow Python style guidelines and best practices"
        )

    def _create_empty_result(self) -> AnalysisResult:
        """Create an empty analysis result."""
        metrics = AnalysisMetrics(
            analyzer_name=self.name,
            execution_time_seconds=0.0,
            files_analyzed=0,
            findings_count=0,
            error_count=0,
            success=True,
        )
        return AnalysisResult(findings=[], metrics=metrics, metadata={})

"""
Compliance Analysis Module
Analyzes code for licensing and data-privacy compliance issues.
"""

import os
import subprocess
import json
import asyncio
import logging
import traceback
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict
from utils.logs_service.logger import AppLogger
from core.file_utils import find_python_files
from core.interfaces import ComplianceAnalyzer
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


class ComplianceAnalyzer(ComplianceAnalyzer):
    """Analyzer for code licensing and data privacy compliance issues."""

    def __init__(self):
        self.findings = []
        super().__init__("compliance", "1.0.0")
        self.supported_tools = ["ScanCode", "Semgrep"]
        self.quality_categories = [
            "License Compliance",
            "Data Privacy",
            "Copyright Issues",
        ]

    def get_supported_file_types(self) -> List[str]:
        """Return supported file types."""
        return [".py"]

    def get_quality_categories(self) -> List[str]:
        """Get quality categories this analyzer covers."""
        return self.quality_categories

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this analyzer."""
        return {""}

    def _find_python_files(self, path: str) -> List[str]:
        """Find all Python files under the given path, excluding virtual environments."""
        return find_python_files(path, exclude_test_files=False)

    def check_compliance(self, config: AnalysisConfiguration) -> Dict[str, bool]:
        """Check compliance based on the provided configuration."""
        # Placeholder implementation
        return {"GDPR": True, "CCPA": True}

    async def analyze(self, config: AnalysisConfiguration) -> AnalysisResult:
        """Run all compliance checks on provided codebase path."""
        # Discover all files including non-Python for license scans
        # python_files = find_python_files(codebase_path)
        error_count = 0
        start_time = asyncio.get_event_loop().time()

        # python_files = self._find_python_files(config.target_path)
        if getattr(config, "files", None):
            # Use the explicit file list passed from CLI
            python_files = config.files
        else:
            # Fallback: discover files automatically
            python_files = self._find_python_files(config.target_path)

        if not python_files:
            logger.warning(f"No Python files found in {config.target_path}")
            return self._create_empty_result()

        await self.check_license_compliance(config.target_path)
        # await self.check_data_privacy_compliance(config.target_path)

        execution_time = asyncio.get_event_loop().time() - start_time
        metrics = AnalysisMetrics(
            analyzer_name=self.name,
            execution_time_seconds=execution_time,
            files_analyzed=len(python_files),
            findings_count=len(self.findings),
            error_count=error_count,
            success=True,
        )
        logger.info(
            f"Compliance analysis completed: {len(self.findings)} findings in {execution_time:.2f}s"
        )
        findings = self._generate_findings(self.findings)
        return AnalysisResult(
            findings=findings,
            metrics=metrics,
            metadata={
                "python_files_count": len(python_files),
            },
        )

    def _generate_findings(
        self,
        results,
    ) -> List[UnifiedFinding]:
        """Generate findings asynchronously."""
        findings = []
        for finding in results:
            unified_finding = UnifiedFinding(
                title=f"{finding['type'].replace('_', ' ').title()}",
                severity=finding.get("severity", SeverityLevel.INFO),
                category=FindingCategory.COMPLIANCE,
                description=finding.get("description", ""),
                details=finding.get("details", None),
                clubbed=finding.get("clubbed", None),
                code_snippet=finding.get("code_snippet", None),
                confidence_score=0.8,
                location=CodeLocation(
                    file_path="/".join(finding.get("file", "").split("/")[-2:]),
                    line_number=finding.get("line", 0),
                ),
                rule_id=finding.get("rule_id", None),
                remediation_guidance=finding.get("suggestion", ""),
                remediation_complexity=ComplexityLevel.MODERATE,
                source_analyzer=self.name,
                tags={"test_files", "econ_files"},
            )
            findings.append(unified_finding)
        return findings

    async def run_semgrep_rules(
        self, target_path, rules_path="utils/privacy_rules.yml"
    ):
        """Runs semgrep with custom rules for data privacy checks."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "semgrep",
                "scan",
                target_path,
                "--config",
                rules_path,
                "--no-git-ignore",
                "--json-output=semgrep_output.json",
                "--quiet",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except subprocess.CalledProcessError:
            traceback.print_exc()

    def result_severity_mapping(self, license_name: str) -> str:
        mapping = {
            "detected_license_expression": "info",
            "license_detections": "medium",
            "license_clues": "low",
            "percentage_of_license_text": "info",
            "copyrights": "medium",
            "holders": "medium",
            "authors": "low",
            "emails": "low",
            "urls": "low",
        }
        return mapping.get(license_name, "info")

    def severity_mapping(self, severity: str) -> SeverityLevel:
        """Map SeverityLevel to string for display."""
        mapping = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "info": SeverityLevel.INFO,
        }
        return mapping.get(severity, SeverityLevel.INFO)

    async def check_license_compliance(self, codebase_path):
        """Checks for licensing compliance violations using ScanCode Toolkit output."""

        # Run ScanCode with output to file
        output_file = "scancode_report.json"
        try:
            proc = await asyncio.create_subprocess_exec(
                "scancode",
                "-clpeui",
                "--json-pp",
                output_file,
                codebase_path,
                "--quiet",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except subprocess.CalledProcessError as e:
            traceback.print_exc()
            self.findings.append(
                {
                    "type": "scancode_error",
                    "severity": SeverityLevel.INFO,
                    "description": f"ScanCode failed to run: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                    "suggestion": "Ensure ScanCode is correctly installed and the path is valid",
                }
            )
            return

        report_file = Path(output_file)
        if not report_file.exists():
            self.findings.append(
                {
                    "type": "report_missing",
                    "severity": SeverityLevel.INFO,
                    "description": "scancode_report.json was not generated.",
                    "suggestion": "Check ScanCode output path or rerun the scan.",
                }
            )
            logger.error("ScanCode report file not found.")
            return

        data = json.loads(report_file.read_text())
        grouped = defaultdict(list)
        for file_info in data.get("files", []):
            path = file_info.get("path")

            # Check each field and add finding if not empty
            if file_info.get("detected_license_expression"):
                grouped[
                    (
                        path,
                        "license_compliance",
                        self.result_severity_mapping("detected_license_expression"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.INFO,
                        "description": f"Detected license: {file_info['detected_license_expression']}",
                        "suggestion": "Review license for compatibility.",
                    }
                )

            if file_info.get("license_detections"):
                for lic in file_info["license_detections"]:
                    self.findings.append(
                        {
                            "file": path,
                            "type": "license_compliance",
                            "severity": SeverityLevel.MEDIUM,
                            "description": f"{len(file_info['license_detections'])} license detection(s) found.",
                            "suggestion": "Inspect license matches and verify usage rights.",
                            "details": lic,
                        }
                    )

            if file_info.get("license_clues"):
                grouped[
                    (
                        path,
                        "license_compliance",
                        self.result_severity_mapping("license_clues"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.LOW,
                        "description": "Potential license clues found in file.",
                        "suggestion": "Verify and clarify license references.",
                    }
                )

            if file_info.get("percentage_of_license_text", 0) > 0:
                grouped[
                    (
                        path,
                        "license_compliance",
                        self.result_severity_mapping("percentage_of_license_text"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.INFO,
                        "description": f"{file_info['percentage_of_license_text']}% license text detected.",
                        "suggestion": "Confirm if this file is a license or contains embedded license.",
                    }
                )

            if file_info.get("copyrights"):
                grouped[
                    (
                        path,
                        "copyright",
                        self.result_severity_mapping("copyrights"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.MEDIUM,
                        "description": "Copyright statement(s) found.",
                        "suggestion": "Check if attribution is required.",
                        "line": [
                            e["start_line"] for e in file_info.get("copyrights", [])
                        ],
                    }
                )

            if file_info.get("holders"):
                grouped[
                    (
                        path,
                        "copyright",
                        self.result_severity_mapping("holders"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.MEDIUM,
                        "description": "Copyright holder(s) listed.",
                        "suggestion": "Ensure holder rights are acknowledged properly.",
                        "line": [e["start_line"] for e in file_info.get("holders", [])],
                    }
                )

            if file_info.get("authors"):
                grouped[
                    (
                        path,
                        "copyright",
                        self.result_severity_mapping("authors"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.LOW,
                        "description": "Author(s) found in file.",
                        "suggestion": "Review author obligations if any.",
                        "line": [e["start_line"] for e in file_info.get("authors", [])],
                    }
                )

            if file_info.get("emails"):
                grouped[
                    (
                        path,
                        "copyright",
                        self.result_severity_mapping("emails"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.LOW,
                        "description": f"{len(file_info['emails'])} email(s) found.",
                        "suggestion": "Ensure these do not leak personal data or violate compliance.",
                        "line": [e["start_line"] for e in file_info.get("emails", [])],
                    }
                )

            if file_info.get("urls"):
                grouped[
                    (
                        path,
                        "copyright",
                        self.result_severity_mapping("urls"),
                    )
                ].append(
                    {
                        "severity": SeverityLevel.LOW,
                        "description": f"{len(file_info['urls'])} URL(s) found.",
                        "suggestion": "Verify these URLs do not point to prohibited or unverified sources.",
                        "line": [e["start_line"] for e in file_info.get("urls", [])],
                    }
                )
        for (path, ftype, severity), items in grouped.items():
            clubbed = {
                "description": [i["description"] for i in items],
                "suggestion": [i["suggestion"] for i in items],
                "lines": [
                    (
                        ", ".join(map(str, i.get("line")))
                        if isinstance(i.get("line"), list)
                        else str(i.get("line"))
                    )
                    for i in items
                ],
            }
            self.findings.append(
                {
                    "type": ftype,
                    "file": path,
                    "severity": self.severity_mapping(severity),
                    "clubbed": clubbed,
                    "description": f"{len(items)} {ftype.replace('_', ' ')} issue(s) found in {os.path.basename(path)}.",
                    "suggestion": "Review the clubbed details for remediation steps.",
                    "rule_id": ftype,
                }
            )

    def process_semgrep_findings(self, json_path="semgrep_output.json"):
        """Parses Semgrep JSON output and appends structured findings."""
        SEVERITY_MAP = {
            "insecure-transmission": SeverityLevel.HIGH,
            "sensitive-data-logging": SeverityLevel.HIGH,
            "missing-data-anonymization": SeverityLevel.MEDIUM,
            "retention-policy-violation": SeverityLevel.MEDIUM,
            "missing-deletion-mechanism": SeverityLevel.LOW,
        }

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            traceback.print_exc()
            self.findings.append(
                {
                    "type": "semgrep_parse_error",
                    "severity": SeverityLevel.HIGH,
                    "description": f"Failed to read Semgrep output: {str(e)}",
                    "suggestion": "Ensure semgrep_output.json exists and is valid JSON.",
                }
            )
            return

        grouped_findings = defaultdict(
            lambda: {
                "check_id": "",
                "path": "",
                "lines": [],
                "message": "",
                "severity": "",
                "category": "",
                "compliance": "",
            }
        )

        violation_counter = defaultdict(int)

        for result in data.get("results", []):
            check_id = result.get("check_id", "")
            path = result.get("path", "")
            start_line = result.get("start", {}).get("line")
            msg = result.get("extra", {}).get("message", "")
            sev = result.get("extra", {}).get("severity", "")
            meta = result.get("extra", {}).get("metadata", {})

            key = (path, check_id)
            grouped = grouped_findings[key]

            grouped["check_id"] = check_id
            grouped["path"] = path
            grouped["message"] = msg
            grouped["severity"] = sev
            grouped["category"] = meta.get("category", "")
            grouped["compliance"] = meta.get("compliance", "")
            if start_line:
                grouped["lines"].append(start_line)

            violation_counter[path] += 1

        # Store per-file violation summary
        self.violation_summary = {
            path: f"{count} violation(s)" for path, count in violation_counter.items()
        }
        # Append findings with merged lines
        for (path, check_id), details in grouped_findings.items():
            lines_str = ", ".join(str(ln) for ln in sorted(set(details["lines"])))
            total_violations = violation_counter[details["path"]]
            type_ = details["check_id"].split(".")[-1]
            severity = SEVERITY_MAP.get(type_, SeverityLevel.INFO)
            title = type_.replace("-", " ").title()
            description = ""
            if type_ == "missing-data-anonymization":
                description = (
                    f"{title}: {details['message']} Total in file {total_violations}"
                )
            else:

                description = f"{title}: {details['message']}"
            snippet = f"Line(s): {lines_str}"
            self.findings.append(
                {
                    "type": "data_privacy",
                    "severity": severity,
                    "file": details["path"],
                    "rule_id": details["check_id"].split(".")[-1],
                    "line": "",
                    "description": (description),
                    "code_snippet": snippet,
                    "category": details["category"],
                    "compliance": details["compliance"],
                    "suggestion": "Review this code for potential privacy/security issues.",
                }
            )

    async def check_data_privacy_compliance(self, codebase_path):
        """Checks for data privacy compliance violations (GDPR, CCPA)."""
        # Run semgrep rules defined for privacy
        await self.run_semgrep_rules(codebase_path)
        self.process_semgrep_findings()

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

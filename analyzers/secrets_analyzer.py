"""
Hardcoded Secrets Analyzer - Refactored from app_Hardcoded_Secrets.py
Extracts analysis logic from UI components.
"""

# Flake8: noqa: E501
import asyncio
import subprocess
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from utils.logs_service.logger import AppLogger
from utils.prod_shift import ensure_gitleaks
from core.interfaces import SecurityAnalyzer, AnalysisResult
from core.models import (
    UnifiedFinding,
    AnalysisConfiguration,
    AnalysisMetrics,
    CodeLocation,
    SeverityLevel,
    FindingCategory,
    ComplexityLevel,
)


logger = AppLogger.get_logger(__name__)


class HardcodedSecretsAnalyzer(SecurityAnalyzer):
    """
    Analyzer for detecting hardcoded secrets using Gitleaks.
    """

    def __init__(self):
        super().__init__("hardcoded_secrets", "1.0.0")
        self.cwe_mapping = self._initialize_cwe_mapping()
        self.gitleaks = "gitleaks"

    def get_supported_file_types(self) -> List[str]:
        """Return supported file types (all files for secrets scanning)."""
        return ["*"]  # Secrets can be in any file type

    def get_security_categories(self) -> List[str]:
        """Get security categories this analyzer covers."""
        return ["secrets", "credentials", "hardcoded_data"]

    def get_cwe_mappings(self) -> Dict[str, str]:
        """Get mapping of rules to CWE identifiers."""
        return self.cwe_mapping

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this analyzer."""
        return {
            "include_entropy_checks": True,
            "max_file_size_mb": 10,
            "exclude_test_files": True,
            "confidence_threshold": 0.7,
            "report_format": "json",
        }

    async def analyze(self, config: AnalysisConfiguration) -> AnalysisResult:
        """
        Perform hardcoded secrets analysis.

        Args:
            config: Analysis configuration

        Returns:
            Analysis result with findings and metrics
        """
        start_time = 0  # Will be set by timing wrapper
        findings = []
        error_count = 0

        try:
            logger.info(
                f"Starting hardcoded secrets analysis of {os.path.basename(config.target_path)}"
            )

            # Check if Gitleaks is available
            if not await self._check_gitleaks_available():
                raise RuntimeError("Gitleaks is not installed or not found in PATH")

            # Run Gitleaks scan
            gitleaks_results = await self._run_gitleaks_scan(config.target_path)

            if gitleaks_results:
                # Transform Gitleaks results to unified findings
                findings = self._transform_gitleaks_results(gitleaks_results)
                logger.info(f"Found {len(findings)} secret findings")
            else:
                logger.info("No secrets detected")

            # Count analyzed files
            files_analyzed = self._count_analyzed_files(config.target_path)

            metrics = AnalysisMetrics(
                analyzer_name=self.name,
                execution_time_seconds=0,  # Will be set by engine
                files_analyzed=files_analyzed,
                findings_count=len(findings),
                error_count=error_count,
                success=True,
            )

            return AnalysisResult(
                findings=findings,
                metrics=metrics,
                metadata={
                    "gitleaks_version": await self._get_gitleaks_version(),
                    "scan_type": "detect",
                },
            )

        except Exception as e:
            logger.error(f"Hardcoded secrets analysis failed: {str(e)}")
            error_count = 1

            metrics = AnalysisMetrics(
                analyzer_name=self.name,
                execution_time_seconds=0,
                files_analyzed=0,
                findings_count=0,
                error_count=error_count,
                success=False,
                error_message=str(e),
            )

            return AnalysisResult(
                findings=[], metrics=metrics, metadata={"error": str(e)}
            )

    async def _check_gitleaks_available(self) -> bool:
        """Check if Gitleaks is available in the system."""
        try:
            gitleaks_bin = ensure_gitleaks()
            self.gitleaks = gitleaks_bin
            proc = await asyncio.create_subprocess_exec(
                gitleaks_bin,
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode == 0
        except (asyncio.TimeoutError, subprocess.TimeoutExpired, FileNotFoundError):
            traceback.print_exc()
            logger.error("Gitleaks error")
            return False

    async def _get_gitleaks_version(self) -> Optional[str]:
        """Get Gitleaks version."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.gitleaks,
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return stdout.decode().strip()
        except (asyncio.TimeoutError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    async def _run_gitleaks_scan(
        self, source_path: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run Gitleaks scan on the source path.

        Args:
            source_path: Path to scan

        Returns:
            List of Gitleaks findings or None if no findings
        """
        try:
            # Create temporary file for report
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as temp_file:
                report_path = temp_file.name
            gitleaks_toml_path = "utils/gitleaks.toml"
            gitleaks_toml_path = os.path.abspath(gitleaks_toml_path)
            # using gitleaks v8.1+ command which also accepts custom rules
            commandv2 = [
                self.gitleaks,
                "dir",
                source_path,
                "-c",
                gitleaks_toml_path,
                "-f",
                "json",
                "-r",
                report_path,
            ]

            cmd = [os.fspath(x) for x in commandv2]
            logger.debug(f"Running gitleaks command:")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=300)

            # Read results from report file
            try:
                if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
                    with open(report_path, "r") as f:
                        results = json.load(f)

                    # Clean up temp file
                    os.unlink(report_path)

                    return results if isinstance(results, list) else []
                else:
                    # No findings
                    if os.path.exists(report_path):
                        os.unlink(report_path)
                    return []

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gitleaks JSON output: {str(e)}")
                if os.path.exists(report_path):
                    os.unlink(report_path)
                return []

        except asyncio.TimeoutError:
            logger.error("Gitleaks scan timed out")
            return []
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Gitleaks scan failed: {str(e)}")
            return []

    def _is_git_repo(self, path: str) -> bool:
        """Check if the path is a Git repository."""
        return Path(path, ".git").exists()

    def _transform_gitleaks_results(
        self, gitleaks_results: List[Dict[str, Any]]
    ) -> List[UnifiedFinding]:
        """
        Transform Gitleaks results to unified findings.

        Args:
            gitleaks_results: Raw Gitleaks results

        Returns:
            List of unified findings
        """
        grouped = defaultdict(list)
        findings = []

        for result in gitleaks_results:
            key = (result.get("File", ""), result.get("RuleID", ""))
            grouped[key].append(result)
        try:
            findings = self._create_unified_finding(grouped)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to transform Gitleaks result: {str(e)}")

        return findings

    def _create_unified_finding(
        self, grouped: Dict[Any, List[Dict[str, Any]]]
    ) -> Optional[UnifiedFinding]:
        """Create a unified finding from Gitleaks result."""
        unified_findings = []
        for (file_path, rule_id), group_items in grouped.items():
            # Collect line numbers and snippets for context
            lines = sorted(
                [
                    item.get("StartLine", item.get("Line"))
                    for item in group_items
                    if item.get("StartLine") or item.get("Line")
                ]
            )

            snippets = [
                (
                    (item.get("Match") or item.get("Secret") or "")
                    if len(item.get("Match", "")) < 50
                    else (item.get("Match") or item.get("Secret") or "")[:50] + "..."
                )
                for item in group_items
            ]

            # Take first item for metadata
            base_item = group_items[0]
            cwe_info = self._map_rule_to_cwe(rule_id)
            severity = self._determine_severity(rule_id)
            location = CodeLocation(
                file_path="/".join(file_path.split("/")[-2:]),
            )

            clubbed = {
                "lines": lines,
                "snippets": snippets,
            }
            unified_findings.append(
                UnifiedFinding(
                    title=f"{' '.join(rule_id.split('-')).title()}",
                    description=base_item.get(
                        "Description",
                        f"Potential hardcoded secret detected ({len(lines)} occurrences)",
                    ),
                    clubbed=clubbed,
                    category=FindingCategory.SECURITY,
                    severity=severity,
                    location=location,
                    rule_id=rule_id,
                    cwe_id=cwe_info.get("cwe"),
                    # code_snippet="\n".join(snippets),  # show up to 3 examples
                    remediation_guidance=self._get_remediation_guidance(rule_id),
                    remediation_complexity=self._get_remediation_complexity(rule_id),
                    source_analyzer=self.name,
                    compliance_frameworks=["PCI-DSS", "SOX", "GDPR", "HIPAA"],
                    confidence_score=self._calculate_confidence(base_item),
                    tags={"secrets", "credentials", "security", "hardcoded"},
                    extra_data={
                        "gitleaks_grouped": clubbed,
                        "cwe_description": cwe_info.get("description"),
                        "cwe_url": cwe_info.get("url"),
                    },
                )
            )
        # file_path = gitleaks_result.get("File", "")
        # line_number = gitleaks_result.get("StartLine", gitleaks_result.get("Line"))
        # rule_id = gitleaks_result.get("RuleID", "")

        # # Get CWE mapping
        # cwe_info = self._map_rule_to_cwe(rule_id)

        # # Determine severity
        # severity = self._determine_severity(rule_id)

        # # Create code location
        # location = CodeLocation(
        #     file_path="/".join(file_path.split("/")[-2:]),
        #     line_number=line_number,
        #     end_line_number=gitleaks_result.get("EndLine"),
        # )

        # # Get secret snippet (truncated for security)
        # matched = gitleaks_result.get("Match", "")
        # secret = gitleaks_result.get("Secret", "")
        # rule_id = gitleaks_result.get("RuleID", "unknown")
        # if matched:
        #     code_snippet = matched[:50] + "..." if len(matched) > 50 else matched
        # else:
        #     code_snippet = secret[:50] + "..." if len(secret) > 50 else secret

        # # Create unified finding
        # finding = UnifiedFinding(
        #     title=f"Hardcoded Secret: {rule_id}",
        #     description=gitleaks_result.get(
        #         "Description", f"Potential hardcoded secret detected: {rule_id}"
        #     ),
        #     category=FindingCategory.SECURITY,
        #     severity=severity,
        #     location=location,
        #     rule_id=rule_id,
        #     cwe_id=cwe_info.get("cwe"),
        #     code_snippet=code_snippet,
        #     remediation_guidance=self._get_remediation_guidance(rule_id),
        #     remediation_complexity=self._get_remediation_complexity(rule_id),
        #     source_analyzer=self.name,
        #     compliance_frameworks=["PCI-DSS", "SOX", "GDPR", "HIPAA"],
        #     confidence_score=self._calculate_confidence(gitleaks_result),
        #     tags={"secrets", "credentials", "security", "hardcoded"},
        #     extra_data={
        #         "gitleaks_data": gitleaks_result,
        #         "cwe_description": cwe_info.get("description"),
        #         "cwe_url": cwe_info.get("url"),
        #     },
        # )

        return unified_findings

    def _map_rule_to_cwe(self, rule_id: str) -> Dict[str, str]:
        """Map Gitleaks rule ID to CWE information."""
        return self.cwe_mapping.get(
            rule_id,
            {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
        )

    def _determine_severity(self, rule_id: str) -> SeverityLevel:
        """Determine severity based on rule type."""
        critical_patterns = [
            "private-key",
            "secret-key",
            "aws-secret",
            "rsa-private-key",
        ]
        high_patterns = ["password", "token", "api-key", "oauth", "github-pat"]
        medium_patterns = ["url", "connection-string"]

        rule_lower = rule_id.lower()

        if any(pattern in rule_lower for pattern in critical_patterns):
            return SeverityLevel.CRITICAL
        elif any(pattern in rule_lower for pattern in high_patterns):
            return SeverityLevel.HIGH
        elif any(pattern in rule_lower for pattern in medium_patterns):
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.HIGH  # Default to HIGH for unknown secret types

    def _calculate_confidence(self, gitleaks_result: Dict[str, Any]) -> float:
        """Calculate confidence score for the finding."""
        # Gitleaks doesn't provide confidence, so we estimate based on rule type
        rule_id = gitleaks_result.get("RuleID", "").lower()

        # High confidence rules (specific patterns)
        high_confidence_patterns = ["github-pat", "aws-secret-key", "private-key"]

        # Medium confidence rules (generic patterns)
        medium_confidence_patterns = ["password", "secret", "token"]

        if any(pattern in rule_id for pattern in high_confidence_patterns):
            return 0.9
        elif any(pattern in rule_id for pattern in medium_confidence_patterns):
            return 0.7
        else:
            return 0.8  # Default confidence

    def _get_remediation_guidance(self, rule_id: str) -> str:
        """Get remediation guidance for the specific rule."""
        guidance_map = {
            "aws-secret-key": "Move AWS credentials to environment variables or AWS credential profiles. Use IAM roles for EC2 instances.",
            "github-pat": "Store GitHub tokens in secure environment variables or use GitHub's secret management.",
            "private-key": "Store private keys in secure key management systems. Never commit private keys to version control.",
            "password": "Use environment variables or secure configuration management for passwords.",
            "api-key": "Store API keys in environment variables or secure configuration files not in version control.",
        }

        # Find matching guidance
        for pattern, guidance in guidance_map.items():
            if pattern in rule_id.lower():
                return guidance

        return "Store sensitive data in environment variables or secure configuration management systems. Remove hardcoded secrets from source code."

    def _get_remediation_complexity(self, rule_id: str) -> ComplexityLevel:
        """Estimate remediation complexity for the rule."""
        simple_fixes = ["password", "api-key", "token"]
        moderate_fixes = ["aws-secret-key", "github-pat"]
        complex_fixes = ["private-key", "certificate"]

        rule_lower = rule_id.lower()

        if any(pattern in rule_lower for pattern in simple_fixes):
            return ComplexityLevel.SIMPLE
        elif any(pattern in rule_lower for pattern in moderate_fixes):
            return ComplexityLevel.MODERATE
        elif any(pattern in rule_lower for pattern in complex_fixes):
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.MODERATE

    def _count_analyzed_files(self, source_path: str) -> int:
        """Count the number of files that were analyzed."""
        if os.path.isfile(source_path):
            return 1

        count = 0
        for root, dirs, files in os.walk(source_path):
            # Skip common non-source directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in ["node_modules", "__pycache__"]
            ]
            count += len(files)

        return count

    def _initialize_cwe_mapping(self) -> Dict[str, Dict[str, str]]:
        """Initialize the CWE mapping for Gitleaks rules."""
        return {
            # AWS and Cloud credentials
            "aws-access-token": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            "aws-secret-key": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            "aws-mws-key": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            "aws-session-token": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            # API Keys and Tokens
            "github-pat": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            "github-oauth": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            "gitlab-pat": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            "google-api-key": {
                "cwe": "CWE-321",
                "description": "Use of Hard-coded Cryptographic Key",
                "url": "https://cwe.mitre.org/data/definitions/321.html",
            },
            "slack-access-token": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            "discord-api-token": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            # Database credentials
            "mysql": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            "postgres": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            "mongodb": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
            # Generic passwords and keys
            "password": {
                "cwe": "CWE-259",
                "description": "Use of Hard-coded Password",
                "url": "https://cwe.mitre.org/data/definitions/259.html",
            },
            "private-key": {
                "cwe": "CWE-321",
                "description": "Use of Hard-coded Cryptographic Key",
                "url": "https://cwe.mitre.org/data/definitions/321.html",
            },
            "jwt": {
                "cwe": "CWE-522",
                "description": "Insufficiently Protected Credentials",
                "url": "https://cwe.mitre.org/data/definitions/522.html",
            },
            "api-key": {
                "cwe": "CWE-321",
                "description": "Use of Hard-coded Cryptographic Key",
                "url": "https://cwe.mitre.org/data/definitions/321.html",
            },
            "secret": {
                "cwe": "CWE-798",
                "description": "Hardcoded Credentials",
                "url": "https://cwe.mitre.org/data/definitions/798.html",
            },
        }

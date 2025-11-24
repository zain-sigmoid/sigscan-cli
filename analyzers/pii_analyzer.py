# Flake8: noqa: E501
"""
PII/PHI Analyzer for detecting personally identifiable information and protected health information.
Ensures compliance with data protection regulations like GDPR, HIPAA, and CCPA.
"""

import os
import re
import ast
import asyncio
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
from utils.logs_service.logger import AppLogger
from core.interfaces import ComplianceAnalyzer
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
    ComplianceStatus,
)

logger = AppLogger.get_logger(__name__)


class PIIType(Enum):
    """Types of PII/PHI detected."""

    EMAIL = "Email Address"
    PHONE = "Phone Number"
    SSN = "Social Security Number"
    CREDIT_CARD = "Credit Card Number"
    IP_ADDRESS = "IP Address"
    MRN = "Medical Record Number"
    DOB = "Date of Birth"
    DRIVERS_LICENSE = "Driver's License"
    PASSPORT = "Passport Number"
    PHI = "Protected Health Information"
    SAMPLE_PII = "Sample/Test PII Data"
    SENSITIVE_VAR = "Sensitive Variable Name"
    PII_LOGGING = "PII Data Logging"
    AADHAAR = "Aadhaar Unique ID Number"
    PAN = "Pan Card Number"


class PHIType(Enum):
    PATIENT = "Patient Name"


class PIIAnalyzer(ComplianceAnalyzer):
    """
    Analyzer for detecting PII/PHI data in Python code to ensure compliance
    with data protection regulations.
    """

    def __init__(self):
        super().__init__("pii_phi", "1.0.0")
        self.compliance_frameworks = ["GDPR", "HIPAA", "CCPA", "PCI_DSS"]
        self._initialize_patterns()
        self.quality_categories = [
            "Personal Identity Information",
            "Personal Health Information",
        ]

    def get_supported_file_types(self) -> List[str]:
        """Return supported file types."""
        return [".py"]

    def get_compliance_frameworks(self) -> List[str]:
        """Get compliance frameworks this analyzer covers."""
        return self.compliance_frameworks

    def get_quality_categories(self) -> List[str]:
        """Get quality categories this analyzer covers."""
        return self.quality_categories

    def get_compliance_status(
        self, findings: List["UnifiedFinding"]
    ) -> ComplianceStatus:
        """Get compliance status based on findings."""
        # Count critical findings that affect compliance
        critical_count = sum(
            1 for f in findings if f.severity == SeverityLevel.CRITICAL
        )
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)

        # Determine overall compliance status
        if critical_count > 0:
            status = "non_compliant"
        elif high_count > 5:
            status = "partially_compliant"
        else:
            status = "compliant"

        return ComplianceStatus(
            framework="Data Protection",
            status=status,
            findings_count=len(findings),
            compliance_score=max(
                0.0, 1.0 - (critical_count * 0.3) - (high_count * 0.1)
            ),
        )

    def check_compliance(self, config: AnalysisConfiguration) -> Dict[str, bool]:
        """
        Check compliance status for all frameworks.

        Args:
            config: Analysis configuration

        Returns:
            Dictionary mapping framework names to compliance status
        """
        # Run analysis first to get findings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.analyze(config))

        compliance_results = {}

        for framework in self.compliance_frameworks:
            framework_upper = framework.upper()

            # Count relevant findings for this framework
            relevant_findings = [
                f
                for f in result.findings
                if framework_upper in [cf.upper() for cf in f.compliance_frameworks]
            ]

            # Framework-specific compliance rules
            if framework_upper == "GDPR":
                # GDPR is strict about any PII data exposure
                critical_pii = sum(
                    1
                    for f in relevant_findings
                    if f.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
                )
                compliance_results[framework] = critical_pii == 0

            elif framework_upper == "HIPAA":
                # HIPAA is very strict about PHI
                phi_findings = sum(
                    1
                    for f in relevant_findings
                    if "PHI" in f.extra_data.get("pii_type", "")
                )
                compliance_results[framework] = phi_findings == 0

            elif framework_upper == "CCPA":
                # CCPA focuses on consumer data protection
                consumer_pii = sum(
                    1
                    for f in relevant_findings
                    if f.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
                )
                compliance_results[framework] = consumer_pii <= 2  # Allow minor issues

            elif framework_upper == "PCI_DSS":
                # PCI DSS is strict about payment card data
                payment_findings = sum(
                    1
                    for f in relevant_findings
                    if "credit_card" in f.extra_data.get("pattern_name", "").lower()
                )
                compliance_results[framework] = payment_findings == 0

            else:
                # Default: allow minor issues but no critical ones
                critical_count = sum(
                    1 for f in relevant_findings if f.severity == SeverityLevel.CRITICAL
                )
                compliance_results[framework] = critical_count == 0

        return compliance_results

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this analyzer."""
        return {
            "enable_pattern_detection": True,
            "enable_ast_analysis": True,
            "enable_phi_detection": True,
            "filter_false_positives": True,
            "filter_test_data": True,
            "enable_variable_analysis": True,
            "enable_logging_analysis": True,
            "risk_threshold": "medium",  # minimum risk level to report
        }

    async def analyze(self, config: AnalysisConfiguration) -> AnalysisResult:
        """
        Perform PII/PHI analysis on the target files.

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
                f"Starting PII/PHI analysis of {os.path.basename(config.target_path)}"
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

            # Analyze each file
            for file_path in python_files:
                try:
                    file_findings = await self._analyze_file(file_path, analyzer_config)
                    findings.extend(file_findings)
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path}: {str(e)}")
                    error_count += 1
            # Apply post-processing filters
            if analyzer_config.get("filter_false_positives", True):
                findings = self._filter_false_positives(findings)
            if analyzer_config.get("filter_test_data", True):
                findings = self._filter_test_data(findings)
            # Remove duplicates
            findings = self._deduplicate_findings(findings)
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
                f"PII/PHI analysis completed: {len(findings)} findings in {execution_time:.2f}s"
            )

            return AnalysisResult(
                findings=findings,
                metrics=metrics,
                metadata={
                    "python_files_count": len(python_files),
                    "pii_types_found": list(
                        set(
                            f.extra_data.get("pii_type")
                            for f in findings
                            if f.extra_data.get("pii_type")
                        )
                    ),
                    "compliance_frameworks": self.compliance_frameworks,
                },
            )

        except Exception as e:
            logger.error(f"PII/PHI analysis failed: {str(e)}")
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

    def _initialize_patterns(self):
        """Initialize PII/PHI detection patterns."""
        self.patterns = {
            "email": r"(?:email\s*:?\s*)?(?P<value>[a-z0-9](?:[a-z0-9._%+-]*[a-z0-9])?@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,})",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?[2-9][0-8][0-9]\)?[-.\s]?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "ip_address": r"\b(?:(?!(?:10|127|169\.254|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.)(?:[0-9]{1,3}\.){3}[0-9]{1,3})\b",
            "pan": r"(?i)\b(?:pan(?:\s*no\.?)?\s*[:=]?\s*)?(?P<value>[A-Z]{5}[0-9]{4}[A-Z])\b",
            "aadhaar": r"(?i)\b(?:aa?dhaar|uid)\s*[:=]?\s*(?P<value>\d{4}[\s-]?\d{4}[\s-]?\d{4})\b",
            "mrn": r'(?i)\b(?:mrn|medical[-_\s]?record|patient[-_\s]?id)\s*[:=]?\s*[\'"]?\s*(?P<value>[A-Z]{1,3}-\d{5,15})\s*[\'"]?',
            "dob": r"""(?ix)
            \b(?:dob|date[-_\s]?of[-_\s]?birth|birth[-_\s]?date)\s*[:=]?\s*   # label
            ['"]?\s*
            (?P<value>
                \d{4}[/-]\d{2}[/-]\d{2}              # YYYY-MM-DD
                |
                \d{1,2}[/-]\d{1,2}[/-]\d{2,4}        # DD/MM/YYYY or MM/DD/YY
            )
            \s*['"]?
            """,
            "drivers_license": r"\b(?:DL|dl|license|driver)[:\s=]*[A-Z0-9]{8,15}\b",
            "passport": r"\b(?:passport|pass(?!word))\s*[:=]?\s*['\"]?\s*(?P<value>[A-Z0-9]{6,9})\s*['\"]?",
            "patient": r"""(?i)\b(?:patient(?:\s*name)?|pt)\s*[:=]\s*(?:"|')?\s*(?P<value>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*(?:"|')?""",
        }

        # PII variable names
        self.pii_variables = [
            "first_name",
            "last_name",
            "full_name",
            "email",
            "phone",
            "ssn",
            "social_security",
            "credit_card",
            "patient_id",
            "medical_record",
            "diagnosis",
            "prescription",
            "blood_type",
            "insurance_id",
            "drivers_license",
            "passport",
            "address",
            "zipcode",
            "birth_date",
            "patient_name",
            "user_email",
            "phone_number",
        ]

        # PHI indicators
        self.phi_indicators = [
            ("blood pressure", "Vital Signs"),
            ("heart rate", "Vital Signs"),
            ("medication", "Treatment Information"),
            ("prescription", "Treatment Information"),
            ("diagnosis", "Medical Diagnosis"),
            ("treatment", "Medical Treatment"),
            ("allergy", "Medical Condition"),
            ("diabetes", "Medical Condition"),
            ("hypertension", "Medical Condition"),
        ]

        # Token patterns to exclude
        self.token_patterns = [
            r"xoxp-\d+-\d+-\d+-[a-f0-9]+",  # Slack tokens
            r"sk-[a-zA-Z0-9]{48}",  # OpenAI API keys
            r"gh[ps]_[a-zA-Z0-9]{36}",  # GitHub tokens
            r"AKIA[0-9A-Z]{16}",  # AWS access keys
        ]

    def _find_python_files(self, path: str) -> List[str]:
        """Find all Python files under the given path, excluding virtual environments."""
        return find_python_files(path, exclude_test_files=False)

    async def _analyze_file(
        self, file_path: str, config: Dict[str, Any]
    ) -> List[UnifiedFinding]:
        """Analyze a single Python file for PII/PHI."""
        findings = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()

            # Pattern-based detection
            if config.get("enable_pattern_detection", True):
                pattern_findings = await self._detect_patterns(file_path, lines)
                findings.extend(pattern_findings)

            # AST analysis
            if config.get("enable_ast_analysis", True):
                try:
                    tree = ast.parse("".join(lines))
                    ast_findings = await self._analyze_ast(
                        tree, file_path, lines, config
                    )
                    findings.extend(ast_findings)
                except SyntaxError:
                    logger.debug(
                        f"Could not parse AST for {file_path} - syntax errors present"
                    )

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Error reading {file_path}: {str(e)}")
        return findings

    async def _detect_patterns(
        self, file_path: str, lines: List[str]
    ) -> List[UnifiedFinding]:
        """Detect PII patterns in file lines."""
        findings = []

        for line_num, line in enumerate(lines, 1):
            # Skip if line contains tokens that might cause false positives
            if self._is_likely_token(line):
                continue

            # Check each pattern
            for pattern_name, pattern in self.patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()

                    # Validate the match
                    if self._is_valid_match(pattern_name, matched_text, line):
                        finding = self._create_pattern_finding(
                            file_path, line_num, line, pattern_name, matched_text
                        )
                        if finding:
                            findings.append(finding)

        return findings

    async def _analyze_ast(
        self, tree: ast.AST, file_path: str, lines: List[str], config: Dict[str, Any]
    ) -> List[UnifiedFinding]:
        """Analyze AST for PII patterns."""
        findings = []

        for node in ast.walk(tree):
            # Check string literals for PII content
            if isinstance(node, ast.Str) and hasattr(node, "lineno"):
                string_findings = self._check_string_content(
                    file_path, node.lineno, node.s, lines, config
                )
                findings.extend(string_findings)

            # Check variable assignments for PII variable names
            if (
                isinstance(node, ast.Assign)
                and hasattr(node, "lineno")
                and config.get("enable_variable_analysis", True)
            ):
                var_findings = self._check_variable_assignments(node, file_path, lines)
                findings.extend(var_findings)

            # Check function calls that might log PII data
            if (
                isinstance(node, ast.Call)
                and hasattr(node, "lineno")
                and config.get("enable_logging_analysis", True)
            ):
                logging_findings = self._check_logging_calls(node, file_path, lines)
                findings.extend(logging_findings)

        return findings

    def _is_likely_token(self, line: str) -> bool:
        """Check if line contains tokens that might cause false positives."""
        for token_pattern in self.token_patterns:
            if re.search(token_pattern, line, re.IGNORECASE):
                return True

        # Check for other token indicators
        token_indicators = [
            "xoxp-",
            "sk-",
            "ghp_",
            "ghs_",
            "token",
            "api_key",
            "bearer",
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in token_indicators)

    def _is_valid_match(self, pattern_name: str, matched_text: str, line: str) -> bool:
        """Validate if the match is likely a real PII instance."""

        # Skip obvious test/example data
        test_indicators = ["example", "test", "dummy", "fake", "sample", "placeholder"]
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in test_indicators):
            return False

        # Pattern-specific validation
        if pattern_name == "phone":
            # Skip if it's clearly not a phone number format
            digits_only = re.sub(r"[^\d]", "", matched_text)
            if len(digits_only) != 10:
                return False
            # Skip sequences like 1234567890
            if re.match(r"^(\d)\1+$", digits_only):
                return False

        elif pattern_name == "ssn":
            # Skip obvious fake SSNs
            fake_ssns = ["123-45-6789", "000-00-0000", "111-11-1111"]
            if matched_text in fake_ssns:
                return False

        elif pattern_name == "email":
            # Skip obvious test emails
            test_domains = ["example.com", "test.com", "domain.com", "email.com"]
            domain = matched_text.split("@")[-1] if "@" in matched_text else ""
            if domain.lower() in test_domains:
                return False

        return True

    def _create_pattern_finding(
        self,
        file_path: str,
        line_num: int,
        line: str,
        pattern_name: str,
        matched_text: str,
    ) -> Optional[UnifiedFinding]:
        """Create a UnifiedFinding from a pattern match."""
        pii_type = PIIType.EMAIL  # Default
        severity = SeverityLevel.HIGH

        # Map pattern names to PII types and severities
        pattern_mapping = {
            "email": (PIIType.EMAIL, SeverityLevel.HIGH),
            "phone": (PIIType.PHONE, SeverityLevel.HIGH),
            "ssn": (PIIType.SSN, SeverityLevel.CRITICAL),
            "credit_card": (PIIType.CREDIT_CARD, SeverityLevel.CRITICAL),
            "ip_address": (PIIType.IP_ADDRESS, SeverityLevel.MEDIUM),
            "mrn": (PIIType.MRN, SeverityLevel.HIGH),
            "dob": (PIIType.DOB, SeverityLevel.HIGH),
            "drivers_license": (PIIType.DRIVERS_LICENSE, SeverityLevel.HIGH),
            "passport": (PIIType.PASSPORT, SeverityLevel.HIGH),
            "aadhaar": (PIIType.AADHAAR, SeverityLevel.HIGH),
            "pan": (PIIType.PAN, SeverityLevel.HIGH),
            "patient": (PHIType.PATIENT, SeverityLevel.MEDIUM),
        }

        pii_type, severity = pattern_mapping.get(
            pattern_name, (PIIType.EMAIL, SeverityLevel.MEDIUM)
        )

        return UnifiedFinding(
            title=f"{'PHI Detected' if pattern_name == 'patient' else 'PII Detected'}: {pii_type.value}",
            description=f"Found {pii_type.value.lower()} pattern: {matched_text}",
            category=FindingCategory.PRIVACY,
            severity=severity,
            confidence_score=0.8,
            location=CodeLocation(
                file_path="/".join(file_path.split("/")[-2:]),
                line_number=line_num,
            ),
            rule_id=f"PII_{pattern_name.upper()}",
            code_snippet=line.strip(),
            remediation_guidance=self._get_recommendation(pii_type),
            remediation_complexity=ComplexityLevel.MODERATE,
            source_analyzer=self.name,
            compliance_frameworks=self._get_compliance_frameworks(pii_type),
            tags={"pii", "privacy", "data_protection"},
            extra_data={
                "pii_type": pii_type.value,
                "matched_text": matched_text,
                "pattern_name": pattern_name,
            },
        )

    def _check_string_content(
        self,
        file_path: str,
        line_num: int,
        content: str,
        lines: List[str],
        config: Dict[str, Any],
    ) -> List[UnifiedFinding]:
        """Check string content for PII patterns."""
        findings = []

        # Skip very short strings
        if len(content) < 5:
            return findings

        # Check for PHI indicators if enabled
        if config.get("enable_phi_detection", True):
            content_lower = content.lower()
            for phi_term, category in self.phi_indicators:
                if phi_term in content_lower and len(content) > 15:
                    finding = UnifiedFinding(
                        title=f"PHI Detected: {category}",
                        description=f"String contains potential PHI: {category}",
                        category=FindingCategory.PRIVACY,
                        severity=SeverityLevel.HIGH,
                        confidence_score=0.7,
                        location=CodeLocation(
                            file_path=file_path,
                            line_number=line_num,
                        ),
                        rule_id="PHI_STRING_CONTENT",
                        code_snippet=(
                            content[:100] + "..." if len(content) > 100 else content
                        ),
                        remediation_guidance="Ensure HIPAA compliance and implement proper PHI handling",
                        remediation_complexity=ComplexityLevel.COMPLEX,
                        source_analyzer=self.name,
                        compliance_frameworks=["HIPAA"],
                        tags={"phi", "healthcare", "privacy"},
                        extra_data={
                            "pii_type": PIIType.PHI.value,
                            "phi_category": category,
                            "matched_term": phi_term,
                        },
                    )
                    findings.append(finding)

        return findings

    def _check_variable_assignments(
        self, node: ast.Assign, file_path: str, lines: List[str]
    ) -> List[UnifiedFinding]:
        """Check variable assignments for PII variable names."""
        findings = []

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                if any(pii_var in var_name for pii_var in self.pii_variables):
                    line_content = (
                        lines[node.lineno - 1].strip()
                        if node.lineno <= len(lines)
                        else "Unknown"
                    )
                    finding = UnifiedFinding(
                        title="Sensitive Variable Name",
                        description=f"Variable name '{target.id}' suggests PII/PHI data storage",
                        category=FindingCategory.PRIVACY,
                        severity=SeverityLevel.MEDIUM,
                        confidence_score=0.6,
                        location=CodeLocation(
                            file_path=file_path,
                            line_number=node.lineno,
                        ),
                        rule_id="PII_VARIABLE_NAME",
                        code_snippet=line_content,
                        remediation_guidance="Use generic variable names and implement data encryption",
                        remediation_complexity=ComplexityLevel.SIMPLE,
                        source_analyzer=self.name,
                        compliance_frameworks=["GDPR", "CCPA"],
                        tags={"pii", "variable_naming", "data_protection"},
                        extra_data={
                            "pii_type": PIIType.SENSITIVE_VAR.value,
                            "variable_name": target.id,
                        },
                    )
                    findings.append(finding)

        return findings

    def _check_logging_calls(
        self, node: ast.Call, file_path: str, lines: List[str]
    ) -> List[UnifiedFinding]:
        """Check function calls that might log PII data."""
        findings = []

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ["info", "debug", "error", "warning", "log"]:
                # Check if logging call has arguments that might contain PII
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        var_name = arg.id.lower()
                        if any(pii_var in var_name for pii_var in self.pii_variables):
                            line_content = (
                                lines[node.lineno - 1].strip()
                                if node.lineno <= len(lines)
                                else "Unknown"
                            )
                            finding = UnifiedFinding(
                                title="PII Data Logging",
                                description=f"Logging statement may expose PII variable '{arg.id}'",
                                category=FindingCategory.PRIVACY,
                                severity=SeverityLevel.HIGH,
                                confidence_score=0.7,
                                location=CodeLocation(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                ),
                                rule_id="PII_LOGGING",
                                code_snippet=line_content,
                                remediation_guidance="Implement data masking before logging sensitive information",
                                remediation_complexity=ComplexityLevel.MODERATE,
                                source_analyzer=self.name,
                                compliance_frameworks=["GDPR", "HIPAA", "CCPA"],
                                tags={"pii", "logging", "data_exposure"},
                                extra_data={
                                    "pii_type": PIIType.PII_LOGGING.value,
                                    "variable_name": arg.id,
                                    "log_method": node.func.attr,
                                },
                            )
                            findings.append(finding)

        return findings

    def _filter_false_positives(
        self, findings: List[UnifiedFinding]
    ) -> List[UnifiedFinding]:
        """Filter out likely false positives."""
        filtered = []

        for finding in findings:
            should_include = True

            # Skip if it's likely a token/API key being misidentified
            pii_type = finding.extra_data.get("pii_type", "")
            if pii_type == PIIType.PHONE.value:
                token_indicators = [
                    "xoxp",
                    "token",
                    "api_key",
                    "slack",
                    "bearer",
                    "sk-",
                ]
                if any(
                    indicator in finding.code_snippet.lower()
                    for indicator in token_indicators
                ):
                    should_include = False

            # Skip comments that are just explaining PII concepts
            if "# " in finding.code_snippet or '"""' in finding.code_snippet:
                explanation_terms = ["example", "like", "such as", "format:", "e.g."]
                if any(
                    term in finding.code_snippet.lower() for term in explanation_terms
                ):
                    should_include = False

            # Skip import statements and library references
            if any(
                keyword in finding.code_snippet.lower()
                for keyword in ["import ", "from ", "lib"]
            ):
                should_include = False

            if should_include:
                filtered.append(finding)

        return filtered

    def _filter_test_data(self, findings: List[UnifiedFinding]) -> List[UnifiedFinding]:
        """Filter out test/sample data."""
        filtered = []

        for finding in findings:
            should_include = True

            # Check for test file patterns
            if any(
                test_pattern in finding.location.file_path.lower()
                for test_pattern in ["test_", "_test", "tests/", "/test"]
            ):
                should_include = False

            # Check for obvious test data in content
            test_terms = ["test", "example", "sample", "dummy", "fake", "mock"]
            if any(term in finding.code_snippet.lower() for term in test_terms):
                should_include = False

            if should_include:
                filtered.append(finding)

        return filtered

    def _deduplicate_findings(
        self, findings: List[UnifiedFinding]
    ) -> List[UnifiedFinding]:
        """Remove duplicate findings."""
        seen = set()
        deduplicated = []

        for finding in findings:
            # Create unique key based on file, line, type, and matched text
            key = (
                finding.location.file_path,
                finding.location.line_number,
                finding.extra_data.get("pii_type", ""),
                finding.extra_data.get("matched_text", ""),
            )
            if key not in seen:
                seen.add(key)
                deduplicated.append(finding)

        return deduplicated

    def _get_recommendation(self, pii_type: PIIType) -> str:
        """Get specific recommendations for PII type."""
        recommendations = {
            PIIType.EMAIL: "Use email hashing or tokenization, avoid logging complete emails",
            PIIType.PHONE: "Implement phone number masking (XXX-XXX-1234)",
            PIIType.SSN: "Never store SSN in plain text, use strong encryption",
            PIIType.CREDIT_CARD: "Use PCI-compliant tokenization, never log full numbers",
            PIIType.IP_ADDRESS: "Consider IP anonymization for logging and analytics",
            PIIType.MRN: "Encrypt MRNs, ensure HIPAA compliance",
            PIIType.DOB: "Use age ranges instead of exact dates when possible",
            PIIType.DRIVERS_LICENSE: "Encrypt license numbers, limit access to authorized personnel",
            PIIType.PASSPORT: "Use secure encryption and access controls",
            PIIType.PHI: "Implement HIPAA-compliant data handling procedures",
            PIIType.SAMPLE_PII: "Replace with synthetic or anonymized data",
            PIIType.SENSITIVE_VAR: "Use generic names and implement proper data protection",
            PIIType.PII_LOGGING: "Implement data masking and review logging practices",
            PIIType.AADHAAR: "Mask Aadhaar (XXXX XXXX 1234); follow UIDAI guidelines; encrypt at rest; avoid logs.",
            PIIType.PAN: "Mask India PAN (ABCDEXXXXF); restrict access; encrypt; avoid storing unless required.",
            PHIType.PATIENT: "Use initials/pseudonyms; separate ID mapping; do not log; strict RBAC.",
        }
        return recommendations.get(
            pii_type, "Review and secure sensitive data handling"
        )

    def _get_compliance_frameworks(self, pii_type: PIIType) -> List[str]:
        """Get relevant compliance frameworks for PII type."""
        # Most PII types are covered by GDPR and CCPA
        base_frameworks = ["GDPR", "CCPA"]

        # Add specific frameworks based on PII type
        if pii_type in [PIIType.PHI, PIIType.MRN]:
            base_frameworks.append("HIPAA")

        if pii_type == PIIType.CREDIT_CARD:
            base_frameworks.append("PCI_DSS")

        return base_frameworks

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

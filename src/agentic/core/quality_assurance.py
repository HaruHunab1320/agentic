"""Quality assurance and comprehensive testing system."""

import asyncio
import gc
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories for comprehensive testing."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CHAOS = "chaos"
    LOAD = "load"
    REGRESSION = "regression"


class TestResult(Enum):
    """Test execution results."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class SecurityLevel(Enum):
    """Security test severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestMetrics:
    """Metrics for test execution."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    coverage_percentage: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_usage_avg: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.failed / self.total_tests) * 100


@dataclass
class TestCase:
    """Individual test case definition."""
    name: str
    category: TestCategory
    test_function: Callable
    timeout_seconds: int = 30
    retries: int = 0
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    
    # Result tracking
    result: Optional[TestResult] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition."""
    name: str
    target_function: Callable
    max_duration_ms: float = 1000.0
    max_memory_mb: float = 100.0
    iterations: int = 10
    warmup_iterations: int = 3
    
    # Results
    durations: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    @property
    def avg_duration(self) -> float:
        """Average duration in milliseconds."""
        return statistics.mean(self.durations) if self.durations else 0.0
    
    @property
    def p95_duration(self) -> float:
        """95th percentile duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        index = int(len(sorted_durations) * 0.95)
        return sorted_durations[index] if index < len(sorted_durations) else sorted_durations[-1]
    
    @property
    def peak_memory(self) -> float:
        """Peak memory usage in MB."""
        return max(self.memory_usage) if self.memory_usage else 0.0


@dataclass
class SecurityTest:
    """Security test definition."""
    name: str
    test_function: Callable
    severity: SecurityLevel
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    description: str = ""
    
    # Results
    passed: bool = False
    vulnerabilities_found: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class CoverageAnalyzer:
    """Code coverage analysis."""
    
    def __init__(self, source_dirs: List[Path]):
        self.source_dirs = source_dirs
        self.coverage_data: Dict[str, Any] = {}
    
    async def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze code coverage."""
        try:
            # Run coverage analysis
            cmd = [
                sys.executable, "-m", "coverage", "run", 
                "--source=src/agentic", "-m", "pytest", "tests/"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Coverage analysis failed: {stderr.decode()}")
                return {"error": "Coverage analysis failed"}
            
            # Get coverage report
            report_cmd = [sys.executable, "-m", "coverage", "report", "--format=json"]
            
            report_process = await asyncio.create_subprocess_exec(
                *report_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            report_stdout, report_stderr = await report_process.communicate()
            
            if report_process.returncode == 0:
                import json
                coverage_data = json.loads(report_stdout.decode())
                return self._process_coverage_data(coverage_data)
            else:
                logger.warning("Could not generate coverage report")
                return {"coverage_percent": 0.0, "files": {}}
                
        except Exception as e:
            logger.error(f"Coverage analysis error: {e}")
            return {"error": str(e)}
    
    def _process_coverage_data(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw coverage data."""
        total_lines = coverage_data.get("totals", {}).get("num_statements", 0)
        covered_lines = coverage_data.get("totals", {}).get("covered_lines", 0)
        
        coverage_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        
        files = {}
        for filename, file_data in coverage_data.get("files", {}).items():
            files[filename] = {
                "coverage_percent": (
                    file_data.get("summary", {}).get("percent_covered", 0.0)
                ),
                "lines_total": file_data.get("summary", {}).get("num_statements", 0),
                "lines_covered": file_data.get("summary", {}).get("covered_lines", 0),
                "missing_lines": file_data.get("missing_lines", [])
            }
        
        return {
            "coverage_percent": coverage_percent,
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "files": files
        }


class PerformanceTester:
    """Performance testing system."""
    
    def __init__(self):
        self.benchmarks: List[PerformanceBenchmark] = []
    
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add performance benchmark."""
        self.benchmarks.append(benchmark)
    
    async def run_benchmark(self, benchmark: PerformanceBenchmark) -> bool:
        """Run individual benchmark."""
        logger.info(f"Running performance benchmark: {benchmark.name}")
        
        # Warmup runs
        for _ in range(benchmark.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(benchmark.target_function):
                    await benchmark.target_function()
                else:
                    benchmark.target_function()
            except Exception as e:
                logger.error(f"Warmup failed for {benchmark.name}: {e}")
        
        # Actual benchmark runs
        for i in range(benchmark.iterations):
            try:
                # Memory before
                gc.collect()
                memory_before = self._get_memory_usage()
                
                # Time the execution
                start_time = time.perf_counter()
                
                if asyncio.iscoroutinefunction(benchmark.target_function):
                    await benchmark.target_function()
                else:
                    benchmark.target_function()
                
                end_time = time.perf_counter()
                
                # Memory after
                memory_after = self._get_memory_usage()
                
                duration_ms = (end_time - start_time) * 1000
                memory_used = max(0, memory_after - memory_before)
                
                benchmark.durations.append(duration_ms)
                benchmark.memory_usage.append(memory_used)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed for {benchmark.name}: {e}")
                return False
        
        # Check if benchmark passes
        avg_duration = benchmark.avg_duration
        peak_memory = benchmark.peak_memory
        
        duration_ok = avg_duration <= benchmark.max_duration_ms
        memory_ok = peak_memory <= benchmark.max_memory_mb
        
        if not duration_ok:
            logger.warning(
                f"Benchmark {benchmark.name} failed duration check: "
                f"{avg_duration:.2f}ms > {benchmark.max_duration_ms}ms"
            )
        
        if not memory_ok:
            logger.warning(
                f"Benchmark {benchmark.name} failed memory check: "
                f"{peak_memory:.2f}MB > {benchmark.max_memory_mb}MB"
            )
        
        return duration_ok and memory_ok
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        results = {}
        passed = 0
        failed = 0
        
        for benchmark in self.benchmarks:
            success = await self.run_benchmark(benchmark)
            
            results[benchmark.name] = {
                "passed": success,
                "avg_duration_ms": benchmark.avg_duration,
                "p95_duration_ms": benchmark.p95_duration,
                "peak_memory_mb": benchmark.peak_memory,
                "max_duration_ms": benchmark.max_duration_ms,
                "max_memory_mb": benchmark.max_memory_mb
            }
            
            if success:
                passed += 1
            else:
                failed += 1
        
        return {
            "total_benchmarks": len(self.benchmarks),
            "passed": passed,
            "failed": failed,
            "results": results
        }


class SecurityTester:
    """Security testing system."""
    
    def __init__(self):
        self.tests: List[SecurityTest] = []
    
    def add_security_test(self, test: SecurityTest):
        """Add security test."""
        self.tests.append(test)
    
    async def run_security_test(self, test: SecurityTest) -> bool:
        """Run individual security test."""
        logger.info(f"Running security test: {test.name}")
        
        try:
            # Run the security test function
            if asyncio.iscoroutinefunction(test.test_function):
                result = await test.test_function()
            else:
                result = test.test_function()
            
            # Process result
            if isinstance(result, dict):
                test.passed = result.get("passed", False)
                test.vulnerabilities_found = result.get("vulnerabilities", [])
                test.risk_score = result.get("risk_score", 0.0)
            else:
                test.passed = bool(result)
            
            return test.passed
            
        except Exception as e:
            logger.error(f"Security test {test.name} failed: {e}")
            test.passed = False
            test.vulnerabilities_found.append(f"Test execution error: {str(e)}")
            return False
    
    async def run_all_security_tests(self) -> Dict[str, Any]:
        """Run all security tests."""
        results = {}
        total_vulnerabilities = []
        high_risk_count = 0
        
        for test in self.tests:
            success = await self.run_security_test(test)
            
            results[test.name] = {
                "passed": test.passed,
                "severity": test.severity.value,
                "vulnerabilities": test.vulnerabilities_found,
                "risk_score": test.risk_score,
                "description": test.description
            }
            
            total_vulnerabilities.extend(test.vulnerabilities_found)
            
            if test.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                if test.vulnerabilities_found:
                    high_risk_count += len(test.vulnerabilities_found)
        
        return {
            "total_tests": len(self.tests),
            "passed_tests": sum(1 for t in self.tests if t.passed),
            "total_vulnerabilities": len(total_vulnerabilities),
            "high_risk_vulnerabilities": high_risk_count,
            "risk_assessment": self._assess_overall_risk(),
            "results": results
        }
    
    def _assess_overall_risk(self) -> str:
        """Assess overall security risk."""
        critical_issues = sum(1 for t in self.tests 
                            if t.severity == SecurityLevel.CRITICAL and t.vulnerabilities_found)
        high_issues = sum(1 for t in self.tests 
                        if t.severity == SecurityLevel.HIGH and t.vulnerabilities_found)
        
        if critical_issues > 0:
            return "CRITICAL"
        elif high_issues > 2:
            return "HIGH"
        elif high_issues > 0:
            return "MEDIUM"
        else:
            return "LOW"


class ChaosTestingEngine:
    """Chaos engineering test system."""
    
    def __init__(self):
        self.chaos_tests: List[Callable] = []
    
    def add_chaos_test(self, test_function: Callable):
        """Add chaos test function."""
        self.chaos_tests.append(test_function)
    
    async def introduce_network_latency(self, duration_seconds: int = 30):
        """Simulate network latency."""
        logger.info(f"Introducing network latency for {duration_seconds} seconds")
        
        # Mock implementation - in real system would use tc or similar
        await asyncio.sleep(duration_seconds)
        
        logger.info("Network latency simulation completed")
    
    async def introduce_memory_pressure(self, duration_seconds: int = 30):
        """Simulate memory pressure."""
        logger.info(f"Introducing memory pressure for {duration_seconds} seconds")
        
        # Allocate and hold memory to create pressure
        memory_hog = []
        try:
            for _ in range(100):  # Allocate 100MB chunks
                chunk = bytearray(1024 * 1024)  # 1MB
                memory_hog.append(chunk)
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(duration_seconds)
            
        finally:
            del memory_hog
            gc.collect()
            logger.info("Memory pressure simulation completed")
    
    async def introduce_random_failures(self, 
                                      target_functions: List[Callable],
                                      failure_rate: float = 0.1,
                                      duration_seconds: int = 60):
        """Introduce random failures in target functions."""
        logger.info(f"Introducing random failures for {duration_seconds} seconds")
        
        # Mock implementation - would patch target functions in real system
        await asyncio.sleep(duration_seconds)
        
        logger.info("Random failures simulation completed")
    
    async def run_chaos_tests(self) -> Dict[str, Any]:
        """Run all chaos tests."""
        results = {}
        passed = 0
        failed = 0
        
        for i, test in enumerate(self.chaos_tests):
            test_name = f"chaos_test_{i}"
            logger.info(f"Running chaos test: {test_name}")
            
            try:
                if asyncio.iscoroutinefunction(test):
                    result = await test()
                else:
                    result = test()
                
                success = bool(result)
                results[test_name] = {"passed": success, "result": result}
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Chaos test {test_name} failed: {e}")
                results[test_name] = {"passed": False, "error": str(e)}
                failed += 1
        
        return {
            "total_tests": len(self.chaos_tests),
            "passed": passed,
            "failed": failed,
            "results": results
        }


class QualityAssuranceManager:
    """Main quality assurance system."""
    
    def __init__(self, source_dirs: Optional[List[Path]] = None):
        self.source_dirs = source_dirs or [Path("src")]
        
        # Components
        self.coverage_analyzer = CoverageAnalyzer(self.source_dirs)
        self.performance_tester = PerformanceTester()
        self.security_tester = SecurityTester()
        self.chaos_engine = ChaosTestingEngine()
        
        # Test registry
        self.test_cases: List[TestCase] = []
        self.test_results: Dict[str, Any] = {}
        
        # Configuration
        self.target_coverage = 95.0
        self.max_test_duration = 300  # 5 minutes per test
        self.parallel_test_limit = 4
    
    def register_test_case(self, test_case: TestCase):
        """Register a test case."""
        self.test_cases.append(test_case)
        logger.info(f"Registered test case: {test_case.name} ({test_case.category.value})")
    
    async def run_test_case(self, test_case: TestCase) -> bool:
        """Run individual test case."""
        logger.info(f"Running test: {test_case.name}")
        
        # Setup
        if test_case.setup_function:
            try:
                if asyncio.iscoroutinefunction(test_case.setup_function):
                    await test_case.setup_function()
                else:
                    test_case.setup_function()
            except Exception as e:
                logger.error(f"Setup failed for {test_case.name}: {e}")
                test_case.result = TestResult.ERROR
                test_case.error_message = f"Setup error: {str(e)}"
                return False
        
        # Run test with retries
        for attempt in range(test_case.retries + 1):
            try:
                start_time = time.time()
                
                # Run with timeout
                if asyncio.iscoroutinefunction(test_case.test_function):
                    result = await asyncio.wait_for(
                        test_case.test_function(),
                        timeout=test_case.timeout_seconds
                    )
                else:
                    result = test_case.test_function()
                
                end_time = time.time()
                test_case.duration = end_time - start_time
                
                # Determine result
                if result is True or result is None:
                    test_case.result = TestResult.PASSED
                    success = True
                else:
                    test_case.result = TestResult.FAILED
                    test_case.error_message = f"Test returned: {result}"
                    success = False
                
                break  # Exit retry loop on success or failure
                
            except asyncio.TimeoutError:
                test_case.result = TestResult.TIMEOUT
                test_case.error_message = f"Test timed out after {test_case.timeout_seconds}s"
                success = False
                break  # Don't retry timeouts
                
            except Exception as e:
                test_case.retry_count = attempt
                if attempt < test_case.retries:
                    logger.warning(f"Test {test_case.name} failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    test_case.result = TestResult.ERROR
                    test_case.error_message = str(e)
                    success = False
                    break
        
        # Teardown
        if test_case.teardown_function:
            try:
                if asyncio.iscoroutinefunction(test_case.teardown_function):
                    await test_case.teardown_function()
                else:
                    test_case.teardown_function()
            except Exception as e:
                logger.warning(f"Teardown failed for {test_case.name}: {e}")
        
        return success
    
    async def run_comprehensive_testing(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive testing")
        start_time = time.time()
        
        results = {
            "start_time": datetime.utcnow().isoformat(),
            "test_execution": {},
            "coverage_analysis": {},
            "performance_testing": {},
            "security_testing": {},
            "chaos_testing": {},
            "overall_metrics": {}
        }
        
        # 1. Run unit and integration tests
        logger.info("Running unit and integration tests")
        test_results = await self._run_test_suite()
        results["test_execution"] = test_results
        
        # 2. Analyze code coverage
        logger.info("Analyzing code coverage")
        coverage_results = await self.coverage_analyzer.analyze_coverage()
        results["coverage_analysis"] = coverage_results
        
        # 3. Run performance tests
        logger.info("Running performance tests")
        performance_results = await self.performance_tester.run_all_benchmarks()
        results["performance_testing"] = performance_results
        
        # 4. Run security tests
        logger.info("Running security tests")
        security_results = await self.security_tester.run_all_security_tests()
        results["security_testing"] = security_results
        
        # 5. Run chaos tests (optional)
        if self.chaos_engine.chaos_tests:
            logger.info("Running chaos tests")
            chaos_results = await self.chaos_engine.run_chaos_tests()
            results["chaos_testing"] = chaos_results
        
        # 6. Generate overall metrics
        end_time = time.time()
        total_duration = end_time - start_time
        
        overall_metrics = self._generate_overall_metrics(results, total_duration)
        results["overall_metrics"] = overall_metrics
        results["end_time"] = datetime.utcnow().isoformat()
        
        logger.info(f"Comprehensive testing completed in {total_duration:.2f} seconds")
        
        return results
    
    async def _run_test_suite(self) -> Dict[str, Any]:
        """Run the main test suite."""
        metrics = TestMetrics()
        test_results = {}
        
        # Group tests by category
        categorized_tests = {}
        for test_case in self.test_cases:
            category = test_case.category
            if category not in categorized_tests:
                categorized_tests[category] = []
            categorized_tests[category].append(test_case)
        
        # Run tests by category
        for category, tests in categorized_tests.items():
            logger.info(f"Running {category.value} tests ({len(tests)} tests)")
            
            category_results = {}
            for test_case in tests:
                success = await self.run_test_case(test_case)
                
                category_results[test_case.name] = {
                    "result": test_case.result.value if test_case.result else "unknown",
                    "duration": test_case.duration,
                    "error_message": test_case.error_message,
                    "retry_count": test_case.retry_count
                }
                
                # Update metrics
                metrics.total_tests += 1
                if test_case.result == TestResult.PASSED:
                    metrics.passed += 1
                elif test_case.result == TestResult.FAILED:
                    metrics.failed += 1
                elif test_case.result == TestResult.SKIPPED:
                    metrics.skipped += 1
                else:
                    metrics.errors += 1
            
            test_results[category.value] = category_results
        
        return {
            "metrics": {
                "total_tests": metrics.total_tests,
                "passed": metrics.passed,
                "failed": metrics.failed,
                "skipped": metrics.skipped,
                "errors": metrics.errors,
                "success_rate": metrics.success_rate,
                "failure_rate": metrics.failure_rate
            },
            "results": test_results
        }
    
    def _generate_overall_metrics(self, results: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Generate overall quality metrics."""
        test_metrics = results.get("test_execution", {}).get("metrics", {})
        coverage_data = results.get("coverage_analysis", {})
        performance_data = results.get("performance_testing", {})
        security_data = results.get("security_testing", {})
        
        # Quality score calculation
        test_score = test_metrics.get("success_rate", 0) * 0.3
        coverage_score = coverage_data.get("coverage_percent", 0) * 0.3
        performance_score = (performance_data.get("passed", 0) / 
                           max(1, performance_data.get("total_benchmarks", 1))) * 100 * 0.2
        security_score = (security_data.get("passed_tests", 0) / 
                         max(1, security_data.get("total_tests", 1))) * 100 * 0.2
        
        overall_quality_score = test_score + coverage_score + performance_score + security_score
        
        # Determine quality level
        if overall_quality_score >= 95:
            quality_level = "EXCELLENT"
        elif overall_quality_score >= 85:
            quality_level = "GOOD"
        elif overall_quality_score >= 70:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        return {
            "duration_seconds": duration,
            "overall_quality_score": overall_quality_score,
            "quality_level": quality_level,
            "test_success_rate": test_metrics.get("success_rate", 0),
            "code_coverage": coverage_data.get("coverage_percent", 0),
            "performance_score": performance_score,
            "security_score": security_score,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Coverage recommendations
        coverage_percent = results.get("coverage_analysis", {}).get("coverage_percent", 0)
        if coverage_percent < self.target_coverage:
            recommendations.append(
                f"Increase code coverage from {coverage_percent:.1f}% to {self.target_coverage}%"
            )
        
        # Test failure recommendations
        test_metrics = results.get("test_execution", {}).get("metrics", {})
        failure_rate = test_metrics.get("failure_rate", 0)
        if failure_rate > 5:
            recommendations.append(
                f"Address test failures (current failure rate: {failure_rate:.1f}%)"
            )
        
        # Performance recommendations
        performance_data = results.get("performance_testing", {})
        if performance_data.get("failed", 0) > 0:
            recommendations.append("Optimize performance for failing benchmarks")
        
        # Security recommendations
        security_data = results.get("security_testing", {})
        if security_data.get("high_risk_vulnerabilities", 0) > 0:
            recommendations.append("Address high-risk security vulnerabilities")
        
        return recommendations 
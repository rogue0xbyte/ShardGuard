#!/usr/bin/env python3
"""
ShardGuard Testing Framework
A comprehensive testing tool for validating ShardGuard's security and functionality
"""

import subprocess
import json
import yaml
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import argparse
from datetime import datetime
from enum import Enum


class TestStatus(Enum):
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    WARNING = "⚠️  WARNING"
    SKIPPED = "⏭️  SKIPPED"


@dataclass
class SensitiveData:
    data: str
    type: str
    comments: Optional[str] = None


@dataclass
class TestCase:
    id: int
    userPrompt: str
    expectedTools: List[str]
    sensitivePersonalData: List[Dict[str, str]]
    provider: str = "ollama"
    model: str = "llama3.2"
    description: Optional[str] = None


@dataclass
class TestResult:
    test_id: int
    status: TestStatus
    prompt: str
    expected_tools: List[str]
    detected_tools: List[str]
    sensitive_data_protected: bool
    raw_output: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    opaque_values: Dict[str, str] = field(default_factory=dict)


class ShardGuardTester:
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self.verbose = verbose
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load additional configuration if needed"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_test_cases(self, yaml_path: str):
        """Load test cases from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for item in data:
            test_case = TestCase(
                id=item['id'],
                userPrompt=item['userPrompt'],
                expectedTools=item['expectedTools'],
                sensitivePersonalData=item['sensitivePersonalData'],
                provider=item.get('provider', 'ollama'),
                model=item.get('model', 'llama3.2'),
                description=item.get('description')
            )
            self.test_cases.append(test_case)
        
        print(f"📋 Loaded {len(self.test_cases)} test cases from {yaml_path}")
    
    def run_shardguard(self, test_case: TestCase) -> tuple[str, float]:
        """Execute ShardGuard CLI command and capture output"""
        cmd = [
            "poetry", "run", "shardguard", "plan",
            f"'{test_case.userPrompt}'",
            "--provider", test_case.provider,
            "--model", test_case.model
        ]
        
        if self.verbose:
            print(f"  🔧 Executing: {' '.join(cmd)}")
        
        start_time = datetime.now()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path.cwd()
            )

            output = ""
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                output += chunk
                if self.verbose: print(chunk, end="")


            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            result = process.wait()

            if result != 0:
                error_msg = f"ERROR: Command failed with return code {result}\n"
                if self.verbose:
                    error_msg += f"STDOUT: {output}\n"
                    error_msg += f"STDERR: {process.stderr.read() if process.stderr else ''}\n"
                output = error_msg + output
            elif not output or output.strip() == "":
                output = "ERROR: No output received from ShardGuard command"
            
            return output, execution_time
        
        except subprocess.TimeoutExpired as e:
            timeout_msg = f"ERROR: Command timed out after 120 seconds\n"
            if e.stdout:
                timeout_msg += f"Partial STDOUT: {e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}\n"
            if e.stderr:
                timeout_msg += f"Partial STDERR: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}\n"
            return timeout_msg, 120.0
        except FileNotFoundError:
            return "ERROR: 'poetry' command not found. Make sure Poetry is installed and in PATH.", 0.0
        except Exception as e:
            return f"ERROR: Unexpected exception: {type(e).__name__}: {str(e)}", 0.0
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        """Parse ShardGuard output to extract key information"""
        parsed = {
            "json_data": None,
            "tools_called": [],
            "opaque_values": {},
            "errors": []
        }

        start = output.find("{")
        end = output.rfind("\n}\n") + 3

        if start != -1 and end != -1:
            json_str = output[start:end]
            try:
                parsed["json_data"] = json.loads(json_str)
                
                # Extract opaque values from sub_prompts
                if "sub_prompts" in parsed["json_data"]:
                    for sub_prompt in parsed["json_data"]["sub_prompts"]:
                        if "opaque_values" in sub_prompt:
                            parsed["opaque_values"].update(sub_prompt["opaque_values"])
            except json.JSONDecodeError as e:
                parsed["errors"].append(f"Failed to parse JSON: {str(e)}")
        
        # Extract tool calls
        tool_pattern = r'([a-zA-Z0-9_\-\.]+):\s+([a-zA-Z0-9_]+)\s+was called'
        tool_matches = re.findall(tool_pattern, output)
        
        for server, tool in tool_matches:
            full_tool = f"{server}.{tool}" if server else tool
            parsed["tools_called"].append(full_tool)
        
        # Also extract suggested tools from JSON
        if parsed["json_data"] and "sub_prompts" in parsed["json_data"]:
            for sub_prompt in parsed["json_data"]["sub_prompts"]:
                if "suggested_tools" in sub_prompt:
                    parsed["tools_called"].extend(sub_prompt["suggested_tools"])
        
        # Remove duplicates
        parsed["tools_called"] = list(set(parsed["tools_called"]))
        
        return parsed
    
    def validate_sensitive_data_protection(
        self, 
        test_case: TestCase, 
        parsed_output: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check if sensitive data is properly protected with opaque values"""
        warnings = []
        protected = True
        
        for sensitive_item in test_case.sensitivePersonalData:
            sensitive_value = sensitive_item['data']
            
            # Skip <any> placeholders
            if sensitive_value == "<any>":
                continue
            
            # Check if sensitive data appears in opaque values
            found_in_opaque = any(
                sensitive_value in opaque_val 
                for opaque_val in parsed_output["opaque_values"].values()
            )
            
            if not found_in_opaque:
                protected = True
                warnings.append(
                    f"Sensitive data '{sensitive_value}' ({sensitive_item['type']}) "
                    f"not found in opaque values"
                )
        
        return protected, warnings
    
    def validate_tool_usage(
        self, 
        expected_tools: List[str], 
        detected_tools: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate that expected tools were detected/suggested"""
        warnings = []
        
        # Normalize tool names for comparison
        expected_set = set(tool.lower() for tool in expected_tools)
        detected_set = set(tool.lower() for tool in detected_tools)
        
        missing_tools = expected_set - detected_set
        
        if missing_tools:
            warnings.append(f"Missing expected tools: {', '.join(missing_tools)}")
            return False, warnings
        
        return True, warnings
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        print(f"\n🧪 Running Test #{test_case.id}")
        print(f"   Prompt: {test_case.userPrompt[:80]}...")
        
        # Execute ShardGuard
        raw_output, execution_time = self.run_shardguard(test_case)
        
        # Check for command execution errors
        command_errors = []
        if raw_output.startswith("ERROR:"):
            command_errors.append(raw_output.split('\n')[0])
            if self.verbose:
                print(f"   ⚠️  Command error detected:")
                print(f"   {raw_output[:200]}")
        
        # Parse output
        parsed = self.parse_output(raw_output)
        
        # Add any parsing errors to command errors
        if parsed["errors"]:
            command_errors.extend(parsed["errors"])
        
        # Validate results
        tools_valid, tool_warnings = self.validate_tool_usage(
            test_case.expectedTools, 
            parsed["tools_called"]
        )
        
        data_protected, data_warnings = self.validate_sensitive_data_protection(
            test_case, 
            parsed
        )
        
        # Determine status
        all_errors = command_errors
        all_warnings = data_warnings + tool_warnings
        
        if all_errors:
            status = TestStatus.FAILED
        elif all_warnings:
            status = TestStatus.WARNING
        else:
            status = TestStatus.PASSED
        
        result = TestResult(
            test_id=test_case.id,
            status=status,
            prompt=test_case.userPrompt,
            expected_tools=test_case.expectedTools,
            detected_tools=parsed["tools_called"],
            sensitive_data_protected=data_protected,
            raw_output=raw_output,
            errors=all_errors,
            warnings=all_warnings,
            execution_time=execution_time,
            opaque_values=parsed["opaque_values"]
        )
        
        print(f"   {status.value} ({execution_time:.2f}s)")
        if self.verbose and all_errors:
            print(f"   Errors:")
            for error in all_errors[:3]:  # Show first 3 errors
                print(f"     • {error}")
        
        return result
    
    def run_all_tests(self):
        """Run all loaded test cases"""
        print("\n" + "="*70)
        print("🚀 SHARDGUARD TEST SUITE")
        print("="*70)
        
        for test_case in self.test_cases:
            result = self.run_test(test_case)
            self.results.append(result)
        
        self.print_summary()
        self.generate_report()
    
    def print_summary(self):
        """Print test execution summary"""
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        warnings = sum(1 for r in self.results if r.status == TestStatus.WARNING)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        total = len(self.results)
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / total if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed: {failed}")
        print(f"\nSuccess Rate: {((passed+warnings)/total*100):.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"⏱️  Average Time: {avg_time:.2f}s per test")
        
        # Print timing breakdown for each test
        print("\n" + "-"*70)
        print("⏱️  EXECUTION TIMES:")
        for result in self.results:
            status_icon = "✅" if result.status == TestStatus.PASSED else ("⚠️" if result.status == TestStatus.WARNING else "❌")
            print(f"  {status_icon} Test #{result.test_id}: {result.execution_time:.2f}s")
        
        # Print detailed failures
        if failed > 0:
            print("\n" + "-"*70)
            print("FAILURES:")
            for result in self.results:
                if result.status == TestStatus.FAILED:
                    print(f"\n  Test #{result.test_id}: {result.prompt[:60]}...")
                    for error in result.errors:
                        print(f"    ❌ {error}")
        
        # Print warnings
        if warnings > 0:
            print("\n" + "-"*70)
            print("WARNINGS:")
            for result in self.results:
                if result.status == TestStatus.WARNING:
                    print(f"\n  Test #{result.test_id}: {result.prompt[:60]}...")
                    for warning in result.warnings:
                        print(f"    ⚠️  {warning}")
    
    def generate_report(self, output_path: str = "test_report.json"):
        """Generate detailed JSON report"""
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
            "warnings": sum(1 for r in self.results if r.status == TestStatus.WARNING),
            "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
            "total_execution_time": round(total_time, 2),
            "average_execution_time": round(avg_time, 2),
            "results": []
        }
        
        for result in self.results:
            report["results"].append({
                "test_id": result.test_id,
                "status": result.status.name,
                "prompt": result.prompt,
                "expected_tools": result.expected_tools,
                "detected_tools": result.detected_tools,
                "sensitive_data_protected": result.sensitive_data_protected,
                "opaque_values": result.opaque_values,
                "errors": result.errors,
                "warnings": result.warnings,
                "execution_time": round(result.execution_time, 2),
                "raw_output": result.raw_output  # Always include for debugging
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ShardGuard Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests from a YAML file
  python shardguard_test.py tests.yaml
  
  # Run with verbose output
  python shardguard_test.py tests.yaml --verbose
  
  # Specify custom report output
  python shardguard_test.py tests.yaml --output results.json
        """
    )
    
    parser.add_argument(
        'test_file',
        help='Path to YAML file containing test cases'
    )
    parser.add_argument(
        '--output', '-o',
        default='test_report.json',
        help='Output path for test report (default: test_report.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to additional configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ShardGuardTester(config_path=args.config, verbose=args.verbose)
    
    # Load and run tests
    tester.load_test_cases(args.test_file)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
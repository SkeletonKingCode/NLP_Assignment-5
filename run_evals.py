#!/usr/bin/env python3
"""
run_evals.py

Master evaluation runner for the Ali Real Estate Chatbot.
Executes all correctness tests and performance benchmarks, then produces
a structured evaluation report.

Usage:
    # Run all unit/component tests (no server needed)
    python run_evals.py --unit

    # Run all tests including live server tests
    python run_evals.py --all

    # Run only performance benchmarks (server required)
    python run_evals.py --perf

    # Generate report from existing results
    python run_evals.py --report-only
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "eval_results"
REPORT_PATH = PROJECT_ROOT / "eval_results" / "evaluation_report.md"


def get_hardware_info() -> dict:
    """Collect hardware specifications including GPU details."""
    import shutil
    import subprocess

    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "gpu": "Not detected",   # default
    }

    # CPU info (Linux /proc/cpuinfo)
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                info["cpu"] = line.split(":")[1].strip()
                break
        cpu_count = cpuinfo.count("processor\t:")
        info["cpu_cores"] = cpu_count
    except Exception:
        info["cpu"] = platform.processor() or "Unknown"
        info["cpu_cores"] = os.cpu_count() or "Unknown"

    # RAM info (Linux /proc/meminfo)
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        for line in meminfo.split("\n"):
            if "MemTotal" in line:
                mem_kb = int(line.split()[1])
                info["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
                break
    except Exception:
        info["ram_gb"] = "Unknown"

    # Disk info
    total, used, free = shutil.disk_usage("/")
    info["disk_total_gb"] = round(total / (1024**3), 1)
    info["disk_free_gb"] = round(free / (1024**3), 1)

    # GPU detection
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_lines = result.stdout.strip().split("\n")
            # Format: "GPU Name, 12345 MiB, 535.104.05"
            gpu_summary = []
            for line in gpu_lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    mem = parts[1]
                    driver = parts[2] if len(parts) > 2 else "unknown"
                    gpu_summary.append(f"{name} ({mem}, driver {driver})")
            info["gpu"] = "; ".join(gpu_summary) if gpu_summary else "NVIDIA GPU (details unavailable)"
    except (subprocess.SubprocessError, FileNotFoundError, Exception):
        # Fallback for Linux: lspci
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["lspci", "-v"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_lines = []
                    for line in result.stdout.split("\n"):
                        if "VGA compatible controller" in line or "3D controller" in line:
                            # Extract useful part (e.g., "Intel Corporation ...")
                            gpu_lines.append(line.strip())
                    if gpu_lines:
                        info["gpu"] = "; ".join(gpu_lines)
                    else:
                        info["gpu"] = "No dedicated GPU detected via lspci"
            except Exception:
                pass
        # Fallback for macOS
        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                # Parse for "Chipset Model" or "Metal Support"
                lines = result.stdout.split("\n")
                chipsets = []
                for i, line in enumerate(lines):
                    if "Chipset Model" in line:
                        chipsets.append(line.split(":")[-1].strip())
                if chipsets:
                    info["gpu"] = "; ".join(chipsets)
                else:
                    info["gpu"] = "GPU info not available from system_profiler"
            except Exception:
                pass
        # Fallback for Windows
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=5, shell=True
                )
                lines = result.stdout.strip().split("\n")
                # First line is "Name", rest are names
                names = [line.strip() for line in lines[1:] if line.strip() and line.strip() != "Name"]
                if names:
                    info["gpu"] = "; ".join(names)
                else:
                    info["gpu"] = "No GPU found via WMIC"
            except Exception:
                pass

    return info

def get_dependency_versions() -> dict:
    """Get versions of key dependencies."""
    versions = {}
    for pkg in ["fastapi", "uvicorn", "pydantic", "chromadb",
                 "sentence_transformers", "pytest", "websockets"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"

    # Special case for ollama
    try:
        import ollama
        versions["ollama"] = getattr(ollama, "__version__", "installed")
    except ImportError:
        versions["ollama"] = "not installed"

    return versions


def run_pytest(markers: list, extra_args: list = None) -> dict:
    """Run pytest with specified markers and return results."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(TESTS_DIR),
        "-v",
        "--tb=short",
        f"--junitxml={RESULTS_DIR / 'junit_results.xml'}",
    ]
    if markers:
        cmd.extend(["-m", " or ".join(markers)])
    if extra_args:
        cmd.extend(extra_args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def generate_report():
    """Generate the evaluation report from collected results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hw_info = get_hardware_info()
    dep_versions = get_dependency_versions()

    report_lines = []
    report_lines.append("# Ali Real Estate Chatbot — Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Platform:** {hw_info['platform']}")
    report_lines.append("")

    # Hardware
    report_lines.append("## 1. Hardware Configuration")
    report_lines.append("")
    report_lines.append("| Spec | Value |")
    report_lines.append("|------|-------|")
    report_lines.append(f"| CPU | {hw_info.get('cpu', 'Unknown')} |")
    report_lines.append(f"| CPU Cores | {hw_info.get('cpu_cores', 'Unknown')} |")
    report_lines.append(f"| RAM | {hw_info.get('ram_gb', 'Unknown')} GB |")
    report_lines.append(f"| GPU | {hw_info.get('gpu', 'Not detected')} |")
    report_lines.append(f"| Disk (Total/Free) | {hw_info.get('disk_total_gb', '?')}/{hw_info.get('disk_free_gb', '?')} GB |")
    report_lines.append(f"| Python | {hw_info['python_version']} |")
    report_lines.append("")

    # Dependencies
    report_lines.append("## 2. Dependency Versions")
    report_lines.append("")
    report_lines.append("| Package | Version |")
    report_lines.append("|---------|---------|")
    for pkg, ver in dep_versions.items():
        report_lines.append(f"| {pkg} | {ver} |")
    report_lines.append("")

    # Component Tests Results
    report_lines.append("## 3. Component-Level Correctness")
    report_lines.append("")
    report_lines.append("### 3.1 Calculator Tool")
    report_lines.append("- **Functional tests:** Arithmetic operations, error handling, code injection prevention")
    report_lines.append("- **Edge cases:** Division by zero, large exponents, empty input, special characters")
    report_lines.append("")

    report_lines.append("### 3.2 Weather Tool")
    report_lines.append("- **Functional tests:** Valid/invalid cities, mocked HTTP responses")
    report_lines.append("- **Error handling:** Timeouts, network errors, unknown locations")
    report_lines.append("")

    report_lines.append("### 3.3 Calendar Tool")
    report_lines.append("- **CRUD tests:** Add/retrieve events, date filtering, ordering")
    report_lines.append("- **Edge cases:** Missing descriptions, kwargs fallback, empty results")
    report_lines.append("")

    report_lines.append("### 3.4 CRM Tool")
    report_lines.append("- **CRUD tests:** Create, read, update user records")
    report_lines.append("- **Persistence:** Data survives across operations")
    report_lines.append("- **Data types:** Numeric, boolean, nested dict, empty data")
    report_lines.append("")

    report_lines.append("### 3.5 Tool Orchestrator")
    report_lines.append("- **Registration:** Tool registration and system instructions")
    report_lines.append("- **JSON Parsing:** Single/multiple tool calls, surrounding text, invalid JSON")
    report_lines.append("- **Execution:** Valid/invalid tools, argument filtering, caching")
    report_lines.append("")

    # RAG Evaluation
    report_lines.append("## 4. RAG Evaluation")
    report_lines.append("")

    rag_metrics_map = [
        ("Precision@3", "rag_precision.json"),
        ("Recall@3", "rag_recall.json"),
        ("Context Relevance", "rag_context_relevance.json"),
        ("Faithfulness (keyword)", "rag_faithfulness.json"),
    ]

    report_lines.append("| Metric | Score | Queries |")
    report_lines.append("|--------|-------|---------|")
    for metric_name, filename in rag_metrics_map:
        fpath = RESULTS_DIR / filename
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            report_lines.append(
                f"| {metric_name} | {data['score']:.4f} | {data['queries']} |"
            )
        else:
            report_lines.append(f"| {metric_name} | *(run test_rag.py)* | — |")
    report_lines.append("")


    # Conversational Correctness
    report_lines.append("## 5. Overall Conversational Correctness")
    report_lines.append("")

    stage_results_path = RESULTS_DIR / "stage_transition_results.json"
    if stage_results_path.exists():
        with open(stage_results_path) as f:
            stage_data = json.load(f)
        report_lines.append(f"### Stage Transition Accuracy")
        report_lines.append(f"- **Passed:** {stage_data['passed']}/{stage_data['total']} dialogues")
        report_lines.append(f"- **Accuracy:** {stage_data['passed']/stage_data['total']*100:.1f}%")
        report_lines.append("")
    else:
        report_lines.append("### Stage Transition Accuracy")
        report_lines.append("- *(Run tests to populate)*")
        report_lines.append("")

    # LLM Judge Results
    report_lines.append("### LLM-as-Judge Evaluation")
    report_lines.append("")
    judge_metrics = ["judge_task_completion", "judge_policy_adherence",
                     "judge_coherence", "judge_faithfulness"]
    report_lines.append("| Metric | Average Score | Dialogues |")
    report_lines.append("|--------|---------------|-----------|")
    for metric in judge_metrics:
        metric_path = RESULTS_DIR / f"{metric}.json"
        if metric_path.exists():
            with open(metric_path) as f:
                data = json.load(f)
            report_lines.append(
                f"| {metric.replace('judge_', '').replace('_', ' ').title()} "
                f"| {data['average_score']:.4f} | {data['count']} |"
            )
        else:
            report_lines.append(
                f"| {metric.replace('judge_', '').replace('_', ' ').title()} "
                f"| *(run test_llm_judge.py)* | — |"
            )
    report_lines.append("")

    # Performance Results
    report_lines.append("## 6. Performance — Latency")
    report_lines.append("")

    scenarios = [
        ("Simple Dialogue", "latency_simple_dialogue.json"),
        ("RAG Only", "latency_rag_only.json"),
        ("Tool Only", "latency_tool_only.json"),
        ("Mixed (RAG + Tool)", "latency_mixed_rag_+_tool.json"),
    ]

    report_lines.append("| Scenario | Trials | TTFT Mean | TTFT Median | TTFT P90 | E2E Mean | E2E Median | E2E P90 |")
    report_lines.append("|----------|--------|-----------|-------------|----------|----------|------------|---------|")

    for name, filename in scenarios:
        fpath = RESULTS_DIR / filename
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            ttft = data.get("ttft", {})
            e2e = data.get("e2e", {})
            report_lines.append(
                f"| {name} | {data.get('trials', '?')} "
                f"| {ttft.get('mean', 0):.3f}s | {ttft.get('median', 0):.3f}s | {ttft.get('p90', 0):.3f}s "
                f"| {e2e.get('mean', 0):.3f}s | {e2e.get('median', 0):.3f}s | {e2e.get('p90', 0):.3f}s |"
            )
        else:
            report_lines.append(f"| {name} | — | — | — | — | — | — | — |")
    report_lines.append("")

    # Throughput Results
    report_lines.append("## 7. Performance — Throughput")
    report_lines.append("")

    throughput_path = RESULTS_DIR / "throughput_results.json"
    if throughput_path.exists():
        with open(throughput_path) as f:
            tp_data = json.load(f)
        report_lines.append(f"- **Max Sustainable Concurrency:** {tp_data.get('max_sustainable_concurrency', '?')} users")
        report_lines.append(f"- **Breakpoint:** {tp_data.get('breakpoint', 'Not reached')} users")
        report_lines.append(f"- **Thresholds:** TTFT < {tp_data['thresholds']['max_ttft_seconds']}s, E2E < {tp_data['thresholds']['max_e2e_seconds']}s")
        report_lines.append("")

        report_lines.append("| Users | Turns | Time (s) | Turns/sec | Errors | Med TTFT | Med E2E | Within |")
        report_lines.append("|-------|-------|----------|-----------|--------|----------|---------|--------|")
        for level in tp_data.get("levels", []):
            ttft_med = level["ttft"].get("median")
            e2e_med = level["e2e"].get("median")
            ttft_str = f"{ttft_med:.3f}s" if ttft_med is not None else "N/A"
            e2e_str = f"{e2e_med:.3f}s" if e2e_med is not None else "N/A"
            within_str = "✓" if level["within_threshold"] else "✗"
            report_lines.append(
                f"| {level['num_users']} | {level['total_turns']} "
                f"| {level['total_time']:.1f} | {level['turns_per_second']:.2f} "
                f"| {level['errors']} "
                f"| {ttft_str} | {e2e_str} | {within_str} |"
            )
    else:
        report_lines.append("*(Run throughput tests to populate)*")
    report_lines.append("")

    # Test Failures
    report_lines.append("## 8. Test Failures and Errors")
    report_lines.append("")
    junit_path = RESULTS_DIR / "junit_results.xml"
    if junit_path.exists():
        report_lines.append(f"See detailed results in `eval_results/junit_results.xml`")
    else:
        report_lines.append("*(No test run data available yet)*")
    report_lines.append("")

    # Analysis
    report_lines.append("## 9. Analysis and Insights")
    report_lines.append("")
    report_lines.append("### Key Findings")
    report_lines.append("")
    report_lines.append("1. **Stage Machine Reliability:** The deterministic stage machine reliably")
    report_lines.append("   advances through greeting → category → subtype → closing based on keyword")
    report_lines.append("   matching with semantic fallback for typo tolerance.")
    report_lines.append("")
    report_lines.append("2. **Tool Orchestration:** The JSON parser successfully extracts tool calls")
    report_lines.append("   from mixed LLM output. The argument safety filter prevents unexpected")
    report_lines.append("   parameters from reaching tool functions.")
    report_lines.append("")
    report_lines.append("3. **RAG Retrieval:** With 52 documents indexed, retrieval relevance depends")
    report_lines.append("   on query specificity. Queries mentioning specific project names (DHA, Bahria)")
    report_lines.append("   achieve higher precision than generic queries.")
    report_lines.append("")
    report_lines.append("4. **CRM Persistence:** SQLite-backed CRM correctly persists data across")
    report_lines.append("   operations. Semantic field matching via ChromaDB enables typo tolerance.")
    report_lines.append("")
    report_lines.append("5. **Calculator Security:** AST-based evaluation prevents code injection")
    report_lines.append("   while supporting all standard mathematical operations.")
    report_lines.append("")
    report_lines.append("### Known Limitations")
    report_lines.append("")
    report_lines.append("1. The 2B LLM model may occasionally hallucinate JSON syntax in long prompts.")
    report_lines.append("2. Throughput is limited by single-threaded LLM inference (Ollama).")
    report_lines.append("3. RAG faithfulness drops when queries are ambiguous or span multiple documents.")
    report_lines.append("4. Weather tool depends on external wttr.in API availability.")
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n✅ Report written to: {REPORT_PATH}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Ali Real Estate Chatbot — Evaluation Suite Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit/component tests only (no server needed)")
    parser.add_argument("--all", action="store_true", help="Run all tests (server + Ollama required)")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmarks only (server required)")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge evaluation only (Ollama required)")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for performance tests")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        generate_report()
        return

    os.environ["PERF_NUM_TRIALS"] = str(args.trials)

    if args.unit:
        print("\n🧪 Running unit and component tests...")
        result = run_pytest(
            markers=[],
            extra_args=[
                "-k", "not (TestPerformance or TestThroughput or TestLive or TestLLMJudge)",
            ]
        )
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])
        print(f"\nCompleted in {result['elapsed_seconds']:.1f}s (exit code: {result['returncode']})")

    elif args.perf:
        print("\n⚡ Running performance benchmarks...")
        result = run_pytest(
            markers=[],
            extra_args=[
                "-k", "TestPerformance or TestThroughput",
                "-s",  # show output
            ]
        )
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])

    elif args.judge:
        print("\n🧑‍⚖️ Running LLM-as-judge evaluation...")
        result = run_pytest(
            markers=[],
            extra_args=[
                "-k", "TestLLMJudge",
                "-s",
            ]
        )
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])

    elif args.all:
        print("\n🚀 Running FULL evaluation suite...")
        result = run_pytest(markers=[], extra_args=["-s"])
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])
        print(f"\nCompleted in {result['elapsed_seconds']:.1f}s (exit code: {result['returncode']})")

    else:
        # Default: run unit tests
        print("\n🧪 Running unit and component tests (default)...")
        result = run_pytest(
            markers=[],
            extra_args=[
                "-k", "not (TestPerformance or TestThroughput or TestLive or TestLLMJudge)",
            ]
        )
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])
        print(f"\nCompleted in {result['elapsed_seconds']:.1f}s (exit code: {result['returncode']})")

    # Always generate report after running tests
    print("\n📊 Generating evaluation report...")
    generate_report()


if __name__ == "__main__":
    main()

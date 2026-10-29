"""Regression checks for the measured-counter CSV boundary (no GPU required)."""
import csv
import io
import unittest

from run_experiment import METRICS, parse_ncu


def fixture(kernels):
    data = io.StringIO()
    writer = csv.writer(data, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    for i, seconds, unit, sectors, dram in kernels:
        for metric, value, metric_unit in zip(METRICS, [seconds, sectors, sectors * 2, dram],
                                               [unit, "sector", "sector", "byte"]):
            writer.writerow([i, "gemm or helper<int, half>", metric, metric_unit, f"{value:,}"])
    return '==PROF== Disconnected\n{"validation":"PASS"}\n' + data.getvalue()


class CounterParsingTests(unittest.TestCase):
    def test_complete_call_includes_helper_kernels_and_converts_units(self):
        result = parse_ncu(fixture([(0, 1000, "ns", 2000, 3000), (1, 2, "us", 100, 600)]))
        self.assertAlmostEqual(result["seconds"], 3e-6)
        self.assertEqual(result["L1_bytes"], 67200)
        self.assertEqual(result["L2_bytes"], 134400)
        self.assertEqual(result["DRAM_bytes"], 3600)
        self.assertEqual(len(result["kernels"]), 2)

    def test_missing_counter_is_an_error(self):
        text = fixture([(0, 1, "ms", 10, 20)])
        text = "\n".join(line for line in text.splitlines() if METRICS[2] not in line)
        with self.assertRaisesRegex(RuntimeError, "指标缺失"):
            parse_ncu(text)

    def test_profiler_failure_cannot_become_synthetic_data(self):
        with self.assertRaisesRegex(RuntimeError, "CSV"):
            parse_ncu("==ERROR== ERR_NVGPUCTRPERM")

    def test_zero_duration_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "时间无效"):
            parse_ncu(fixture([(0, 0, "ns", 10, 20)]))

    def test_nonfinite_counters_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "无效计数器"):
            parse_ncu(fixture([(0, float("nan"), "ns", 10, 20)]))


if __name__ == "__main__":
    unittest.main()

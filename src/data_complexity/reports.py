"""
Module: reports.py
Description: Aggregates computed metric vectors and serializes them into 
             standardized JSON outputs and analytical visual reports.
"""

import json
import os
from typing import Any, Dict


class ComplexityReporter:
    """
    Consolidates data complexity results into machine-readable and 
    human-readable formats.
    """
    def __init__(self):
        self.metrics_summary: Dict[str, Any] = {}

    def log_metric(self, metric_name: str, value: float) -> None:
        """
        Registers a calculated metric into the internal summary state.
        """
        self.metrics_summary[metric_name] = float(value)

    def to_json(self) -> str:
        """
        Serializes the registered metrics into a clean, minified JSON string 
        suitable for API responses or database logging.
        """
        output = {
            "status": "success",
            "analytics": {
                "complexity_metrics": self.metrics_summary
            }
        }
        return json.dumps(output, indent=4)

    def save_json_report(self, output_path: str) -> None:
        """
        Saves the structured JSON data into a physical file on disk.
        """
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

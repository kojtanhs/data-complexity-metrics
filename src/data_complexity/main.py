"""
Module: main.py
Description: Main execution pipeline that orchestrates preprocessing, 
             metric calculation, and report generation.
"""

import pandas as pd
from data_complexity.preprocessing import DataPreprocessor
from data_complexity.metrics.overlap import fisher_discriminant_ratio
from data_complexity.metrics.geometry import neighborhood_frontier_ratio
from data_complexity.reports import ComplexityReporter


def run_pipeline(df: pd.DataFrame, target_column: str) -> str:
    """
    Executes the complete complexity analytical workflow over a DataFrame.
    
    Returns:
        str: A beauty-printed JSON report with metadata and evaluations.
    """
    # 1. Initialize and run pipeline preprocessing
    preprocessor = DataPreprocessor(target_column=target_column, scale_features=True)
    X, y = preprocessor.fit_transform(df)
    
    # 2. Trigger individual feature space metrics
    f1_score = fisher_discriminant_ratio(X, y)
    n1_proxy = neighborhood_frontier_ratio(X, y)
    
    # 3. Log results via the Reporter Layer
    reporter = ComplexityReporter()
    reporter.log_metric("maximum_fisher_discriminant_ratio_F1", f1_score)
    reporter.log_metric("neighborhood_frontier_ratio_N1_proxy", n1_proxy)
    
    return reporter.to_json()


if __name__ == "__main__":
    # Small runtime smoke test simulation (Credit scoring mock data style)
    mock_credit_data = {
        "monthly_income": [2500, 1200, 4000, 800, 3500],
        "debt_ratio": [0.3, 0.6, 0.1, 0.8, 0.2],
        "default_risk": [0, 1, 0, 1, 0]
    }
    sample_df = pd.DataFrame(mock_credit_data)
    
    print("Executing internal data-complexity smoke test pipeline...\n")
    json_result = run_pipeline(sample_df, target_column="default_risk")
    print(json_result)

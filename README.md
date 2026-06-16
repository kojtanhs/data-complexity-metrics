# Data Complexity Engine

A modular Python library designed to evaluate classification dataset complexity within MLOps/DataOps pipelines using strict separation of concerns.

## Pipeline Architecture

```mermaid
graph TD
    %% Estilos de bloques
    classDef input fill:#1a1a2e,stroke:#16a085,stroke-width:2px,color:#fff;
    classDef core fill:#16213e,stroke:#2980b9,stroke-width:2px,color:#fff;
    classDef utils fill:#1f4068,stroke:#8e44ad,stroke-width:2px,color:#fff;
    classDef output fill:#0f3460,stroke:#27ae60,stroke-width:2px,color:#fff;

    %% Flujo de Ingesta
    A[Raw Data Input: DataFrame / SQL Query] --> B[Data Preprocessor & Scaler]
    class A input;
    class B core;

    %% Modulaciones del Core Analytical Engine
    subgraph Core Analytical Engine [Lógica Modular /src]
        B --> C[Overlap Metrics Module]
        B --> D[Geometry & Topology Module]
        B --> E[Dimensionality & Sparsity Module]
        
        C --> C1[F1: Maximum Fisher's Discriminant Ratio]
        D --> D1[Distance-Based Graph Neighborhoods]
        E --> E1[T2: Average Number of Features per Instance]
    end
    class C,D,E core;
    class C1,D1,E1 utils;

    %% Flujo de Evaluación y Optimización
    C1 --> F[Complexity Matrix Aggregator]
    D1 --> F
    E1 --> F
    class F core;

    %% Salidas / Outputs
    subgraph Output Generation Layer
        F --> G[JSON Report: Complexity Vector]
        F --> H[Matplotlib / TikZ Visualizations]
    end
    class G,H output;

    %% Guardrails y Tests
    subgraph Quality Assurance
        I[PyTest: Unit Tests & Edge Cases] -.-> B
    end
    class I utils;
```


## Analytical Design

### 1. Overlap Module
Computes Maximum Fisher's Discriminant Ratio ($F1$):

$$F1 = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}$$

Low scores signal severe class overlapping.

### 2. Geometry Module
Computes Neighborhood Frontier Ratio ($N1$ proxy) using Euclidean metrics:

$$N1 = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( y_i \neq y_{\text{NN}(i)} \right)$$

Values near $1.0$ indicate high topological boundary complexity.

## Architecture

```text
data-complexity-metrics/
├── pyproject.toml
├── src/data_complexity/
│   ├── preprocessing.py
│   ├── reports.py (JSON output)
│   └── metrics/ (overlap.py, geometry.py)
└── tests/ (pytest automated validation)
```
## Verification
```
pip install -e .
pytest -v tests/
python src/data_complexity/main.py
```

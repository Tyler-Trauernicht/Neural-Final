# EE4745 Final Results Compilation - Usage Guide

**Project:** "Defending LSU's Sports AI: Classification, Attacks, and Compression"
**Authors:** Tyler Trauernicht, Vinh Le
**Date:** November 2025

---

## Quick Start

### 1. Run Complete Analysis
```bash
# From project root directory
cd /Users/ty/Neural-Final-Tyler_Vinh
python3 results/final/run_final_analysis.py
```

### 2. View Results
- **Start here:** `results/final/README.md`
- **Executive Summary:** `results/final/summary/executive_summary.md`
- **Master Dashboard:** `results/final/figures/master_performance_dashboard.png`

---

## System Overview

This comprehensive final results compilation system provides:

### 📊 **Automated Results Compilation**
- Loads experimental data from all three problems
- Generates performance comparison tables (CSV + LaTeX)
- Creates publication-quality visualizations
- Produces detailed analysis reports
- Generates executive summaries

### 🎯 **Key Components**

#### 1. Results Compiler (`analysis/final_results_compiler.py`)
- **Purpose:** Consolidate all experimental results
- **Output:** Performance tables, statistical summaries
- **Features:** Handles missing data with template structures

#### 2. Visualization Dashboard (`analysis/visualization_dashboard.py`)
- **Purpose:** Create comprehensive visual analysis
- **Output:** Publication-quality figures and plots
- **Features:** Multi-problem comparisons, trade-off analysis

#### 3. Report Generator (`analysis/report_generator.py`)
- **Purpose:** Generate detailed analysis reports
- **Output:** Markdown reports, executive summaries
- **Features:** Cross-problem insights, deployment recommendations

#### 4. Master Runner (`run_final_analysis.py`)
- **Purpose:** Coordinate entire analysis pipeline
- **Output:** Complete results package
- **Features:** Validation, documentation generation

---

## File Organization

```
results/final/
├── README.md                    # Main documentation index
├── USAGE_GUIDE.md              # This file
├── run_final_analysis.py       # Master execution script
│
├── analysis/                   # Analysis scripts
│   ├── final_results_compiler.py
│   ├── visualization_dashboard.py
│   └── report_generator.py
│
├── tables/                     # Performance data
│   ├── problem_a_model_comparison.csv/.tex
│   ├── problem_b_attack_comparison.csv/.tex
│   ├── problem_c_pruning_comparison.csv/.tex
│   └── master_performance_comparison.csv/.tex
│
├── figures/                    # Visualizations
│   ├── master_performance_dashboard.png
│   ├── problem_a_*.png
│   ├── problem_b_*.png
│   └── problem_c_*.png
│
├── reports/                    # Detailed analysis
│   ├── problem_a_analysis_report.md
│   ├── problem_b_analysis_report.md
│   ├── problem_c_analysis_report.md
│   └── integrated_analysis_report.md
│
└── summary/                    # Executive materials
    ├── executive_summary.md
    ├── methodology_summary.md
    └── project_summary.tex
```

---

## Individual Script Usage

### Results Compilation Only
```python
from results.final.analysis.final_results_compiler import FinalResultsCompiler

compiler = FinalResultsCompiler("/path/to/project")
compiler.load_all_results()
compiler.generate_all_tables()
```

### Visualizations Only
```python
from results.final.analysis.visualization_dashboard import VisualizationDashboard

dashboard = VisualizationDashboard("/path/to/project")
dashboard.create_all_visualizations()
```

### Reports Only
```python
from results.final.analysis.report_generator import ReportGenerator

reporter = ReportGenerator("/path/to/project")
reporter.generate_all_reports()
```

---

## Customization

### Adding New Metrics
1. **Modify data loading** in `final_results_compiler.py`
2. **Update table generation** methods
3. **Add visualizations** in `visualization_dashboard.py`
4. **Update reports** in `report_generator.py`

### Template Data vs Real Results
The system includes template data for demonstration. To use with real experimental results:

1. **Save experimental data** to:
   - `results/problem_a/classification_results.json`
   - `results/problem_b/adversarial_results.json`
   - `results/problem_c/pruning_results.json`

2. **Follow JSON structure** as defined in compiler templates

3. **Re-run analysis** to load actual experimental data

### Expected JSON Format

#### Problem A Results
```json
{
  "models": {
    "SimpleCNN": {
      "train_accuracy": 0.854,
      "val_accuracy": 0.832,
      "test_accuracy": 0.828,
      "parameters": 147000,
      "training_time": 45.2,
      "train_losses": [...],
      "val_losses": [...],
      "train_accuracies": [...],
      "val_accuracies": [...]
    }
  }
}
```

#### Problem B Results
```json
{
  "attacks": {
    "FGSM": {
      "untargeted": {
        "success_rates": {"eps_0.01": 0.15, "eps_0.03": 0.45},
        "perturbation_norms": {"eps_0.01": 0.01, "eps_0.03": 0.03}
      }
    }
  }
}
```

#### Problem C Results
```json
{
  "pruning_levels": {
    "20%": {
      "SimpleCNN": {
        "accuracy": 0.83,
        "model_size_mb": 0.47,
        "inference_time_ms": 1.8
      }
    }
  }
}
```

---

## Output Descriptions

### Tables (CSV/LaTeX)
- **Problem A:** Model accuracy, parameters, training time comparison
- **Problem B:** Attack success rates across epsilon values
- **Problem C:** Pruning trade-offs (accuracy vs compression)
- **Master:** Comprehensive cross-problem performance summary

### Visualizations (PNG)
- **Training curves:** Loss and accuracy progression
- **Model comparisons:** Performance vs complexity trade-offs
- **Attack effectiveness:** Success rates vs perturbation strength
- **Pruning analysis:** Accuracy vs compression trade-offs
- **Master dashboard:** Comprehensive overview with key insights

### Reports (Markdown)
- **Individual problem reports:** Detailed analysis and findings
- **Integrated analysis:** Cross-problem correlations and insights
- **Executive summary:** High-level overview and recommendations
- **Methodology summary:** Experimental procedures and reproducibility

---

## Integration with Existing Project

### For Active Development
1. **Save experimental results** in the expected JSON format
2. **Run analysis pipeline** after each major experiment
3. **Update visualizations** for presentations
4. **Generate reports** for documentation

### For Final Submission
1. **Ensure all experiments** are completed
2. **Run complete analysis** pipeline
3. **Review generated reports** for accuracy
4. **Package results** for submission

---

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
pip install matplotlib seaborn pandas numpy
```

#### Path Issues
- Ensure you're running from the project root directory
- Use absolute paths in script arguments

#### Missing Data
- System uses template data when real results are unavailable
- Check console output for warnings about missing files

#### Python Version
- Requires Python 3.7+
- Tested with Python 3.8-3.11

### Error Messages

#### "No module named..."
**Solution:** Install required dependencies with pip

#### "File not found"
**Solution:** Check that you're running from the correct directory

#### "Permission denied"
**Solution:** Ensure write permissions in results directory

---

## Best Practices

### For Reproducibility
1. **Use fixed random seeds** in all experiments
2. **Document hyperparameters** in result files
3. **Save model checkpoints** for later analysis
4. **Version control** experimental configurations

### For Performance
1. **Run analysis** after all experiments complete
2. **Use template data** for development/testing
3. **Generate visualizations** only when needed
4. **Cache results** to avoid recomputation

### For Collaboration
1. **Use standardized** JSON format for results
2. **Document experiment** configurations
3. **Share analysis scripts** with consistent interfaces
4. **Maintain result** file organization

---

## Extending the System

### Adding New Problems
1. Create new result loading method in compiler
2. Add visualization functions in dashboard
3. Create new report section in generator
4. Update master runner integration

### Adding New Visualizations
1. Implement plot function in dashboard
2. Call from `create_all_visualizations()`
3. Update documentation index
4. Test with template data

### Adding New Reports
1. Create report generation method
2. Add to report generator pipeline
3. Update master summary
4. Include in validation check

---

## Contact and Support

**Authors:** Tyler Trauernicht, Vinh Le
**Course:** EE4745 Neural Networks
**Repository:** https://github.com/Tyler-Trauernicht/Neural-Final.git

For questions about the analysis system or to report issues, please refer to the methodology summary and code documentation.

---

## License and Attribution

This analysis system is part of the EE4745 Neural Network Final Project. The code is designed for educational purposes and can be extended for research applications with proper attribution.

All visualizations and reports are generated automatically from experimental data and analysis scripts.
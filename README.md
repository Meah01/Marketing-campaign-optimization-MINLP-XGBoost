# Optimizing Marketing Campaigns Using Machine Learning - Analysing and Predicting Ad Performance for Improved Budget Allocation
Author: Alexandru Constantinescu
Date: 30.05.2025

Description
This code represents an integration of a Mixed Integer Nonlinear Programming (MINLP) model with an XGBoost to optimize and predict marketing campaign metrics. It was created as a bachelor's thesis project.

---------------

Abstract
This repository presents a novel hybrid optimization system that integrates Mixed Integer Nonlinear Programming (MINLP) with XGBoost machine learning for digital marketing campaign optimization. The system addresses critical gaps in Marketing Mix Modeling by combining prescriptive optimization with predictive analytics. Through comprehensive testing on multi-platform campaign data (Facebook, Google, DV360), the MINLP model demonstrates 13-25% performance improvements over traditional Linear Programming approaches while maintaining robust stability (75% of tests within ±15% variance). The integrated XGBoost models achieve high prediction accuracy (R² up to 0.94) for campaign metrics forecasting. This work provides the first empirical validation of MINLP effectiveness in marketing applications and establishes a practical framework for practitioners seeking data-driven campaign optimization solutions.

---------------

Acknoledgement
Claude Sonnet 3.5 was used during the development process.

---------------

Features
- **MINLP Optimization Engine**: Advanced nonlinear optimization with diminishing returns modeling
- **XGBoost Prediction Models**: High-accuracy performance forecasting (R² up to 0.94)
- **Multi-Platform Support**: Facebook Ads, Google Ads, DV360
- **Campaign Types**: Awareness and Conversion campaigns
- **GUI Interface**: User-friendly interface for practitioners
- **Comprehensive Analysis**: Statistical significance testing and sensitivity analysis
- **Benchmark Comparisons**: Performance evaluation against Linear Programming baselines

---------------

BE ADVISED
- If the file in MINLP - prototype is run, it can potentially block the system temporarly, as
the GridSearchCV uses a lot of memory
- If the file in XGBoost - prototype is run, it can also block or even crash the system, as
the file contains multiple Optuna searches. Otherwise, it can take an approximate 1-2 hours to 
run in its entirety, though this time largely depends on the system's specs 
- The pkl files that are the outputs of the XGBoost's prototype will be placed in the same directory
as the prototype file 
- The paths for the files might need to be adapted and changed based on the location of the downloaded
files
- Make sure you have intalled all the requirements prior to running the code. The Bonmin solver can be tricky
to download. Use the instructions provided on the COIN-OR website.
---------------

Intallation
1. Download repository
2. Intall Bonmin solver from COIN-OR (https://www.coin-or.org/Bonmin/Obtain.html)
3. Intall requirements (pip and packages mentioned in "requirements.txt")
4. Make sure Jupyter Notebooks are openable in the IDE of your choosing
5. Run the code in your python IDE

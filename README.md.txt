Optimizing Marketing Campaigns Using Machine Learning - Analysing and 
Predicting Ad Performance for Improved Budget Allocation
Date:

[Description]

[Overview/Abstract]

Acknoledgement
Claude Sonnet 3.5 was used during the development process.

---------------

Repository Structure
├── data
│	├── raw data
│	└── clean data
├── EDA
├── interface
├── MINLP
│	├── final
│	├── prototype
│	├── sensitivity analysis
│	└── comparison
├── XGBoost
│	├── final
│	├── prototype
│	└── hypothesis testing
├── README.md
├── license 
└── requirements	

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

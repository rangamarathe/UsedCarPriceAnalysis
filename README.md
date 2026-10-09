# What Drives the Price of a Used Car?



## Project overview

This project uses 426,880 used-vehicle listings to determine which vehicle characteristics are most useful for estimating advertised price. The analysis follows the CRISP-DM framework and translates the modeling results into recommendations for a used-car dealership.

The work addresses two questions:

1. How accurately can listing price be estimated from observable vehicle attributes?
2. Which attributes have the greatest influence on those estimates?

The response variable is advertised price rather than completed sale price. Consequently, the findings describe predictive associations in the listing data and should not be interpreted as causal effects or exact market premiums.

## Project files

```text
practical_application_II_starter/
├── README.md
├── used_car_price_analysis.ipynb
├── data/
│   └── vehicles.csv
└── images/
    ├── crisp.png
    └── kurt.jpeg
```

- `used_car_price_analysis.ipynb` contains the completed and executed analysis.
- `data/vehicles.csv` is the course-provided dataset.

## CRISP-DM workflow

### 1. Business understanding

The dealership's question is formulated as a supervised regression problem. Each row represents a vehicle listing, `price` is the continuous target, and the remaining selected vehicle attributes are predictors.

The primary evaluation metric is mean absolute error (MAE), reported in dollars. MAE is directly interpretable as the typical absolute difference between an advertised price and a model estimate. RMSE and R² are also reported to capture large errors and explained variance.

### 2. Data understanding

The notebook examines:

- dataset shape, column types, and descriptive statistics;
- missing-value rates;
- raw duplicates and category cardinality;
- price, year, and odometer distributions;
- listings by manufacturer, model, condition, fuel, transmission, body type, paint color, and drive configuration;
- the relationship between price, condition, and odometer.

Important quality findings include extreme prices, odometer values as high as 10 million miles, implausible model years, substantial missingness in several categorical fields, and almost 30,000 free-text model labels.

### 3. Data preparation

The following rules are applied:

- Remove `id` and `VIN`, which are vehicle/listing identifiers.
- Remove `model` because its approximately 30,000 free-text values create excessive cardinality and noise.
- Remove `region`; `state` is retained as the broader geographic control.
- Remove duplicates after dropping those identifier and high-cardinality fields.
- Retain listings priced from **$500 to $100,000**.
- Retain odometer readings from **0 to 300,000 miles**.
- Retain model years from **1990 to 2026**.
- Drop records missing required low-missingness fields: manufacturer, odometer, year, fuel, transmission, or title status.
- Fill missing condition, cylinders, drive, size, type, and paint color with an explicit `unknown` category.
- Convert model year into an interpretable `age` feature.
- Compare models trained on raw price with models trained on `log1p(price)`.

Pipeline-based preprocessing prevents information leakage:

- Numeric features are median-imputed and standardized for linear models.
- Categorical features are imputed and one-hot encoded for linear models.
- Categorical features are ordinal-encoded for the Random Forest.
- Unknown categories at prediction time are handled safely.

## Prepared data

| Stage | Rows remaining |
|---|---:|
| Raw dataset | 426,880 |
| After duplicate removal | 296,662 |
| After range/outlier filters | 256,027 |
| After removing records missing required fields | **242,211** |

The final model matrix contains 13 raw predictors:

- Numeric: `odometer`, `age`
- Categorical: `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `size`, `type`, `paint_color`, `state`

An 80/20 split with `random_state=42` produces:

- Training set: **193,768 listings**
- Test set: **48,443 listings**

## Modeling approach

Five regression approaches are evaluated on both raw price and log-transformed price:

1. Median-price baseline
2. Ordinary Linear Regression
3. Ridge Regression
4. Lasso Regression
5. Random Forest Regression

Ridge and Lasso penalty strengths are selected with shuffled three-fold cross-validation and `GridSearchCV`. Random Forest provides a nonlinear comparison that can learn thresholds and interactions among age, mileage, and vehicle configuration.

## Model results

All values below come from the executed notebook's untouched test set. Log-target predictions are converted back to dollars before evaluation.

| Model | Log-target MAE | Log-target RMSE | Log-target R² | Raw-target MAE | Raw-target RMSE | Raw-target R² |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | $3,180 | $5,734 | 0.830 | **$3,072** | **$5,385** | **0.850** |
| Ridge | $4,653 | $7,628 | 0.698 | $5,318 | $7,762 | 0.688 |
| Lasso | $4,653 | $7,634 | 0.698 | $5,318 | $7,766 | 0.687 |
| Linear Regression | $4,655 | $7,640 | 0.697 | $5,318 | $7,766 | 0.687 |
| Median baseline | $10,575 | $14,411 | -0.077 | $10,575 | $14,411 | -0.077 |

### Selected model

The **Random Forest trained on raw dollar prices** is the final model:

- **MAE:** $3,072
- **RMSE:** $5,385
- **R²:** 0.850

It reduces test MAE by approximately **71%** compared with the median baseline and by approximately **42%** compared with raw-target Ridge Regression. The performance gain suggests that important price relationships are nonlinear and include interactions.

The log transformation helps the linear models substantially: log-target Ridge improves from a $5,318 MAE to $4,653. The Random Forest performs best on the untransformed dollar target.

## Feature importance and findings

The notebook uses two complementary interpretation methods:

- **Permutation importance** measures the increase in test error when an original feature is shuffled.
- **SHAP values** estimate each feature's contribution and direction for individual Random Forest predictions.

### Most influential features

Both methods identify the same broad set of leading price drivers:

1. Vehicle age
2. Odometer reading
3. Drive configuration
4. Cylinder count
5. Fuel type
6. Manufacturer
7. Vehicle/body type

The executed SHAP analysis ranks age and odometer far above the remaining predictors. It also shows that greater vehicle age generally pushes predicted price downward, while newer vehicles push it upward.

Condition, transmission, state, paint color, title status, and size add predictive information, but their global influence is smaller than age, mileage, and major mechanical configuration.

## Recommendations for the dealership

-  Vehicle age is the single largest influence on valuation, driving rapid depreciation across inventory. Dealerships should implement strict aging thresholds to markdown or wholesale older stock before margins erode.
-  Mileage ranks as the second most critical price driver, with values dropping sharply at key usage milestones. Acquisition teams must appraise cars aggressively near these thresholds while highlighting low-mileage benefits in digital marketing.
-  Drivetrain, cylinder count, and fuel type heavily outweigh brand and body style in determining secondary market worth. Inventory acquisition should   prioritize these high-demand mechanical configurations to maximize pricing power.
- Cosmetic condition ranks lower in overall impact than core mechanical and structural attributes like age or mileage. Managers should limit heavy spending on minor cosmetic fixes and instead direct capital toward powertrain soundness and desirable vehicle specifications.
-  Track MAE by age, price band, manufacturer, fuel type, and state, then retrain when the inventory mix or market changes.

## Limitations

- Advertised price is not necessarily the final transaction price.
- The dataset does not include accident history, trim and options, service records, number of owners, reconditioning cost, or days on market.
- The model-year upper bound of 2026 follows the notebook's executed cleaning rule and should be revised for a fixed historical dataset or future production use.
- Ordinal encoding assigns numeric codes to categories for the Random Forest. Trees can split on those codes even though the categories have no natural order; one-hot, target, or native categorical modeling would be useful comparisons.
- A random train/test split measures interpolation among similar listings. A production evaluation should also use a time-based split to test future-market performance.
- Economic conditions, fuel prices, incentives, and local supply can shift consumer demand after the data was collected.
- The analysis identifies predictive association, not causation.

## Running the notebook

### Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- SHAP

Install the dependencies with:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn shap
```

From the project directory, start Jupyter:

```bash
jupyter notebook used_car_price_analysis.ipynb
```

The notebook expects the dataset at:

```text
data/vehicles.csv
```

Run the cells from top to bottom. Model fitting, permutation importance, and SHAP analysis may take several minutes depending on available CPU and memory.

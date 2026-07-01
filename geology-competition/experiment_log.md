## v001 - Baseline

### Date
2026-05-27

### Changes
- raw features only
- no feature engineering

### RMSE
XGBoost: 9.610, CatBoost: 10.253

### LB
25.527

### Notes
First working submission. Since there is a huge difference between LB score and RMSE, there might be leakage, possibly due to lack of groupKfolds.

## v002 - new target variable + change model

### Date
2026-05-28

### Changes
- Add the new 'target' = TVT - TVT_input
- change XGBoost to LightGBM

### RMSE
LightGBM: 10.689, CatBoost: 13.127

### LB
17.359

### Notes
LightGBM runs fasters. Still need to add GroupKfold.

## v??? - add Ridge stacking

### Date
2026-06-16

### Changes
- Add ridge stacking

### RMSE
Blending: 10.659, Stacking: 10.403

### LB
-

### Notes
keep stacking
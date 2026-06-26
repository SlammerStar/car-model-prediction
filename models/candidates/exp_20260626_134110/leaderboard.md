# Experiment Leaderboard: exp_20260626_134110

|    | Model        | Features          |       R2 |   CV_R2_Mean |    MAE |     MAPE |   Train_Time |
|---:|:-------------|:------------------|---------:|-------------:|-------:|---------:|-------------:|
|  0 | XGBoost      | baseline          | 0.907038 |     0.920975 | 107411 | 0.168146 |     1.31362  |
|  1 | XGBoost      | pruned_engineered | 0.892009 |     0.912373 | 110119 | 0.172019 |     1.21409  |
|  2 | XGBoost      | full_engineered   | 0.901716 |     0.911618 | 105271 | 0.16397  |     1.46089  |
|  3 | DecisionTree | baseline          | 0.856497 |     0.875554 | 134855 | 0.205664 |     0.359893 |
|  4 | DecisionTree | pruned_engineered | 0.822169 |     0.8541   | 140445 | 0.212754 |     0.450358 |
|  5 | DecisionTree | full_engineered   | 0.868531 |     0.850482 | 128060 | 0.198366 |     0.390072 |
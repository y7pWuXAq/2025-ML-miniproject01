import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import set_config
import joblib

# 경고 메시지 제거
set_config(display="text")

# 한글 및 마이너스 처리
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

class Models:
    def __init__(self):
        self.results = []

    def get_results(self):
        return pd.DataFrame(self.results)

    def evaluate_model(self, model_name, scaler_name, train_score, val_score, test_score, 
                       train_mae, train_mse, train_r2,
                       val_mae, val_mse, val_r2,
                       test_mae, test_mse, test_r2, is_grid_search):
        results = {
            "model_nm": model_name,
            "Scaler": scaler_name,
            "GSCV": "Y" if is_grid_search else "N",
            "train_score": train_score,
            "val_score": val_score,
            "test_score": test_score,
            "과적합여부": train_score - val_score
        }

        # 당신이 배운 기준에 맞춘 수정: 일반화 범위 완화 (0.0~0.1)
        if train_score < 1 and 0.0 <= train_score - val_score <= 0.1:
            results["사용"] = "Y"
            results["train_mae"] = train_mae
            results["train_mse"] = train_mse
            results["train_r2"] = train_r2
            results["val_mae"] = val_mae
            results["val_mse"] = val_mse
            results["val_r2"] = val_r2
            results["test_mae"] = test_mae
            results["test_mse"] = test_mse
            results["test_r2"] = test_r2
            print(f"-*** {model_name} with {scaler_name} ***-")
            print(f"훈련: {train_score:.4f}, 검증: {val_score:.4f}, 테스트: {test_score:.4f}, 과적합여부: {train_score - val_score:.4f}")
            print("사용 가능한 모델입니다 (일반화).\n")
        else:
            print(f"-*** {model_name} with {scaler_name} ***-")
            print(f"훈련: {train_score:.4f}, 검증: {val_score:.4f}, 테스트: {test_score:.4f}, 과적합여부: {train_score - val_score:.4f}")
            if train_score - val_score < 0:
                print("과소적합으로 사용 불가능한 모델입니다.\n")
            elif train_score - val_score > 0.1:
                print("과대적합으로 사용 불가능한 모델입니다.\n")
            elif train_score >= 1:
                print("훈련 정확도 1로 과대적합, 사용 불가능한 모델입니다.\n")
            results["사용"] = "N"
            results["train_mae"] = train_mae
            results["train_mse"] = train_mse
            results["train_r2"] = train_r2
            results["val_mae"] = val_mae
            results["val_mse"] = val_mse
            results["val_r2"] = val_r2
            results["test_mae"] = test_mae
            results["test_mse"] = test_mse
            results["test_r2"] = test_r2
        
        self.results.append(results)

    def total_models(self, train_input, train_target, val_input, val_target, test_input, test_target):
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
            'XGB': XGBRegressor(random_state=42)
        }

        param_distributions = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [5, 10]
            },
            'HistGradientBoosting': {
                'max_iter': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            },
            'XGB': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        }

        scalers = {
            "None": None,
            "Standard": StandardScaler(),
            "MinMax": MinMaxScaler(),
            "Robust": RobustScaler()
        }

        for scaler_name, scaler in scalers.items():
            if scaler:
                scaler.fit(train_input)
                train_scaled = scaler.transform(train_input)
                val_scaled = scaler.transform(val_input)
                test_scaled = scaler.transform(test_input)
            else:
                train_scaled = train_input
                val_scaled = val_input
                test_scaled = test_input

            for model_name, model in models.items():
                print(f"Tuning and training {model_name} with {scaler_name}...")
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_distributions[model_name],
                    n_iter=8,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(train_scaled, train_target)

                best_model = search.best_estimator_
                best_params = search.best_params_
                print(f"Best Parameters for {model_name}: {best_params}")

                train_pred = best_model.predict(train_scaled)
                val_pred = best_model.predict(val_scaled)
                test_pred = best_model.predict(test_scaled)

                train_mae = mean_absolute_error(train_target, train_pred)
                train_mse = mean_squared_error(train_target, train_pred)
                train_r2 = r2_score(train_target, train_pred)

                val_mae = mean_absolute_error(val_target, val_pred)
                val_mse = mean_squared_error(val_target, val_pred)
                val_r2 = r2_score(val_target, val_pred)

                test_mae = mean_absolute_error(test_target, test_pred)
                test_mse = mean_squared_error(test_target, test_pred)
                test_r2 = r2_score(test_target, test_pred)

                self.evaluate_model(model_name, scaler_name, train_r2, val_r2, test_r2,
                                   train_mae, train_mse, train_r2,
                                   val_mae, val_mse, val_r2,
                                   test_mae, test_mse, test_r2, True)

                if train_r2 > 0.7 and val_r2 > 0.7:
                    joblib.dump(best_model, f"{model_name}_{scaler_name}_final_model.pkl")
                    print(f"Saved {model_name} with {scaler_name} as final model.")
"""
Machine Learning Experiment for MXenes Work Function Prediction - GradientBoosting Only
===============================================================================
This experiment only generates GradientBoosting model scatter plots
"""

# ========================= Import Required Libraries =========================
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from datetime import datetime

# Ignore warning messages
warnings.filterwarnings('ignore')

# Set font and figure style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ========================= Configuration Parameters =========================
class Config:
    """Experiment configuration class"""
    DATA_PATH = r'C:\Users\Shang\Desktop\code\cv\c2db_Mxenes+3d描述符.xlsx'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Columns to drop
    DROP_COLUMNS = ['Formula', 'Workfunction', 'X', 'T', 'M']
    TARGET_COLUMN = 'Workfunction'

# ========================= Data Processing Module =========================
class DataProcessor:
    """Data processing class"""
    
    @staticmethod
    def load_and_preprocess(file_path, n_features=None):
        """Load and preprocess data"""
        try:
            data = pd.read_excel(file_path)
            print(f"Data loaded successfully, shape: {data.shape}")
            
            # Drop unnecessary columns
            X = data.drop(Config.DROP_COLUMNS, axis=1)
            y = data[Config.TARGET_COLUMN]
            
            # Select number of features
            if n_features is not None:
                X = X.iloc[:, :n_features]
                print(f"Using first {n_features} features")
            else:
                print(f"Using all {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            return None, None

# ========================= Model Building Module =========================
class ModelBuilder:
    """Model building class"""
    
    @staticmethod
    def get_gradient_boosting_stacking_config():
        """Get GradientBoosting stacking configuration"""
        return {
            'base_models': [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2))),
                ('extra_trees', ExtraTreesRegressor(n_estimators=102, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)))
            ],
            'meta_model': GradientBoostingRegressor(n_estimators=80, random_state=33, max_depth=2),
            'name': 'GradientBoosting Meta-learner'
        }
    
    @staticmethod
    def get_gradient_boosting_single_model():
        """Get GradientBoosting single model"""
        return GradientBoostingRegressor(n_estimators=165, random_state=Config.RANDOM_STATE)

# ========================= Experiment Evaluation Module =========================
class Evaluator:
    """Model evaluation class"""
    
    @staticmethod
    def evaluate_stacking_model(X_train, X_test, y_train, y_test, base_models, meta_model, cv=5):
        """Evaluate Stacking model"""
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=cv)
        model.fit(X_train, y_train)
        
        # Prediction
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred)
        }
        
        return train_metrics, test_metrics
    
    @staticmethod
    def cross_validate_stacking_model_with_fixed_split(X, y, base_models, meta_model, X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, cv_folds=5):
        """Cross-validate Stacking model with fixed train-test split as first fold"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        train_mae_scores = []
        test_mae_scores = []
        train_r2_scores = []
        test_r2_scores = []
        
        predictions_data = []
        
        fold_indices = list(kf.split(X))
        
        for i in range(cv_folds):
            if i == 0:
                # Use fixed split for first fold
                X_train_fold, X_test_fold = X_train_fixed, X_test_fixed
                y_train_fold, y_test_fold = y_train_fixed, y_test_fixed
            else:
                # Use KFold splits for other folds
                train_idx, test_idx = fold_indices[i]
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_train_pred = model.predict(X_train_fold)
            y_test_pred = model.predict(X_test_fold)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            test_mae = mean_absolute_error(y_test_fold, y_test_pred)
            train_r2 = r2_score(y_train_fold, y_train_pred)
            test_r2 = r2_score(y_test_fold, y_test_pred)
            
            train_mae_scores.append(train_mae)
            test_mae_scores.append(test_mae)
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)
            
            # Store predictions for plotting
            predictions_data.append({
                'fold': i + 1,
                'train_actual': y_train_fold.values,
                'train_predicted': y_train_pred,
                'test_actual': y_test_fold.values,
                'test_predicted': y_test_pred,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            })
        
        return {
            'train_MAE_scores': train_mae_scores,
            'test_MAE_scores': test_mae_scores,
            'train_R2_scores': train_r2_scores,
            'test_R2_scores': test_r2_scores,
            'mean_train_MAE': np.mean(train_mae_scores),
            'std_train_MAE': np.std(train_mae_scores),
            'mean_test_MAE': np.mean(test_mae_scores),
            'std_test_MAE': np.std(test_mae_scores),
            'mean_train_R2': np.mean(train_r2_scores),
            'std_train_R2': np.std(train_r2_scores),
            'mean_test_R2': np.mean(test_r2_scores),
            'std_test_R2': np.std(test_r2_scores),
            'predictions_data': predictions_data
        }
    
    @staticmethod
    def cross_validate_model_with_fixed_split(X, y, model, X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, cv_folds=5):
        """Cross-validate single model with fixed train-test split as first fold"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        train_mae_scores = []
        test_mae_scores = []
        train_r2_scores = []
        test_r2_scores = []
        
        predictions_data = []
        
        fold_indices = list(kf.split(X))
        
        for i in range(cv_folds):
            if i == 0:
                # Use fixed split for first fold
                X_train_fold, X_test_fold = X_train_fixed, X_test_fixed
                y_train_fold, y_test_fold = y_train_fixed, y_test_fixed
            else:
                # Use KFold splits for other folds
                train_idx, test_idx = fold_indices[i]
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_train_pred = model_copy.predict(X_train_fold)
            y_test_pred = model_copy.predict(X_test_fold)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_fold, y_train_pred)
            test_mae = mean_absolute_error(y_test_fold, y_test_pred)
            train_r2 = r2_score(y_train_fold, y_train_pred)
            test_r2 = r2_score(y_test_fold, y_test_pred)
            
            train_mae_scores.append(train_mae)
            test_mae_scores.append(test_mae)
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)
            
            # Store predictions for plotting
            predictions_data.append({
                'fold': i + 1,
                'train_actual': y_train_fold.values,
                'train_predicted': y_train_pred,
                'test_actual': y_test_fold.values,
                'test_predicted': y_test_pred,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            })
        
        return {
            'train_MAE_scores': train_mae_scores,
            'test_MAE_scores': test_mae_scores,
            'train_R2_scores': train_r2_scores,
            'test_R2_scores': test_r2_scores,
            'mean_train_MAE': np.mean(train_mae_scores),
            'std_train_MAE': np.std(train_mae_scores),
            'mean_test_MAE': np.mean(test_mae_scores),
            'std_test_MAE': np.std(test_mae_scores),
            'mean_train_R2': np.mean(train_r2_scores),
            'std_train_R2': np.std(train_r2_scores),
            'mean_test_R2': np.mean(test_r2_scores),
            'std_test_R2': np.std(test_r2_scores),
            'predictions_data': predictions_data
        }

# ========================= Results Display Module =========================
class ResultsDisplay:
    """Results display class"""
    
    @staticmethod
    def plot_fold_error_comparison_by_model_type(stacking_cv_results, single_cv_results, model_type_name):
        """绘制特定模型类型的每一折堆叠模型和普通模型误差对比散点图"""
        # 创建5个子图
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Each Fold Error Comparison: {model_type_name} Stacked vs General Model', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
        
        # 根据模型类型找到对应的堆叠模型和单一模型
        stacking_model = None
        single_model = None
        
        # 查找堆叠模型
        for name, result in stacking_cv_results.items():
            if model_type_name in name or (model_type_name == 'GradientBoosting' and 'GradientBoosting' in name):
                stacking_model = (name, result)
                break
        
        # 查找单一模型
        for name, result in single_cv_results.items():
            if model_type_name in name:
                single_model = (name, result)
                break
        
        # 检查是否找到了对应的模型
        if stacking_model is None:
            print(f"Warning: No stacking model found for {model_type_name}")
            return None
        if single_model is None:
            print(f"Warning: No single model found for {model_type_name}")
            return None
        
        # 检查是否有预测数据
        if 'predictions_data' not in stacking_model[1]:
            print(f"Warning: No predictions data found for stacking {model_type_name}")
            return None
        if 'predictions_data' not in single_model[1]:
            print(f"Warning: No predictions data found for single {model_type_name}")
            return None
        
        # 为每个折创建散点图
        for fold_idx in range(5):
            ax = axes[fold_idx]
            
            # 获取当前折的数据
            fold_num = fold_idx + 1
            
            # 堆叠模型数据
            stacking_fold_data = None
            for fold_data in stacking_model[1]['predictions_data']:
                if fold_data['fold'] == fold_num:
                    stacking_fold_data = fold_data
                    break
            
            # 单一模型数据
            single_fold_data = None
            for fold_data in single_model[1]['predictions_data']:
                if fold_data['fold'] == fold_num:
                    single_fold_data = fold_data
                    break
            
            if stacking_fold_data is None or single_fold_data is None:
                ax.text(0.5, 0.5, f'Fold {fold_num}\nData Not Available', 
                       ha='center', va='center', transform=ax.transAxes, fontfamily='Times New Roman', fontsize=14, fontweight='bold')
                continue
            
            # 提取测试集数据
            stacking_actual = stacking_fold_data['test_actual']
            stacking_predicted = stacking_fold_data['test_predicted']
            single_actual = single_fold_data['test_actual']
            single_predicted = single_fold_data['test_predicted']
            
            # 计算误差
            stacking_errors = stacking_predicted - stacking_actual
            single_errors = single_predicted - single_actual
            
            # 绘制散点图 - 实际值 vs 预测值
            ax.scatter(stacking_actual, stacking_predicted, 
                      alpha=0.6, s=50, c='blue', label=f'Stacked Model', 
                      marker='o', edgecolors='darkblue', linewidth=0.5)
            
            ax.scatter(single_actual, single_predicted, 
                      alpha=0.6, s=50, c='red', label=f'General Model', 
                      marker='^', edgecolors='darkred', linewidth=0.5)
            
            # 添加完美预测线 (y=x)
            min_val = min(min(stacking_actual), min(single_actual))
            max_val = max(max(stacking_actual), max(single_actual))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')
            
            # 计算并显示MAE
            stacking_mae = np.mean(np.abs(stacking_errors))
            single_mae = np.mean(np.abs(single_errors))
            
            # 计算合适的坐标轴范围
            val_range = max_val - min_val
            padding = max(0.15 * val_range, 0.1)
            x_min, x_max = min_val - padding, max_val + padding
            y_min, y_max = min_val - padding, max_val + padding
            
            # 设置图形属性
            ax.set_xlabel('Actual Workfunction', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
            ax.set_ylabel('Predicted Workfunction', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
            ax.set_title(f'Fold {fold_num}\nStacked MAE: {stacking_mae:.4f}\nGeneral MAE: {single_mae:.4f}', 
                        fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴刻度标签字体
            ax.tick_params(axis='both', which='major', labelsize=12, width=2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontfamily('Times New Roman')
            
            # 设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 图例
            legend = ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True, shadow=True, 
                              bbox_to_anchor=(0.99, 0.01), borderaxespad=0)
            for text in legend.get_texts():
                text.set_fontweight('bold')
                text.set_fontfamily('Times New Roman')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# ========================= Main Function =========================
def main():
    """Main experiment workflow - GradientBoosting only"""
    print("Starting GradientBoosting Model Experiment - 5-Fold Cross-Validation")
    print("="*70)
    
    # 1. Data loading and preprocessing
    print("\n1. Data Loading and Preprocessing")
    X_15, y = DataProcessor.load_and_preprocess(Config.DATA_PATH, n_features=15)
    
    if X_15 is None:
        print("Data loading failed, experiment terminated")
        return
    
    print(f"Dataset size: {X_15.shape}")
    
    # 2. Prepare fixed split
    print("\n2. Data Split Preparation")
    X_train_15, X_test_15, y_train, y_test = train_test_split(
        X_15, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    print(f"Training set size: {X_train_15.shape[0]}, Test set size: {X_test_15.shape[0]}")
    
    # 3. GradientBoosting stacking model evaluation
    print("\n3. GradientBoosting Stacking Model Evaluation")
    gb_config = ModelBuilder.get_gradient_boosting_stacking_config()
    print(f"Evaluating: {gb_config['name']}")
    
    stacking_cv_results = {}
    result = Evaluator.cross_validate_stacking_model_with_fixed_split(
        X_15, y, gb_config['base_models'], gb_config['meta_model'], 
        X_train_15, X_test_15, y_train, y_test, Config.CV_FOLDS
    )
    stacking_cv_results[gb_config['name']] = result
    
    # 4. GradientBoosting single model evaluation
    print("\n4. GradientBoosting Single Model Evaluation")
    gb_single_model = ModelBuilder.get_gradient_boosting_single_model()
    print("Evaluating: GradientBoosting")
    
    single_cv_results = {}
    single_cv_results['GradientBoosting'] = Evaluator.cross_validate_model_with_fixed_split(
        X_15, y, gb_single_model, X_train_15, X_test_15, y_train, y_test, Config.CV_FOLDS
    )
    
    # 5. Generate GradientBoosting scatter plot
    print("\n5. Generating GradientBoosting Scatter Plot")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("生成GradientBoosting模型的每一折误差对比散点图...")
    fold_comparison_fig = ResultsDisplay.plot_fold_error_comparison_by_model_type(
        stacking_cv_results, single_cv_results, 'GradientBoosting'
    )
    
    if fold_comparison_fig is not None:
        comparison_filename = f"MXenes_5FoldCV_GradientBoosting_Comparison_{timestamp}.png"
        fold_comparison_fig.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"    GradientBoosting散点图已保存: {comparison_filename}")
        plt.close(fold_comparison_fig)
    else:
        print("    警告: 未能生成GradientBoosting模型的对比图")
    
    # 6. Results summary
    print(f"\n6. Results Summary")
    print(f"GradientBoosting Stacking Model:")
    print(f"  - Mean Test MAE: {result['mean_test_MAE']:.4f}±{result['std_test_MAE']:.4f}")
    print(f"  - Mean Test R²: {result['mean_test_R2']:.4f}±{result['std_test_R2']:.4f}")
    
    single_result = single_cv_results['GradientBoosting']
    print(f"GradientBoosting Single Model:")
    print(f"  - Mean Test MAE: {single_result['mean_test_MAE']:.4f}±{single_result['std_test_MAE']:.4f}")
    print(f"  - Mean Test R²: {single_result['mean_test_R2']:.4f}±{single_result['std_test_R2']:.4f}")
    
    print(f"\nExperiment completed! Generated file: {comparison_filename}")

# Run main experiment
if __name__ == "__main__":
    main()
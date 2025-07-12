"""
MXenes材料功函数预测的机器学习实验
=====================================
本实验使用集成学习方法预测MXenes材料的功函数性质
"""

# ========================= 导入所需库 =========================
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
import lightgbm as lgb

# 忽略警告信息
warnings.filterwarnings('ignore')

# ========================= 配置参数 =========================
class Config:
    """实验配置类"""
    DATA_PATH = r'C:\Users\Shang\Desktop\实验二\数据文件\c2db_Mxenes+3d描述符.xlsx'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # 要删除的列
    DROP_COLUMNS = ['Formula', 'Workfunction', 'X', 'T', 'M']
    TARGET_COLUMN = 'Workfunction'

# ========================= 数据处理模块 =========================
class DataProcessor:
    """数据处理类"""
    
    @staticmethod
    def load_and_preprocess(file_path, n_features=None):
        """
        加载和预处理数据
        
        Args:
            file_path: 数据文件路径
            n_features: 使用的特征数量，None表示使用所有特征
            
        Returns:
            X: 特征矩阵
            y: 目标变量
        """
        try:
            data = pd.read_excel(file_path)
            print(f"数据加载成功，形状: {data.shape}")
            
            # 删除不需要的列
            X = data.drop(Config.DROP_COLUMNS, axis=1)
            y = data[Config.TARGET_COLUMN]
            
            # 选择特征数量
            if n_features is not None:
                X = X.iloc[:, :n_features]
                print(f"使用前 {n_features} 个特征")
            else:
                print(f"使用所有 {X.shape[1]} 个特征")
            
            return X, y
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None, None

# ========================= 模型构建模块 =========================
class ModelBuilder:
    """模型构建类"""
    
    @staticmethod
    def get_base_models():
        """获取基学习器"""
        return [
            ('extra_trees', ExtraTreesRegressor(n_estimators=100, random_state=Config.RANDOM_STATE)),
            ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
            ('random_forest', RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE)),
            ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=6)))
        ]
    
    @staticmethod
    def get_meta_models():
        """获取元学习器"""
        return [
            lgb.LGBMRegressor(num_leaves=32, learning_rate=0.05, n_estimators=74, verbose=-1),
            GradientBoostingRegressor(n_estimators=80, random_state=33, max_depth=2),  # 保持原始random_state=33
            RandomForestRegressor(n_estimators=56, random_state=Config.RANDOM_STATE)
        ]
    
    @staticmethod
    def get_stacking_configurations():
        """获取所有Stacking配置（保持与原始代码完全一致）"""
        configs = []
        
        # 配置1: 原始第一个配置
        configs.append({
            'base_models': [
                ('et_g', ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
                ('rf_g', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=6)))
            ],
            'meta_model': lgb.LGBMRegressor(num_leaves=32, learning_rate=0.05, n_estimators=74, verbose=-1),
            'name': 'LightGBM元学习器'
        })
        
        # 配置2: 原始第二个配置
        configs.append({
            'base_models': [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2))),  # epsilon=0.2
                ('extra_trees', ExtraTreesRegressor(n_estimators=102, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)))  # n_neighbors=5
            ],
            'meta_model': GradientBoostingRegressor(n_estimators=80, random_state=33, max_depth=2),
            'name': 'GradientBoosting元学习器'
        })
        
        # 配置3: 原始第三个配置
        configs.append({
            'base_models': [
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=2)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
                ('extra_trees', ExtraTreesRegressor(n_estimators=94, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9)))  # n_neighbors=9
            ],
            'meta_model': RandomForestRegressor(n_estimators=56, random_state=42),
            'name': 'RandomForest元学习器'
        })
        
        return configs
    
    @staticmethod
    def get_stacking_configurations_15():
        """获取15特征的Stacking配置（与原始代码configurations_2对应）"""
        configs = []
        
        # 配置1: 15特征版本1
        configs.append({
            'base_models': [
                ('et_g', ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
                ('rf_g', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=6)))
            ],
            'meta_model': lgb.LGBMRegressor(num_leaves=32, learning_rate=0.05, n_estimators=60, verbose=-1),  # n_estimators=60
            'name': 'LightGBM元学习器(15特征)'
        })
        
        # 配置2: 15特征版本2
        configs.append({
            'base_models': [
                ('et_g', ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
                ('rf_g', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=6)))
            ],
            'meta_model': GradientBoostingRegressor(n_estimators=50, random_state=42),  # n_estimators=50
            'name': 'GradientBoosting元学习器(15特征)'
        })
        
        # 配置3: 15特征版本3
        configs.append({
            'base_models': [
                ('gb', GradientBoostingRegressor(n_estimators=60, random_state=42, max_depth=2)),  # n_estimators=60
                ('svr', make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3))),
                ('extra_trees', ExtraTreesRegressor(n_estimators=900, random_state=42)),  # n_estimators=900
                ('knn', make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9)))
            ],
            'meta_model': RandomForestRegressor(n_estimators=40, random_state=42),  # n_estimators=40
            'name': 'RandomForest元学习器(15特征)'
        })
        
        return configs
    
    @staticmethod
    def get_single_models():
        """获取单一模型列表"""
        return {
            'LightGBM': lgb.LGBMRegressor(num_leaves=32, learning_rate=0.05, n_estimators=180, verbose=-1),
            'RandomForest': RandomForestRegressor(n_estimators=84, random_state=Config.RANDOM_STATE),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=165, random_state=Config.RANDOM_STATE)
        }

# ========================= 实验评估模块 =========================
class Evaluator:
    """模型评估类"""
    
    @staticmethod
    def evaluate_stacking_model(X_train, X_test, y_train, y_test, base_models, meta_model, cv=5):
        """
        评估Stacking模型
        
        Args:
            X_train, X_test, y_train, y_test: 训练测试数据
            base_models: 基学习器列表
            meta_model: 元学习器
            cv: 交叉验证折数
            
        Returns:
            训练和测试性能指标
        """
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=cv)
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算指标
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
    def cross_validate_model(X, y, model, cv_folds=5):
        """
        交叉验证评估单一模型
        
        Args:
            X, y: 特征和目标变量
            model: 待评估模型
            cv_folds: 交叉验证折数
            
        Returns:
            平均性能指标
        """
        kf = KFold(n_splits=cv_folds, random_state=Config.RANDOM_STATE, shuffle=True)
        
        train_maes, train_r2s = [], []
        test_maes, test_r2s = [], []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 计算指标
            train_maes.append(mean_absolute_error(y_train, y_train_pred))
            train_r2s.append(r2_score(y_train, y_train_pred))
            test_maes.append(mean_absolute_error(y_test, y_test_pred))
            test_r2s.append(r2_score(y_test, y_test_pred))
        
        return {
            'train_MAE': np.mean(train_maes),
            'train_R2': np.mean(train_r2s),
            'test_MAE': np.mean(test_maes),
            'test_R2': np.mean(test_r2s)
        }

# ========================= 结果展示模块 =========================
class ResultsDisplay:
    """结果展示类"""
    
    @staticmethod
    def print_all_results(stacking_all_results, stacking_15_results, cv_results):
        """统一打印所有9个实验结果"""
        print(f"\n{'='*100}")
        print(f"{'MXenes功函数预测实验 - 完整结果汇总 (共9个结果)':^100}")
        print(f"{'='*100}")
        
        # Stacking实验结果 - 18个特征
        print(f"\n【第一组：Stacking集成学习实验 - 使用全部18个特征】")
        print("-" * 80)
        for i, result in enumerate(stacking_all_results, 1):
            print(f"{i}. {result['name']}:")
            print(f"   训练集: MAE={result['train']['MAE']:.4f}, R²={result['train']['R2']:.4f}")
            print(f"   测试集: MAE={result['test']['MAE']:.4f}, R²={result['test']['R2']:.4f}")
            print()
        
        # Stacking实验结果 - 15个特征
        print(f"【第二组：Stacking集成学习实验 - 使用前15个特征】")
        print("-" * 80)
        for i, result in enumerate(stacking_15_results, 4):
            print(f"{i}. {result['name']}:")
            print(f"   训练集: MAE={result['train']['MAE']:.4f}, R²={result['train']['R2']:.4f}")
            print(f"   测试集: MAE={result['test']['MAE']:.4f}, R²={result['test']['R2']:.4f}")
            print()
        
        # 交叉验证结果
        print(f"【第三组：单一模型交叉验证实验 - 使用前15个特征】")
        print("-" * 80)
        counter = 7
        for model_name, metrics in cv_results.items():
            print(f"{counter}. {model_name}模型 (5折交叉验证):")
            print(f"   平均训练集: MAE={metrics['train_MAE']:.4f}, R²={metrics['train_R2']:.4f}")
            print(f"   平均测试集: MAE={metrics['test_MAE']:.4f}, R²={metrics['test_R2']:.4f}")
            print()
            counter += 1
        
        # 性能排序总结
        print(f"{'='*100}")
        print(f"{'性能总结 (按测试集MAE排序)':^100}")
        print(f"{'='*100}")
        
        all_results = []
        # 收集所有结果
        for i, result in enumerate(stacking_all_results, 1):
            all_results.append((f"{i}. {result['name']}", result['test']['MAE'], result['test']['R2']))
        
        for i, result in enumerate(stacking_15_results, 4):
            all_results.append((f"{i}. {result['name']}", result['test']['MAE'], result['test']['R2']))
        
        counter = 7
        for model_name, metrics in cv_results.items():
            all_results.append((f"{counter}. {model_name}模型", metrics['test_MAE'], metrics['test_R2']))
            counter += 1
        
        # 按MAE排序
        all_results.sort(key=lambda x: x[1])
        
        print("排名   模型名称                              测试MAE    测试R²")
        print("-" * 80)
        for rank, (name, mae, r2) in enumerate(all_results, 1):
            print(f"{rank:2d}.   {name:<35} {mae:.4f}    {r2:.4f}")
        
        print(f"\n{'='*100}")
        print(f"实验完成！总共9个模型配置，最佳模型: {all_results[0][0]} (MAE={all_results[0][1]:.4f})")
        print(f"{'='*100}")
    
    @staticmethod
    def print_stacking_results(results, experiment_name="Stacking实验"):
        """打印Stacking实验结果"""
        print(f"\n{'='*50}")
        print(f"{experiment_name}结果")
        print(f"{'='*50}")
        
        for idx, result in enumerate(results):
            print(f"配置 {idx+1}: "
                  f"训练MAE={result['train']['MAE']:.4f}, "
                  f"训练R²={result['train']['R2']:.4f}, "
                  f"测试MAE={result['test']['MAE']:.4f}, "
                  f"测试R²={result['test']['R2']:.4f}")
    
    @staticmethod
    def print_cv_results(results, experiment_name="交叉验证实验"):
        """打印交叉验证结果"""
        print(f"\n{'='*50}")
        print(f"{experiment_name}结果")
        print(f"{'='*50}")
        
        for model_name, metrics in results.items():
            print(f"{model_name}: "
                  f"平均测试MAE={metrics['test_MAE']:.4f}, "
                  f"平均测试R²={metrics['test_R2']:.4f}, "
                  f"平均训练MAE={metrics['train_MAE']:.4f}, "
                  f"平均训练R²={metrics['train_R2']:.4f}")

# ========================= 主实验流程 =========================
def main():
    """主实验流程"""
    print("开始MXenes材料功函数预测实验")
    print("="*60)
    
    # 1. 数据加载和预处理
    print("\n1. 数据加载和预处理")
    X_all, y = DataProcessor.load_and_preprocess(Config.DATA_PATH)
    X_15, _ = DataProcessor.load_and_preprocess(Config.DATA_PATH, n_features=15)
    
    if X_all is None or X_15 is None:
        print("数据加载失败，实验终止")
        return
    
    # 2. 数据分割
    print("\n2. 数据分割")
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    X_train_15, X_test_15, _, _ = train_test_split(
        X_15, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    print(f"训练集大小: {X_train_all.shape[0]}, 测试集大小: {X_test_all.shape[0]}")
    
    # 3. Stacking实验 - 使用18个特征（完全按原始配置）
    print("\n3. 运行Stacking实验...")
    stacking_configs_all = ModelBuilder.get_stacking_configurations()
    stacking_results_all = []
    
    for config in stacking_configs_all:
        train_metrics, test_metrics = Evaluator.evaluate_stacking_model(
            X_train_all, X_test_all, y_train, y_test, config['base_models'], config['meta_model']
        )
        stacking_results_all.append({
            'name': config['name'],
            'train': train_metrics,
            'test': test_metrics
        })
    
    # 4. Stacking实验 - 使用15个特征（完全按原始configurations_2）
    stacking_configs_15 = ModelBuilder.get_stacking_configurations_15()
    stacking_results_15 = []
    
    for config in stacking_configs_15:
        train_metrics, test_metrics = Evaluator.evaluate_stacking_model(
            X_train_15, X_test_15, y_train, y_test, config['base_models'], config['meta_model']
        )
        stacking_results_15.append({
            'name': config['name'],
            'train': train_metrics,
            'test': test_metrics
        })
    
    # 5. 单一模型交叉验证实验
    single_models = ModelBuilder.get_single_models()
    cv_results = {}
    
    for model_name, model in single_models.items():
        cv_results[model_name] = Evaluator.cross_validate_model(X_15, y, model, Config.CV_FOLDS)
    
    # 统一展示所有9个结果
    ResultsDisplay.print_all_results(stacking_results_all, stacking_results_15, cv_results)

# 运行主实验
if __name__ == "__main__":
    main()
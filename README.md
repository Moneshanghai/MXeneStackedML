# 🧬 MXenes Material Work Function Prediction Research Project

<div align="center">

![Project Status](https://img.shields.io/badge/Project-Active-brightgreen?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-Academic-orange?style=for-the-badge)
![AI/ML](https://img.shields.io/badge/AI/ML-SISSO+Stacking+SHAP-red?style=for-the-badge)

</div>

---

## 🎯 Project Overview

> **🔬 Scientific Objective**: Predict work function properties of MXenes materials through advanced machine learning techniques, **SISSO** algorithm, **Stacking Models**, and **SHAP** interpretability analysis

This project integrates **SISSO** (Sure Independence Screening and Sparsifying Operator) algorithm with multiple **machine learning stacking models**, focusing on work function prediction research for MXenes materials. The project adopts a multi-level modeling strategy, from feature engineering through **stacking model** ensembles to **SHAP**-based interpretability analysis, forming a complete material property prediction workflow.

### 🌟 Core Innovations
- 🎛️ **Multi-parameter SISSO Optimization**: 7 different configuration groups of SISSO experiments
- 🤖 **Stacking Model Architecture**: Intelligent combination of 6 algorithms using advanced stacking techniques
- 📊 **Comprehensive Interpretability**: Deep analysis based on **SHAP** (SHapley Additive exPlanations)
- 🔄 **Robust Validation System**: 5-fold cross-validation ensuring reliability

---

## 📁 Project Structure

```
🗂️ Repository Integration/
├── 🧮 SISSSO/                     # **SISSO** Algorithm Core Module
│   ├── 🔬 sisso-f-1/             # Experiment Config 1 - Basic Parameters
│   ├── 🔬 sisso-f-2/             # Experiment Config 2 - Feature Optimization
│   ├── 🔬 sisso-f-3/             # Experiment Config 3 - Extended Operators
│   ├── 🔬 sisso-f-4/             # Experiment Config 4 - Complexity Tuning
│   ├── 🔬 sisso-f-5/             # Experiment Config 5 - Dimension Optimization
│   ├── 🔬 sisso-f-6/             # Experiment Config 6 - Regularization
│   └── 🔬 sisso-f-7/             # Experiment Config 7 - Ensemble Strategy
├── 💻 code/                       # Core Algorithm Implementation
│   ├── 🎨 SHAP/                  # **SHAP** Interpretability Analysis Module
│   │   └── 📓 SHAP.ipynb         # Interactive **SHAP** Analysis
│   ├── 🏗️ Stacked Model/        # **Stacking Model** Ensemble Learning
│   │   └── 🐍 Stacked Model.py   # Multi-algorithm **Stacking** Implementation
│   └── 📈 cv-GBoosting/          # Cross-Validation Gradient Boosting
│       └── 🚀 5 -cv-mae.py       # High-Performance Gradient Boosting
├── 📊 data/                       # Data Resource Center
│   ├── 📋 Original dataset.xlsx        # Original Complete Dataset
│   ├── 🧪 dataset Mxenes.xlsx          # MXenes-Specific Data
│   ├── 📁 classified-M/                # M-element Classified Data
│   └── 📁 classified-T/                # T-element Classified Data
└── 📖 README.md                   # Project Documentation
```

---

## ⚡ Technical Features

### 🎯 1. **SISSO** Algorithm Integration

| Feature | Description | Advantage |
|---------|-------------|----------|
| 🔄 **Multi-Configuration Experiments** | 7 different parameter settings for **SISSO** experiments | Comprehensive parameter space exploration |
| ⚙️ **Intelligent Feature Engineering** | Automated feature construction and selection via **SISSO** | Discover hidden feature relationships |
| 🧮 **Rich Operator Library** | 16 mathematical operators supported by **SISSO** | Build complex feature expressions |
| 📐 **Dimension Adaptive** | Configurable descriptor dimension optimization | Balance complexity and accuracy |

**🎨 Supported Operator Set for **SISSO**:**
```
➕ Addition (+)         ➖ Subtraction (-)      ✖️ Multiplication (*)   ➗ Division (/)
📈 Exponential (exp)    📉 Negative Exp (exp-)  🔄 Inverse (^-1)       ⬆️ Square (^2)
🔺 Cube (^3)           √️ Square Root (sqrt)  ∛ Cube Root (cbrt)    📊 Logarithm (log)
📏 Absolute (|-|)      📈 Sixth Power (^6)    〰️ Sine (sin)          〰️ Cosine (cos)
```

### 🤖 2. Machine Learning **Stacking Model** Matrix

#### 🏗️ **Stacking Model Ensemble Architecture**
```
🌳 Random Forest              ➡️  High accuracy, overfitting resistance
🌲 Extra Trees                ➡️  Enhanced randomness, reduced variance
🚀 Gradient Boosting          ➡️  Sequential optimization, complex relationships
🎯 Support Vector Regression   ➡️  Non-linear mapping, strong generalization
🏠 K-Nearest Neighbors        ➡️  Similarity-based, simple and effective
⚡ LightGBM                   ➡️  High efficiency, memory-friendly
```

#### 🔧 **Stacking Model Optimization Strategies**
- ✅ **5-Fold Cross-Validation**: Ensure **stacking model** stability and generalization
- 📏 **Standardization Preprocessing**: Feature normalization for improved convergence
- 🎛️ **Hyperparameter Tuning**: Grid search optimization for **stacking models**
- 📊 **Ensemble Weight Optimization**: Dynamic adjustment of each model's contribution in **stacking**

### 🔍 3. Model Interpretability Analysis with **SHAP**

| Analysis Dimension | Tool | Output Result |
|-------------------|------|---------------|
| 🎯 **Feature Importance** | **SHAP** Values | Global feature contribution ranking |
| 📈 **Local Explanation** | **SHAP** Waterfall | Individual sample prediction process |
| 📊 **Dependency Relationship** | **SHAP** Dependency | Feature interaction analysis |
| ⚡ **Impact Pattern** | **SHAP** Summary | Feature influence direction and magnitude |

---

## 🔧 Core Functionality Modules

### 🎛️ **SISSO** Configuration Center

<details>
<summary>📋 <strong>Click to Expand Detailed Configuration</strong></summary>

#### ⚙️ **Core Parameter Settings for SISSO**
```ini
🎯 Property Type (ptype)        = 1      # Regression prediction mode
🎪 Multi-task Learning (ntask)   = 1      # Single task configuration
⚖️ Task Weighting (task_weighting) = 1    # Equal weight processing
📐 Descriptor Dimension (desc_dim) = 1    # One-dimensional descriptor
📊 Sample Count (nsample)        = 275    # Total training samples
🔄 Restart Mode (restart)        = 0      # Train from scratch
```

#### 🧪 **Feature Engineering Parameters for SISSO**
```ini
🔢 Scalar Features (nsf)         = 15     # Base feature count
🎭 Operator Set (ops)            = '(+)(-)(*)(/)(exp)(^-1)(^2)(^3)'
🔺 Feature Complexity (fcomplexity) = 1    # Maximum operator combinations
📏 Minimum Threshold (fmax_min)   = 1e-3   # Feature filtering lower limit
📈 Maximum Threshold (fmax_max)   = 1e5    # Feature filtering upper limit
🎯 Selected Features (nf_sis)     = 800    # SIS selected feature count
```

</details>

### 💾 Data Processing Capabilities

#### 📊 **Multi-Source Data Integration**
- 📄 **Excel File Support**: Native Excel read/write, multi-worksheet support
- 🧹 **Intelligent Data Cleaning**: Automatic handling of missing values and outliers
- 🏗️ **Feature Engineering**: Domain knowledge-based feature construction
- 🏷️ **Classified Data Processing**: Independent modeling for M-type and T-type materials

#### 🎯 **Data Quality Assurance**
```
✅ Data Integrity Check      ✅ Feature Distribution Analysis  ✅ Correlation Detection
✅ Outlier Identification    ✅ Standardization Processing    ✅ Dimensionality Reduction
```

## Quick Start

### Environment Requirements
```bash
# Python Environment
Python >= 3.7

# Key Dependencies for SISSO, Stacking Models, and SHAP
pandas
numpy
scikit-learn
lightgbm
shap                    # For SHAP interpretability analysis
openpyxl
matplotlib
seaborn
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn lightgbm shap openpyxl matplotlib seaborn
```

### Running Examples

#### 1. Execute **Stacking Model** Prediction
```bash
cd code/Stacked\ Model/
python "Stacked Model.py"    # Run the stacking ensemble model
```

#### 2. Run Cross-Validation Gradient Boosting
```bash
cd code/cv-GBoosting/
python "5 -cv-mae.py"
```

#### 3. View **SHAP** Analysis
```bash
cd code/SHAP/
jupyter notebook SHAP.ipynb    # Interactive SHAP interpretability analysis
```

#### 4. Run **SISSO** Algorithm
```bash
cd SISSSO/sisso-f-1/
# Execute SISSO program (requires pre-installed SISSO software)
sisso
```

## 实验设计

### SISSO实验配置
每个SISSO子目录包含：
- `SISSO.in`：输入配置文件
- `SISSO.out`：输出结果文件
- `train.dat`：训练数据
- `models/`：生成的模型文件
- `SIS_subspaces/`：特征子空间
- `desc_dat/`：描述符数据

### 关键参数设置
- **样本数量**：275个训练样本
- **标量特征**：15个基础特征
- **算子集合**：8种数学运算符
- **特征复杂度**：最大复杂度为1
- **SIS特征数**：每个子空间800个特征

## 数据说明

### 数据集特点
- **原始数据集**：包含完整的MXenes材料信息
- **MXenes专用数据**：针对MXenes材料优化的数据集
- **分类数据**：按M元素和T元素分类的数据

### 特征描述
- **结构特征**：材料的几何结构参数
- **电子特征**：电子结构相关描述符
- **化学特征**：化学成分和键合信息
- **物理特征**：物理性质参数

## 评估指标

### 回归评估
- **平均绝对误差（MAE）**：预测精度的主要指标
- **决定系数（R²）**：模型拟合优度
- **交叉验证得分**：模型泛化能力评估

### 特征重要性
- **SHAP值**：每个特征的贡献度
- **特征权重**：模型中特征的重要性排序
- **相关性分析**：特征间的相关性关系

## 结果分析

### 模型性能
- **集成学习优势**：多模型组合提升预测精度
- **特征工程效果**：SISSO特征工程显著改善性能
- **交叉验证稳定性**：确保模型的泛化能力

### 科学发现
- **关键描述符识别**：发现影响功函数的关键因子
- **结构-性能关系**：揭示材料结构与功函数的关系
- **设计指导**：为新材料设计提供理论指导

## 使用注意事项

1. **数据路径**：请根据实际情况修改代码中的数据文件路径
2. **SISSO软件**：运行SISSO实验需要预先安装SISSO软件
3. **计算资源**：集成学习和交叉验证需要较多计算资源
4. **参数调优**：可根据具体需求调整模型参数

## 贡献与支持

### 项目维护
- 定期更新算法实现
- 优化模型性能
- 扩展数据集规模

### 问题反馈
如遇到问题或有改进建议，请通过以下方式联系：
- 检查代码注释和文档说明
- 验证数据文件路径和格式
- 确认依赖包版本兼容性

## 许可证

本项目用于学术研究目的，请在使用时遵循相关的学术规范和引用要求。

## 更新日志

- **v1.0**：初始版本，包含基础SISSO和机器学习模型
- **v1.1**：添加SHAP可解释性分析
- **v1.2**：优化集成学习模型性能
- **v1.3**：完善交叉验证和评估体系

---

*最后更新：2025年7月*
"# MXeneStackedML" 

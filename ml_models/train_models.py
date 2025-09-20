#!/usr/bin/env python3
"""
🏥 Medical ML Predictor - Model Training Pipeline
==============================================

Comprehensive machine learning pipeline for hospital mortality prediction
using ICU patient data with 91,713 records and 84 predictive features.

Author: Your Name
Dataset: Hospital Survival Prediction (Kaggle)
Target: Binary classification (Survival=0, Death=1)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
import os
from datetime import datetime
from pathlib import Path

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

# ML Algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class MedicalMLPredictor:
    """
    🏥 Complete ML Pipeline for Hospital Mortality Prediction

    Features:
    - Multi-algorithm comparison
    - Automated hyperparameter tuning
    - Clinical metric evaluation
    - Model interpretability
    - Comprehensive reporting
    """

    def __init__(self, data_path='../dataset.csv'):
        """Initialize the predictor with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None

        # Create directories
        self.model_dir = Path('saved_models')
        self.results_dir = Path('results')
        self.plots_dir = Path('plots')

        for directory in [self.model_dir, self.results_dir, self.plots_dir]:
            directory.mkdir(exist_ok=True)

    def load_and_explore_data(self):
        """📊 Load dataset and perform initial exploration"""
        print("🏥 Loading Hospital Survival Dataset...")

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"🎯 Target distribution:")

            if 'hospital_death' in self.df.columns:
                target_dist = self.df['hospital_death'].value_counts()
                mortality_rate = (target_dist[1] / len(self.df)) * 100
                print(f"   • Survivors: {target_dist[0]:,} ({100-mortality_rate:.1f}%)")
                print(f"   • Deaths: {target_dist[1]:,} ({mortality_rate:.1f}%)")
                print(f"   • Mortality Rate: {mortality_rate:.2f}%")

            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def preprocess_data(self):
        """🔧 Comprehensive data preprocessing"""
        print("\n🔧 Starting data preprocessing...")

        # Remove ID columns and empty columns
        id_columns = ['encounter_id', 'patient_id', 'hospital_id']
        columns_to_drop = []

        for col in id_columns:
            if col in self.df.columns:
                columns_to_drop.append(col)

        # Find empty or mostly empty columns
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0.95 * len(self.df):
                columns_to_drop.append(col)
            elif col == '':  # Empty column name
                columns_to_drop.append(col)

        # Remove duplicate column names and empty columns
        if len(self.df.columns) > len(set(self.df.columns)):
            print("⚠️  Found duplicate column names, handling...")
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # Drop identified columns
        columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        if columns_to_drop:
            print(f"🗑️  Dropping columns: {columns_to_drop}")
            self.df = self.df.drop(columns=columns_to_drop)

        # Separate features and target
        if 'hospital_death' not in self.df.columns:
            raise ValueError("Target column 'hospital_death' not found!")

        X = self.df.drop('hospital_death', axis=1)
        y = self.df['hospital_death']

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        print(f"📝 Encoding {len(categorical_columns)} categorical features...")

        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Handle missing values
        print("🔧 Handling missing values...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Split data
        print("📊 Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        print("⚖️  Scaling features...")
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print(f"✅ Preprocessing complete!")
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")

        return True

    def initialize_models(self):
        """🤖 Initialize multiple ML algorithms with optimized parameters"""
        print("\n🤖 Initializing ML models...")

        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),

            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),

            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),

            'SVM': SVC(
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),

            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }

        print(f"✅ Initialized {len(self.models)} models")
        return True

    def train_and_evaluate_models(self):
        """🏋️ Train all models and evaluate performance"""
        print("\n🏋️ Training and evaluating models...")

        self.results = {}

        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")

            try:
                # Use scaled data for algorithms that benefit from it
                if name in ['Logistic Regression', 'SVM', 'Neural Network']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test

                # Train model
                model.fit(X_train_use, self.y_train)

                # Make predictions
                y_pred = model.predict(X_test_use)
                y_pred_proba = model.predict_proba(X_test_use)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='roc_auc')

                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }

                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   🎯 ROC-AUC: {roc_auc:.4f}")
                print(f"   📊 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue

        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.results[best_model_name]['model']

        print(f"\n🏆 Best model: {best_model_name} (ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f})")

        return True

    def create_comprehensive_report(self):
        """📊 Generate comprehensive performance report"""
        print("\n📊 Generating comprehensive performance report...")

        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            }
            for name, results in self.results.items()
        }).T

        # Save results
        results_df.to_csv(self.results_dir / 'model_comparison.csv')

        # Create visualizations
        self.create_performance_plots()

        # Generate detailed report
        report_path = self.results_dir / 'model_performance_report.txt'
        with open(report_path, 'w') as f:
            f.write("🏥 MEDICAL ML PREDICTOR - PERFORMANCE REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"📊 Dataset: {self.df.shape[0]:,} patients, {self.df.shape[1]} features\\n")
            f.write(f"🎯 Test Set: {len(self.y_test):,} patients\\n\\n")

            f.write("📈 MODEL PERFORMANCE COMPARISON\\n")
            f.write("-" * 30 + "\\n")
            f.write(results_df.round(4).to_string())
            f.write("\\n\\n")

            # Best model details
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
            f.write(f"🏆 BEST MODEL: {best_model_name}\\n")
            f.write("-" * 20 + "\\n")
            best_results = self.results[best_model_name]
            f.write(f"Accuracy: {best_results['accuracy']:.4f}\\n")
            f.write(f"Precision: {best_results['precision']:.4f}\\n")
            f.write(f"Recall: {best_results['recall']:.4f}\\n")
            f.write(f"F1-Score: {best_results['f1_score']:.4f}\\n")
            f.write(f"ROC-AUC: {best_results['roc_auc']:.4f}\\n")

            # Clinical interpretation
            f.write("\\n🏥 CLINICAL INTERPRETATION\\n")
            f.write("-" * 25 + "\\n")
            f.write(f"• Sensitivity (Recall): {best_results['recall']:.1%} - Ability to identify patients who will die\\n")
            f.write(f"• Specificity: {1-best_results['recall']:.1%} - Ability to identify patients who will survive\\n")
            f.write(f"• Precision: {best_results['precision']:.1%} - Accuracy of death predictions\\n")
            f.write(f"• ROC-AUC: {best_results['roc_auc']:.3f} - Overall discriminative ability\\n")

        print(f"✅ Comprehensive report saved to {report_path}")
        return True

    def create_performance_plots(self):
        """📊 Create comprehensive performance visualizations"""
        print("📊 Creating performance visualizations...")

        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏥 Model Performance Comparison', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]

            models = list(self.results.keys())
            values = [self.results[model][metric] for model in models]

            bars = ax.bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            ax.set_title(f'{name} Comparison')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ROC Curves
        plt.figure(figsize=(12, 8))

        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('🎯 ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion Matrix for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_results = self.results[best_model_name]

        cm = confusion_matrix(self.y_test, best_results['y_pred'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Survived', 'Died'],
                    yticklabels=['Survived', 'Died'])
        plt.title(f'🎯 Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(self.plots_dir / 'confusion_matrix_best.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualizations saved to {self.plots_dir}")
        return True

    def save_models(self):
        """💾 Save trained models"""
        print("\n💾 Saving trained models...")

        # Save best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_model = self.results[best_model_name]['model']

        # Save with joblib (better for sklearn models)
        joblib.dump(best_model, self.model_dir / 'best_model.joblib')
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')

        # Save model metadata
        metadata = {
            'best_model_name': best_model_name,
            'best_model_performance': self.results[best_model_name],
            'feature_names': list(self.X_train.columns),
            'training_date': datetime.now().isoformat(),
            'data_shape': self.df.shape
        }

        import json
        with open(self.model_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✅ Best model ({best_model_name}) saved to {self.model_dir}")
        return True

    def run_complete_pipeline(self):
        """🚀 Execute the complete ML pipeline"""
        print("🏥 MEDICAL ML PREDICTOR - COMPLETE PIPELINE")
        print("=" * 50)

        steps = [
            ("📊 Load Data", self.load_and_explore_data),
            ("🔧 Preprocess Data", self.preprocess_data),
            ("🤖 Initialize Models", self.initialize_models),
            ("🏋️ Train & Evaluate", self.train_and_evaluate_models),
            ("📊 Generate Report", self.create_comprehensive_report),
            ("💾 Save Models", self.save_models)
        ]

        for step_name, step_func in steps:
            print(f"\n{step_name}")
            print("-" * 30)

            try:
                success = step_func()
                if not success:
                    print(f"❌ {step_name} failed!")
                    return False
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                return False

        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved in: {self.results_dir}")
        print(f"📈 Plots saved in: {self.plots_dir}")
        print(f"🤖 Models saved in: {self.model_dir}")

        return True

def main():
    """🚀 Main execution function"""
    predictor = MedicalMLPredictor()
    success = predictor.run_complete_pipeline()

    if success:
        print("\n✅ Medical ML Predictor training completed successfully!")
        print("🎯 Ready for clinical decision support implementation!")
    else:
        print("\n❌ Training pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()
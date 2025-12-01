import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_names = []
        
    def load_trained_models(self):
        """Load trained models and test data"""
        from utils import load_model
        
        # Load models
        self.best_model = load_model("models/best_model.pkl")
        self.feature_names = load_model("models/feature_names.pkl")
        
        # Load processed data
        df = pd.read_csv(self.config['data']['processed_data_path'])
        X = df[self.feature_names]
        y = df[self.config['features']['target']]
        
        # Split data (same random state as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['model']['test_size'], 
            random_state=self.config['model']['random_state'], stratify=y
        )
        
        print(f"📊 Loaded test data: {X_test.shape}")
        return X_test, y_test
    
    def generate_classification_report(self, y_true, y_pred, model_name="Best Model"):
        """Generate detailed classification report"""
        print(f"\n📋 CLASSIFICATION REPORT - {model_name.upper()}")
        print("=" * 50)
        
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Best Model"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Best Model"):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="Best Model"):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = np.mean(precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/precision_recall_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_precision
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            plt.title("Feature Importance")
            plt.barh(range(min(top_n, len(indices))), 
                    importances[indices][:top_n][::-1], 
                    color='skyblue', align='center')
            plt.yticks(range(min(top_n, len(indices))), 
                      [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            return feature_importance_df
        else:
            print("⚠️ Model doesn't have feature_importances_ attribute")
            return None
    
    def generate_comprehensive_evaluation(self, model, X_test, y_test, model_name="Best Model"):
        """Generate comprehensive evaluation for a single model"""
        print(f"\n🔍 Evaluating {model_name}...")
        print("=" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print(f"📊 Basic Metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        # Generate detailed reports and plots
        classification_rep = self.generate_classification_report(y_test, y_pred, model_name)
        confusion_mat = self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        if y_pred_proba is not None:
            roc_auc = self.plot_roc_curve(y_test, y_pred_proba, model_name)
            avg_precision = self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
            metrics['roc_auc'] = roc_auc
            metrics['avg_precision'] = avg_precision
            print(f"   ROC AUC: {roc_auc:.4f}")
            print(f"   Average Precision: {avg_precision:.4f}")
        
        # Feature importance
        feature_importance_df = self.plot_feature_importance(model, self.feature_names)
        
        evaluation_results = {
            'metrics': metrics,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'feature_importance': feature_importance_df
        }
        
        return evaluation_results
    
    def create_model_comparison_report(self, all_models_results):
        """Create comparison report for all trained models"""
        comparison_data = []
        
        for model_name, results in all_models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'CV Score': results['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Plot model comparison
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + i*width, comparison_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def generate_final_report(self, evaluation_results, comparison_df):
        """Generate final evaluation report"""
        print("\n" + "=" * 60)
        print("🎯 FINAL MODEL EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n🏆 BEST MODEL PERFORMANCE:")
        best_metrics = evaluation_results['metrics']
        for metric, value in best_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n📈 MODEL COMPARISON:")
        print(comparison_df.round(4))
        
        # Business insights
        print(f"\n💡 BUSINESS INSIGHTS:")
        cm = evaluation_results['confusion_matrix']
        total = cm.sum()
        
        print(f"   • Correct predictions: {(cm[0,0] + cm[1,1]) / total * 100:.1f}%")
        print(f"   • Survival correctly identified: {cm[1,1] / cm[1,:].sum() * 100:.1f}%")
        print(f"   • Non-survival correctly identified: {cm[0,0] / cm[0,:].sum() * 100:.1f}%")
        
        if evaluation_results['feature_importance'] is not None:
            top_features = evaluation_results['feature_importance'].head(5)
            print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
            for _, row in top_features.iterrows():
                print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    def run_complete_evaluation(self, all_models_results):
        """Execute complete model evaluation pipeline"""
        print("🚀 Starting Model Evaluation Pipeline...")
        print("=" * 50)
        
        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)
        
        # Load test data and best model
        X_test, y_test = self.load_trained_models()
        
        # Evaluate best model comprehensively
        evaluation_results = self.generate_comprehensive_evaluation(
            self.best_model, X_test, y_test, "Best Model"
        )
        
        # Create model comparison
        comparison_df = self.create_model_comparison_report(all_models_results)
        
        # Generate final report
        self.generate_final_report(evaluation_results, comparison_df)
        
        print("=" * 50)
        print("🎉 Model Evaluation Completed!")
        print("📊 Check 'results/' folder for all visualizations and reports")
        
        return evaluation_results, comparison_df

# Usage example
if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    evaluator = ModelEvaluator(config)
    
    # You would typically pass the results from model training here
    # evaluation_results, comparison_df = evaluator.run_complete_evaluation(all_models_results)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ExploratoryAnalysis:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.target = config['features']['target']
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data for analysis"""
        self.df = pd.read_csv(self.config['data']['cleaned_data_path'])
        print(f"📊 Loaded cleaned data: {self.df.shape}")
        return self.df
    
    def create_summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        print("📈 Generating Summary Statistics...")
        
        # Basic statistics
        summary = self.df.describe(include='all').T
        summary['missing'] = self.df.isnull().sum()
        summary['missing_pct'] = (self.df.isnull().sum() / len(self.df)) * 100
        summary['dtype'] = self.df.dtypes
        summary['unique'] = self.df.nunique()
        
        return summary
    
    def plot_target_distribution(self):
        """Plot distribution of target variable"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        self.df[self.target].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Variable Distribution')
        plt.xlabel('Survived')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.pie(self.df[self.target].value_counts(), 
                labels=self.df[self.target].value_counts().index,
                autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
        plt.title('Target Variable Proportion')
        
        plt.tight_layout()
        plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_numerical_distributions(self):
        """Plot distributions for all numerical features"""
        numerical_cols = self.config['features']['numerical']
        n_cols = 2
        n_rows = (len(numerical_cols) + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if col in self.df.columns:
                # Histogram with KDE
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                axes[i].legend()
        
        # Remove empty subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('results/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_categorical_distributions(self):
        """Plot distributions for categorical features"""
        categorical_cols = self.config['features']['categorical']
        n_cols = 2
        n_rows = (len(categorical_cols) + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                bars = axes[i].bar(value_counts.index.astype(str), value_counts.values, 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('results/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numerical features"""
        numerical_cols = self.config['features']['numerical'] + [self.target]
        numerical_df = self.df[numerical_cols]
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = numerical_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def plot_target_vs_features(self):
        """Plot relationships between target and features"""
        numerical_cols = self.config['features']['numerical']
        categorical_cols = self.config['features']['categorical']
        
        # Numerical features vs target
        n_numerical = len(numerical_cols)
        n_cols = 2
        n_rows = (n_numerical + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if col in self.df.columns:
                self.df.boxplot(column=col, by=self.target, ax=axes[i])
                axes[i].set_title(f'{col} by {self.target}')
        
        # Remove empty subplots
        for i in range(n_numerical, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('results/numerical_vs_target.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Categorical features vs target
        n_categorical = len(categorical_cols)
        n_cols = 2
        n_rows = (n_categorical + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if col in self.df.columns:
                pd.crosstab(self.df[col], self.df[self.target], normalize='index').plot(
                    kind='bar', ax=axes[i], color=['lightcoral', 'skyblue'])
                axes[i].set_title(f'Survival Rate by {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Proportion')
                axes[i].legend(['Died', 'Survived'])
                axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(n_categorical, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('results/categorical_vs_target.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self) -> Dict:
        """Generate key insights from EDA"""
        insights = {
            'data_quality': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'duplicate_rows': self.df.duplicated().sum()
            },
            'target_analysis': {
                'survival_rate': self.df[self.target].mean(),
                'class_imbalance': abs(self.df[self.target].mean() - 0.5)
            },
            'key_correlations': {},
            'important_findings': []
        }
        
        # Calculate correlations with target
        for col in self.config['features']['numerical']:
            if col in self.df.columns and col != self.target:
                corr = self.df[col].corr(self.df[self.target])
                insights['key_correlations'][col] = corr
        
        # Generate findings
        if 'Sex' in self.df.columns:
            survival_by_sex = self.df.groupby('Sex')[self.target].mean()
            insights['important_findings'].append(
                f"Gender impact: Female survival rate: {survival_by_sex.get('female', 0):.1%}, "
                f"Male survival rate: {survival_by_sex.get('male', 0):.1%}"
            )
        
        if 'Pclass' in self.df.columns:
            survival_by_class = self.df.groupby('Pclass')[self.target].mean()
            insights['important_findings'].append(
                f"Class impact: 1st class survival: {survival_by_class.get(1, 0):.1%}, "
                f"3rd class survival: {survival_by_class.get(3, 0):.1%}"
            )
        
        return insights
    
    def run_complete_analysis(self):
        """Execute complete EDA pipeline"""
        print("🔍 Starting Exploratory Data Analysis...")
        print("=" * 50)
        
        # Load data
        self.load_cleaned_data()
        
        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)
        
        # Generate plots and analysis
        summary = self.create_summary_statistics()
        self.plot_target_distribution()
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        correlation_matrix = self.plot_correlation_heatmap()
        self.plot_target_vs_features()
        
        # Generate insights
        insights = self.generate_insights()
        
        # Print insights
        print("\n📊 KEY INSIGHTS:")
        print("=" * 30)
        for finding in insights['important_findings']:
            print(f"• {finding}")
        
        print(f"\n🎯 Survival Rate: {insights['target_analysis']['survival_rate']:.1%}")
        print(f"📏 Dataset Size: {insights['data_quality']['total_rows']} passengers")
        
        print("\n📈 Top Correlations with Survival:")
        for feature, corr in sorted(insights['key_correlations'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:3]:
            print(f"• {feature}: {corr:.3f}")
        
        print("=" * 50)
        print("🎉 EDA Completed! Check 'results/' folder for visualizations.")
        
        return insights

# Usage example
if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    eda = ExploratoryAnalysis(config)
    insights = eda.run_complete_analysis()
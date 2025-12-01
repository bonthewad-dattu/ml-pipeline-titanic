import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.cleaning_report = {}
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        \"\"\"Load raw data from CSV file\"\"\"
        if filepath is None:
            filepath = self.config['data']['raw_data_path']
        
        self.df = pd.read_csv(filepath)
        print(f'✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns')
        return self.df
    
    def get_data_info(self) -> dict:
        \"\"\"Get comprehensive data information\"\"\"
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        return info
    
    def handle_missing_values(self) -> pd.DataFrame:
        \"\"\"Handle missing values based on data type and percentage\"\"\"
        missing_before = self.df.isnull().sum().sum()
        
        # Calculate missing percentage for each column
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        
        # Handle numerical columns
        numerical_cols = self.config['features']['numerical']
        for col in numerical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                if self.df[col].skew() > 1:  # Use median for skewed data
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f'📊 Filled missing {col} with median: {self.df[col].median():.2f}')
                else:  # Use mean for normal distribution
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f'📊 Filled missing {col} with mean: {self.df[col].mean():.2f}')
        
        # Handle categorical columns
        categorical_cols = self.config['features']['categorical']
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
                print(f'📝 Filled missing {col} with mode: {self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'}')
        
        missing_after = self.df.isnull().sum().sum()
        print(f'✅ Missing values handled: {missing_before} → {missing_after}')
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        \"\"\"Remove duplicate rows\"\"\"
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_after = self.df.duplicated().sum()
        
        print(f'✅ Duplicates removed: {duplicates_before} → {duplicates_after}')
        return self.df
    
    def handle_outliers_iqr(self, column: str) -> pd.DataFrame:
        \"\"\"Handle outliers using IQR method\"\"\"
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config['cleaning']['iqr_multiplier'] * IQR
        upper_bound = Q3 + self.config['cleaning']['iqr_multiplier'] * IQR
        
        outliers_before = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
        
        # Cap outliers instead of removing
        self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
        self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])
        
        outliers_after = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
        
        print(f'📏 Outliers handled in {column}: {outliers_before} → {outliers_after}')
        return self.df
    
    def handle_all_outliers(self) -> pd.DataFrame:
        \"\"\"Handle outliers for all numerical columns\"\"\"
        numerical_cols = self.config['features']['numerical']
        
        for col in numerical_cols:
            if col in self.df.columns:
                self.handle_outliers_iqr(col)
        
        return self.df
    
    def fix_data_types(self) -> pd.DataFrame:
        \"\"\"Convert columns to correct data types\"\"\"
        type_conversions = {
            'Pclass': 'category',
            'Sex': 'category',
            'Embarked': 'category'
        }
        
        for col, dtype in type_conversions.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype)
                print(f'🔄 Converted {col} to {dtype}')
        
        return self.df
    
    def clean_data(self, filepath: str = None) -> pd.DataFrame:
        \"\"\"Execute complete data cleaning pipeline\"\"\"
        print('🚀 Starting Data Cleaning Pipeline...')
        print('=' * 50)
        
        # Load data
        self.load_data(filepath)
        
        # Get initial info
        initial_info = self.get_data_info()
        print(f'📊 Initial data shape: {initial_info['shape']}')
        
        # Execute cleaning steps
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_all_outliers()
        self.fix_data_types()
        
        # Save cleaned data
        output_path = self.config['data']['cleaned_data_path']
        self.df.to_csv(output_path, index=False)
        print(f'💾 Cleaned data saved to: {output_path}')
        
        # Final report
        final_info = self.get_data_info()
        print('=' * 50)
        print('🎉 Data Cleaning Completed!')
        print(f'📈 Final data shape: {final_info['shape']}')
        
        return self.df

# Usage example
if __name__ == '__main__':
    from utils import load_config
    config = load_config()
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean_data()

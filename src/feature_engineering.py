import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.feature_names = []
        
    def load_cleaned_data(self) -> pd.DataFrame:
        \"\"\"Load cleaned data for feature engineering\"\"\"
        self.df = pd.read_csv(self.config['data']['cleaned_data_path'])
        print(f'📊 Loaded cleaned data for feature engineering: {self.df.shape}')
        return self.df
    
    def create_new_features(self) -> pd.DataFrame:
        \"\"\"Create new engineered features\"\"\"
        print('🛠️ Creating New Features...')
        
        # Family size feature
        if all(col in self.df.columns for col in ['SibSp', 'Parch']):
            self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
            self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
            print('✅ Created family-related features')
        
        # Age groups
        if 'Age' in self.df.columns:
            bins = [0, 12, 18, 35, 60, 100]
            labels = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']
            self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=bins, labels=labels)
            print('✅ Created age groups')
        
        return self.df
    
    def handle_categorical_encoding(self) -> pd.DataFrame:
        \"\"\"Encode categorical variables\"\"\"
        print('🔤 Encoding Categorical Features...')
        
        # One-Hot Encoding
        ohe_features = ['Embarked', 'AgeGroup']
        for feature in ohe_features:
            if feature in self.df.columns:
                dummies = pd.get_dummies(self.df[feature], prefix=feature)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(feature, axis=1, inplace=True)
                print(f'✅ One-Hot Encoded: {feature}')
        
        # Label Encoding for binary categorical features
        le_features = ['Sex']
        for feature in le_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[feature] = le.fit_transform(self.df[feature])
                print(f'✅ Label Encoded: {feature}')
        
        return self.df
    
    def handle_numerical_scaling(self) -> pd.DataFrame:
        \"\"\"Scale numerical features\"\"\"
        print('⚖️ Scaling Numerical Features...')
        
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        
        for feature in numerical_features:
            if feature in self.df.columns:
                scaler = StandardScaler()
                self.df[feature] = scaler.fit_transform(self.df[[feature]])
                print(f'✅ Scaled: {feature}')
        
        return self.df
    
    def remove_unnecessary_features(self) -> pd.DataFrame:
        \"\"\"Remove features that won't be used in modeling\"\"\"
        columns_to_remove = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)
                print(f'🗑️ Removed: {col}')
        
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        \"\"\"Execute complete feature engineering pipeline\"\"\"
        print('🚀 Starting Feature Engineering Pipeline...')
        print('=' * 50)
        
        # Load data
        self.load_cleaned_data()
        
        # Execute feature engineering steps
        self.create_new_features()
        self.remove_unnecessary_features()
        self.handle_categorical_encoding()
        self.handle_numerical_scaling()
        
        # Save processed data
        output_path = self.config['data']['processed_data_path']
        self.df.to_csv(output_path, index=False)
        
        # Save feature names
        self.feature_names = [col for col in self.df.columns if col != self.config['features']['target']]
        
        print('=' * 50)
        print('🎉 Feature Engineering Completed!')
        print(f'📈 Final feature set: {len(self.feature_names)} features')
        print(f'💾 Processed data saved to: {output_path}')
        
        return self.df

# Usage example
if __name__ == '__main__':
    from utils import load_config
    config = load_config()
    engineer = FeatureEngineer(config)
    engineered_data = engineer.engineer_features()

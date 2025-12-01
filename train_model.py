"""
TRAIN MODEL SCRIPT - COMPLETE VERSION
Trains and saves the Titanic survival prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import os

print("🚀 ML MODEL TRAINING STARTING")
print("=" * 50)

class TitanicMLPipeline:
    def __init__(self):
        self.df = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.best_model_name = ""
        self.best_score = 0
        self.feature_names = []
        
    def _create_sample_data(self):
        """Create sample Titanic data if file not found"""
        print("   Creating sample Titanic dataset...")
        np.random.seed(42)
        n_samples = 891
        
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.randint(0, 2, n_samples),
            'Pclass': np.random.randint(1, 4, n_samples),
            'Name': [f'Passenger {i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.normal(30, 15, n_samples).clip(0, 80),
            'SibSp': np.random.randint(0, 4, n_samples),
            'Parch': np.random.randint(0, 4, n_samples),
            'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
            'Fare': np.random.exponential(50, n_samples).clip(0, 500),
            'Cabin': np.random.choice([None, 'C123', 'B45', 'D56'], n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
        }
        
        self.df = pd.DataFrame(data)
        # Introduce some missing values for realism
        self.df.loc[self.df.sample(frac=0.2).index, 'Age'] = np.nan
        self.df.loc[self.df.sample(frac=0.05).index, 'Embarked'] = np.nan
        
        # Add realistic survival patterns
        self.df.loc[(self.df['Sex'] == 'female') & (self.df['Pclass'] == 1), 'Survived'] = np.random.choice(
            [0, 1], 
            len(self.df[(self.df['Sex'] == 'female') & (self.df['Pclass'] == 1)]), 
            p=[0.1, 0.9]
        )
        self.df.loc[(self.df['Sex'] == 'male') & (self.df['Pclass'] == 3), 'Survived'] = np.random.choice(
            [0, 1], 
            len(self.df[(self.df['Sex'] == 'male') & (self.df['Pclass'] == 3)]), 
            p=[0.8, 0.2]
        )
        
        print(f"   Created sample dataset with {len(self.df)} records")
        
    def load_and_clean_data(self):
        """STEP 1: Data Loading and Cleaning"""
        print("\n📊 STEP 1: Loading and Cleaning Data...")
        
        # Load dataset
        try:
            self.df = pd.read_csv('titanic.csv')
            print(f"   ✅ Loaded Titanic dataset with {len(self.df)} records")
        except FileNotFoundError:
            print("   ⚠️  titanic.csv not found, using sample data...")
            self._create_sample_data()
        
        # Data Quality Report
        print("\n   📋 DATA QUALITY REPORT:")
        print(f"   Total Records: {len(self.df)}")
        print(f"   Features: {list(self.df.columns)}")
        
        # Handle Missing Values
        print("\n   🧹 HANDLING MISSING VALUES:")
        
        # Age - median imputation
        age_median = self.df['Age'].median()
        self.df['Age'].fillna(age_median, inplace=True)
        age_missing = self.df['Age'].isnull().sum()
        print(f"     Age: Filled {age_missing} missing values with median {age_median:.1f}")
        
        # Embarked - mode imputation
        embarked_mode = self.df['Embarked'].mode()[0]
        self.df['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"     Embarked: Filled missing values with mode '{embarked_mode}'")
        
        # Fare - median imputation
        fare_median = self.df['Fare'].median()
        self.df['Fare'].fillna(fare_median, inplace=True)
        print(f"     Fare: Filled missing values with median {fare_median:.2f}")
        
        # Create HasCabin feature
        self.df['HasCabin'] = self.df['Cabin'].notna().astype(int)
        
        # Remove Duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_count - len(self.df)
        print(f"     Removed {duplicates_removed} duplicate records")
        
        # Handle Outliers
        print("\n   📊 HANDLING OUTLIERS (IQR METHOD):")
        numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            print(f"     {col}: Found {outliers_count} outliers")
            
            # Cap outliers
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
        
        print(f"\n   ✅ CLEANING COMPLETED")
        print(f"   Final dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
        
    def exploratory_data_analysis(self):
        """STEP 2: Exploratory Data Analysis"""
        print("\n📈 STEP 2: Exploratory Data Analysis (EDA)...")
        
        # Create directory for plots
        os.makedirs('eda_plots', exist_ok=True)
        print("   📊 Creating visualizations...")
        
        # 1. Survival Distribution
        plt.figure(figsize=(8, 6))
        survival_counts = self.df['Survived'].value_counts()
        plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], 
                autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'], startangle=90)
        plt.title('Survival Distribution')
        plt.savefig('eda_plots/survival_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Survival by Gender
        plt.figure(figsize=(8, 6))
        survival_by_sex = pd.crosstab(self.df['Sex'], self.df['Survived'])
        survival_by_sex.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title('Survival by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(['Not Survived', 'Survived'])
        plt.xticks(rotation=0)
        plt.savefig('eda_plots/survival_by_gender.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Survival by Class
        plt.figure(figsize=(8, 6))
        survival_by_class = pd.crosstab(self.df['Pclass'], self.df['Survived'])
        survival_by_class.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title('Survival by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Count')
        plt.legend(['Not Survived', 'Survived'])
        plt.xticks(rotation=0)
        plt.savefig('eda_plots/survival_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ EDA completed! Visualizations saved in 'eda_plots/' folder")
        
    def feature_engineering(self):
        """STEP 3: Feature Engineering"""
        print("\n🔧 STEP 3: Feature Engineering...")
        
        # Create copy for feature engineering
        feature_df = self.df.copy()
        
        print("   📝 APPLYING FEATURE TRANSFORMATIONS:")
        
        # 1. Create New Features
        feature_df['FamilySize'] = feature_df['SibSp'] + feature_df['Parch'] + 1
        print("     1. Created 'FamilySize' = SibSp + Parch + 1")
        
        feature_df['IsAlone'] = 0
        feature_df.loc[feature_df['FamilySize'] == 1, 'IsAlone'] = 1
        print("     2. Created 'IsAlone' (1 if traveling alone)")
        
        # 2. Extract Title from Name
        feature_df['Title'] = feature_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        feature_df['Title'] = feature_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        feature_df['Title'] = feature_df['Title'].replace('Mlle', 'Miss')
        feature_df['Title'] = feature_df['Title'].replace('Ms', 'Miss')
        feature_df['Title'] = feature_df['Title'].replace('Mme', 'Mrs')
        print("     3. Extracted and categorized 'Title' from Name")
        
        # 3. Binning - Age Groups
        feature_df['AgeGroup'] = pd.cut(feature_df['Age'], 
                                       bins=[0, 12, 18, 35, 60, 100], 
                                       labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        print("     4. Created 'AgeGroup' using binning (5 categories)")
        
        # 4. Binning - Fare Groups
        feature_df['FareGroup'] = pd.qcut(feature_df['Fare'], 4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        print("     5. Created 'FareGroup' using quantile binning")
        
        # 5. Encoding Categorical Variables
        label_encoders = {}
        binary_cols = ['Sex', 'Title']
        for col in binary_cols:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            label_encoders[col] = le
        print("     6. Applied Label Encoding to 'Sex' and 'Title'")
        
        # One-Hot Encoding
        categorical_cols = ['Embarked', 'AgeGroup', 'FareGroup', 'Pclass']
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, prefix=categorical_cols)
        print(f"     7. Applied One-Hot Encoding to {categorical_cols}")
        
        # 6. Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'PassengerId', 'Cabin']
        feature_df = feature_df.drop(columns_to_drop, axis=1)
        print(f"     8. Dropped columns: {columns_to_drop}")
        
        # 7. Prepare features and target
        self.feature_names = feature_df.drop('Survived', axis=1).columns.tolist()
        X = feature_df.drop('Survived', axis=1)
        y = feature_df['Survived']
        
        # 8. Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 9. Feature Scaling
        numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        if len(numerical_columns) > 0:
            self.X_train[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
            self.X_test[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
            print("     9. Applied StandardScaler to numerical features")
        
        print(f"\n   ✅ FEATURE ENGINEERING COMPLETED")
        print(f"   Final features: {len(self.feature_names)} features")
        print(f"   Training set: {self.X_train.shape[0]} records")
        print(f"   Testing set: {self.X_test.shape[0]} records")
        
    def train_and_evaluate_models(self):
        """STEP 4: Train and Evaluate Models"""
        print("\n🤖 STEP 4: Training and Evaluating Models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n   --- {name} ---")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            # Print results
            print(f"     ✅ Accuracy: {accuracy:.4f}")
            print(f"     ✅ Precision: {precision:.4f}")
            print(f"     ✅ Recall: {recall:.4f}")
            print(f"     ✅ F1-Score: {f1:.4f}")
            print(f"     ✅ ROC-AUC: {roc_auc:.4f}")
            
            # Store model
            self.models[name] = model
        
        # Select best model based on accuracy
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        self.best_score = results[self.best_model_name]['accuracy']
        
        print(f"\n   🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        print(f"   Precision: {results[self.best_model_name]['precision']:.4f}")
        print(f"   Recall: {results[self.best_model_name]['recall']:.4f}")
        print(f"   F1-Score: {results[self.best_model_name]['f1_score']:.4f}")
        print(f"   ROC-AUC: {results[self.best_model_name]['roc_auc']:.4f}")
        
        # Save comparison table
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\n   📊 Model comparison saved to 'model_comparison.csv'")
        
        return results
    
    def save_best_model(self):
        """STEP 5: Save the best model"""
        print("\n💾 STEP 5: Saving the best model...")
        
        # Create model artifacts
        model_artifacts = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'accuracy': self.best_score
        }
        
        joblib.dump(model_artifacts, 'best_model.pkl')
        print("   ✅ Best model saved as 'best_model.pkl'")
        
        # Save feature list
        with open('features.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        print("   ✅ Feature list saved as 'features.txt'")

def main():
    """Main training function"""
    pipeline = TitanicMLPipeline()
    
    try:
        # Execute pipeline steps
        pipeline.load_and_clean_data()
        pipeline.exploratory_data_analysis()
        pipeline.feature_engineering()
        results = pipeline.train_and_evaluate_models()
        pipeline.save_best_model()
        
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("   Model saved as 'best_model.pkl'")
        print("   Run 'python app.py' to deploy the model")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    # NOTHING ELSE HERE!
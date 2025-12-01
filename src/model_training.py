import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def load_processed_data(self) -> tuple:
        \"\"\"Load processed data for model training\"\"\"
        df = pd.read_csv(self.config['data']['processed_data_path'])
        
        # Separate features and target
        X = df.drop(columns=[self.config['features']['target']])
        y = df[self.config['features']['target']]
        
        print(f'📊 Loaded processed data: {X.shape}')
        print(f'🎯 Target distribution: {y.value_counts().to_dict()}')
        
        return X, y
    
    def prepare_data(self, X, y) -> tuple:
        \"\"\"Split data into train and test sets\"\"\"
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f'📚 Data split: Train={X_train.shape}, Test={X_test.shape}')
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test) -> dict:
        \"\"\"Train multiple models and evaluate performance\"\"\"
        print('🚀 Training Multiple Models...')
        print('=' * 50)
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f'\n📚 Training {model_name}...')
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                print(f'✅ {model_name} trained successfully')
                print(f'   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f'❌ Error training {model_name}: {str(e)}')
                continue
        
        self.models = results
        return results
    
    def save_models(self):
        \"\"\"Save trained models to disk\"\"\"
        from utils import save_model
        
        # Save all models
        for model_name, result in self.models.items():
            filename = f'models/trained_models/{model_name}.pkl'
            save_model(result['model'], filename)
        
        # Save best model separately
        save_model(self.best_model, 'models/best_model.pkl')
        
        # Save feature names
        X, _ = self.load_processed_data()
        feature_names = X.columns.tolist()
        save_model(feature_names, 'models/feature_names.pkl')
        
        print(f'💾 Saved {len(self.models)} trained models')
        print(f'🏆 Best model: {self.best_model_name} (Accuracy: {self.best_score:.4f})')
    
    def train_complete_pipeline(self):
        \"\"\"Execute complete model training pipeline\"\"\"
        print('🚀 Starting Model Training Pipeline...')
        print('=' * 50)
        
        # Load and prepare data
        X, y = self.load_processed_data()
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()
        
        print('=' * 50)
        print('🎉 Model Training Completed!')
        
        return results

# Usage example
if __name__ == '__main__':
    from utils import load_config
    config = load_config()
    trainer = ModelTrainer(config)
    results = trainer.train_complete_pipeline()

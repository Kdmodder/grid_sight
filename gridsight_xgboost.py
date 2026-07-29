"""
GridSight: Phase-Aware XGBoost Forecasting Engine
for Power Transmission Infrastructure Projects

Predicts cumulative delay and cost overrun at each project milestone
using sequential masking to simulate real-time risk assessment.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

# MLflow integration
import mlflow
import mlflow.sklearn

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class GridSightForecaster:
    """
    Phase-aware forecasting engine for infrastructure projects.
    """
    
    def __init__(self, data_path, dataset_version="v1"):
        """Initialize the forecaster with data."""
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}
        self.models = {'delay': None, 'cost': None}
        self.feature_names = None
        self.dataset_version = dataset_version  # Track dataset version for MLflow
        self.categorical_features = [
            'Project_Type', 'Voltage_Level', 'Contract_Type',
            'DPR_Status', 'Statutory_Approvals', 'Equip_Delivery_Status',
            'Terrain_Type', 'Trial_Run_Status'
        ]
        
        # XGBoost hyperparameters (stored for MLflow logging)
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'enable_categorical': False
        }
        
    def prepare_features(self):
        """
        Prepare features with proper encoding and phase-aware masking.
        The NaN values for future phases are INTENTIONALLY kept as they
        represent information not yet available at that milestone.
        """
        df = self.data.copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                # Handle NaN values in categorical columns
                mask = df[col].notna()
                if mask.sum() > 0:
                    df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
                    self.label_encoders[col] = le
                # Convert to float to handle NaN and ensure numeric dtype
                df[col] = df[col].astype(float)
        
        # Encode Project_Location (extract state)
        if 'Project_Location' in df.columns:
            df['State'] = df['Project_Location'].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown')
            le = LabelEncoder()
            df['State_Encoded'] = le.fit_transform(df['State'])
            self.label_encoders['State'] = le
            df.drop(['Project_Location', 'State'], axis=1, inplace=True)
        
        # Convert dates to days since start
        if 'Start_Date' in df.columns:
            df['Start_Date'] = pd.to_datetime(df['Start_Date'], format='%d-%m-%Y', errors='coerce')
            df['Days_Since_Start'] = (datetime.now() - df['Start_Date']).dt.days
            df.drop('Start_Date', axis=1, inplace=True)
        
        if 'DOCO_Date' in df.columns:
            df.drop('DOCO_Date', axis=1, inplace=True)
        
        # Define target variables
        self.targets = ['Cumulative_Delay_Days', 'Cumulative_Cost_Overrun_Cr']
        
        # Define metadata features (static project DNA)
        self.metadata_features = [
            'Project_Type', 'Voltage_Level', 'Project_Size', 
            'Contract_Type', 'Planned_Duration', 'Total_Budget_Cr',
            'State_Encoded', 'Days_Since_Start'
        ]
        
        # Phase features (dynamic evidence) - these contain NaN for future phases
        self.phase_features = [col for col in df.columns 
                               if col not in self.metadata_features + self.targets 
                               + ['Project_ID', 'Phase_Num', 'Phase_Name']]
        
        self.all_features = self.metadata_features + self.phase_features
        self.feature_names = self.all_features
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Group-based train/test split to prevent data leakage.
        All 7 phases of a project stay together in train or test.
        """
        # Get unique project IDs
        unique_projects = df['Project_ID'].unique()
        
        # Create group split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        
        # Get indices for train and test
        train_idx, test_idx = next(gss.split(df, groups=df['Project_ID']))
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        print(f"Training Projects: {train_df['Project_ID'].nunique()}")
        print(f"Testing Projects: {test_df['Project_ID'].nunique()}")
        print(f"Training Snapshots: {len(train_df)}")
        print(f"Testing Snapshots: {len(test_df)}")
        
        return train_df, test_df
    
    def train_models(self, train_df):
        """
        Train XGBoost models for delay and cost prediction.
        XGBoost natively handles NaN values as missing data.
        """
        X_train = train_df[self.all_features]
        y_delay = train_df['Cumulative_Delay_Days']
        y_cost = train_df['Cumulative_Cost_Overrun_Cr']
        
        print("\n" + "="*60)
        print("Training Delay Model...")
        print("="*60)
        self.models['delay'] = xgb.XGBRegressor(**self.xgb_params)
        self.models['delay'].fit(
            X_train, y_delay,
            eval_set=[(X_train, y_delay)],
            verbose=50
        )
        
        print("\n" + "="*60)
        print("Training Cost Overrun Model...")
        print("="*60)
        self.models['cost'] = xgb.XGBRegressor(**self.xgb_params)
        self.models['cost'].fit(
            X_train, y_cost,
            eval_set=[(X_train, y_cost)],
            verbose=50
        )
        
        print("\nModels trained successfully!")
        
    def log_to_mlflow(self, phase_metrics, overall_metrics, model_dir='models'):
        """
        Log training results to MLflow for experiment tracking.
        
        Parameters:
        -----------
        phase_metrics : pd.DataFrame
            Phase-wise performance metrics
        overall_metrics : dict
            Overall model performance metrics
        model_dir : str
            Directory where models are saved
        """
        print("\n" + "="*60)
        print("LOGGING TO MLFLOW")
        print("="*60)
        
        # Configure MLflow tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Set experiment name
        experiment_name = "GridSight_PhaseAware_Cumulative_Forecasting"
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            print(f"\nMLflow Run ID: {run.info.run_id}")
            
            # ========== LOG PARAMETERS ==========
            print("\nLogging parameters...")
            
            # Model parameters
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("learning_rate", self.xgb_params['learning_rate'])
            mlflow.log_param("max_depth", self.xgb_params['max_depth'])
            mlflow.log_param("n_estimators", self.xgb_params['n_estimators'])
            mlflow.log_param("subsample", self.xgb_params['subsample'])
            mlflow.log_param("colsample_bytree", self.xgb_params['colsample_bytree'])
            
            # GridSight-specific parameters
            mlflow.log_param("masking_strategy", "future_phases_nan")
            mlflow.log_param("memory_logic", "forward_fill")
            mlflow.log_param("prediction_type", "cumulative")
            mlflow.log_param("split_strategy", "group_by_project_id")
            mlflow.log_param("dataset_version", self.dataset_version)
            
            # ========== LOG METRICS ==========
            print("Logging metrics...")
            
            # Phase-wise metrics for delay
            for _, row in phase_metrics.iterrows():
                phase_num = int(row['Phase'])
                mlflow.log_metric(f"mae_delay_phase_{phase_num}", row['Delay_MAE'])
                mlflow.log_metric(f"mae_cost_phase_{phase_num}", row['Cost_MAE'])
            
            # Overall metrics
            mlflow.log_metric("overall_mae_delay", overall_metrics['delay_mae'])
            mlflow.log_metric("overall_mae_cost", overall_metrics['cost_mae'])
            mlflow.log_metric("overall_rmse_delay", overall_metrics['delay_rmse'])
            mlflow.log_metric("overall_rmse_cost", overall_metrics['cost_rmse'])
            mlflow.log_metric("overall_r2_delay", overall_metrics['delay_r2'])
            mlflow.log_metric("overall_r2_cost", overall_metrics['cost_r2'])
            
            # ========== LOG MODELS ==========
            print("Logging models...")
            
            # Log delay model (using sklearn API for compatibility)
            mlflow.sklearn.log_model(
                self.models['delay'],
                artifact_path="delay_model",
                registered_model_name="GridSight_Delay_Model"
            )
            
            # Log cost model (using sklearn API for compatibility)
            mlflow.sklearn.log_model(
                self.models['cost'],
                artifact_path="cost_model",
                registered_model_name="GridSight_Cost_Model"
            )
            
            # ========== LOG ARTIFACTS ==========
            print("Logging artifacts...")
            
            # Save and log feature list
            feature_list_path = os.path.join(model_dir, 'feature_list.json')
            with open(feature_list_path, 'w') as f:
                json.dump({
                    'all_features': self.all_features,
                    'metadata_features': self.metadata_features,
                    'phase_features': self.phase_features
                }, f, indent=2)
            mlflow.log_artifact(feature_list_path)
            
            # Save and log phase mapping
            phase_mapping_path = os.path.join(model_dir, 'phase_mapping.json')
            phase_mapping = {
                int(row['Phase']): row['Phase_Name'] 
                for _, row in phase_metrics.iterrows()
            }
            with open(phase_mapping_path, 'w') as f:
                json.dump(phase_mapping, f, indent=2)
            mlflow.log_artifact(phase_mapping_path)
            
            # Save and log phase-wise error table
            phasewise_error_path = os.path.join(model_dir, 'phasewise_error_table.csv')
            phase_metrics.to_csv(phasewise_error_path, index=False)
            mlflow.log_artifact(phasewise_error_path)
            
            # Log visualizations if they exist
            if os.path.exists('gridsight_predictions.png'):
                mlflow.log_artifact('gridsight_predictions.png')
            if os.path.exists('gridsight_shap_summary.png'):
                mlflow.log_artifact('gridsight_shap_summary.png')
            
            # ========== LOG TAGS ==========
            print("Logging tags...")
            mlflow.set_tag("project", "GridSight")
            mlflow.set_tag("domain", "power_transmission")
            mlflow.set_tag("training_type", "snapshot_based_sequential")
            mlflow.set_tag("model_purpose", "phase_aware_cumulative_forecasting")
            
            print(f"\n✓ Successfully logged to MLflow")
            print(f"  Experiment: {experiment_name}")
            print(f"  Run ID: {run.info.run_id}")
            print(f"  Tracking URI: http://127.0.0.1:5000")
            
            return run.info.run_id
    
    def save_models(self, model_dir='models'):
        """
        Save trained models and encoders for future use.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save models using pickle
        with open(os.path.join(model_dir, 'delay_model.pkl'), 'wb') as f:
            pickle.dump(self.models['delay'], f)
        
        with open(os.path.join(model_dir, 'cost_model.pkl'), 'wb') as f:
            pickle.dump(self.models['cost'], f)
        
        # Save label encoders and feature names
        with open(os.path.join(model_dir, 'encoders_and_features.pkl'), 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'all_features': self.all_features,
                'metadata_features': self.metadata_features,
                'phase_features': self.phase_features,
                'categorical_features': self.categorical_features
            }, f)
        
        print(f"\nModels saved to '{model_dir}/' directory")
        print("  ✓ delay_model.pkl")
        print("  ✓ cost_model.pkl")
        print("  ✓ encoders_and_features.pkl")
    
    def load_models(self, model_dir='models'):
        """
        Load trained models and encoders from disk.
        """
        # Load models
        with open(os.path.join(model_dir, 'delay_model.pkl'), 'rb') as f:
            self.models['delay'] = pickle.load(f)
        
        with open(os.path.join(model_dir, 'cost_model.pkl'), 'rb') as f:
            self.models['cost'] = pickle.load(f)
        
        # Load encoders and features
        with open(os.path.join(model_dir, 'encoders_and_features.pkl'), 'rb') as f:
            saved_data = pickle.load(f)
            self.label_encoders = saved_data['label_encoders']
            self.feature_names = saved_data['feature_names']
            self.all_features = saved_data['all_features']
            self.metadata_features = saved_data['metadata_features']
            self.phase_features = saved_data['phase_features']
            self.categorical_features = saved_data['categorical_features']
        
        print(f"\nModels loaded from '{model_dir}/' directory")
        print("  ✓ Delay model")
        print("  ✓ Cost model")
        print("  ✓ Encoders and features")
    
    def evaluate_models(self, test_df):
        """
        Evaluate models with phase-wise performance analysis.
        """
        X_test = test_df[self.all_features]
        y_delay_true = test_df['Cumulative_Delay_Days']
        y_cost_true = test_df['Cumulative_Cost_Overrun_Cr']
        
        # Predictions
        y_delay_pred = self.models['delay'].predict(X_test)
        y_cost_pred = self.models['cost'].predict(X_test)
        
        # Overall metrics
        print("\n" + "="*60)
        print("OVERALL MODEL PERFORMANCE")
        print("="*60)
        
        delay_mae = mean_absolute_error(y_delay_true, y_delay_pred)
        delay_rmse = np.sqrt(mean_squared_error(y_delay_true, y_delay_pred))
        delay_r2 = r2_score(y_delay_true, y_delay_pred)
        
        cost_mae = mean_absolute_error(y_cost_true, y_cost_pred)
        cost_rmse = np.sqrt(mean_squared_error(y_cost_true, y_cost_pred))
        cost_r2 = r2_score(y_cost_true, y_cost_pred)
        
        print("\nDelay Prediction Metrics:")
        print(f"  MAE:  {delay_mae:.2f} days")
        print(f"  RMSE: {delay_rmse:.2f} days")
        print(f"  R²:   {delay_r2:.4f}")
        
        print("\nCost Overrun Prediction Metrics:")
        print(f"  MAE:  {cost_mae:.2f} Cr")
        print(f"  RMSE: {cost_rmse:.2f} Cr")
        print(f"  R²:   {cost_r2:.4f}")
        
        # Store overall metrics for MLflow
        overall_metrics = {
            'delay_mae': delay_mae,
            'delay_rmse': delay_rmse,
            'delay_r2': delay_r2,
            'cost_mae': cost_mae,
            'cost_rmse': cost_rmse,
            'cost_r2': cost_r2
        }
        
        # Phase-wise analysis
        print("\n" + "="*60)
        print("PHASE-WISE PERFORMANCE ANALYSIS")
        print("="*60)
        
        test_df_eval = test_df.copy()
        test_df_eval['Delay_Pred'] = y_delay_pred
        test_df_eval['Cost_Pred'] = y_cost_pred
        
        phase_metrics = []
        
        for phase in sorted(test_df_eval['Phase_Num'].unique()):
            phase_data = test_df_eval[test_df_eval['Phase_Num'] == phase]
            
            if len(phase_data) > 0:
                delay_mae = mean_absolute_error(
                    phase_data['Cumulative_Delay_Days'], 
                    phase_data['Delay_Pred']
                )
                cost_mae = mean_absolute_error(
                    phase_data['Cumulative_Cost_Overrun_Cr'], 
                    phase_data['Cost_Pred']
                )
                
                phase_name = phase_data['Phase_Name'].iloc[0]
                
                phase_metrics.append({
                    'Phase': phase,
                    'Phase_Name': phase_name,
                    'Samples': len(phase_data),
                    'Delay_MAE': delay_mae,
                    'Cost_MAE': cost_mae
                })
                
                print(f"\nPhase {phase}: {phase_name}")
                print(f"  Samples: {len(phase_data)}")
                print(f"  Delay MAE: {delay_mae:.2f} days")
                print(f"  Cost MAE: {cost_mae:.2f} Cr")
        
        return pd.DataFrame(phase_metrics), test_df_eval, overall_metrics
    
    def visualize_predictions(self, test_df_eval):
        """
        Create comprehensive visualization of model predictions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Delay: Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(test_df_eval['Cumulative_Delay_Days'], 
                  test_df_eval['Delay_Pred'], 
                  alpha=0.6, c=test_df_eval['Phase_Num'], cmap='viridis')
        max_val = max(test_df_eval['Cumulative_Delay_Days'].max(), 
                     test_df_eval['Delay_Pred'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Delay (Days)', fontsize=12)
        ax.set_ylabel('Predicted Delay (Days)', fontsize=12)
        ax.set_title('Delay Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Cost: Actual vs Predicted
        ax = axes[0, 1]
        scatter = ax.scatter(test_df_eval['Cumulative_Cost_Overrun_Cr'], 
                            test_df_eval['Cost_Pred'], 
                            alpha=0.6, c=test_df_eval['Phase_Num'], cmap='viridis')
        max_val = max(test_df_eval['Cumulative_Cost_Overrun_Cr'].max(), 
                     test_df_eval['Cost_Pred'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Cost Overrun (Cr)', fontsize=12)
        ax.set_ylabel('Predicted Cost Overrun (Cr)', fontsize=12)
        ax.set_title('Cost Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Phase Number')
        
        # 3. Delay Error by Phase
        ax = axes[1, 0]
        test_df_eval['Delay_Error'] = test_df_eval['Delay_Pred'] - test_df_eval['Cumulative_Delay_Days']
        phase_delay_errors = test_df_eval.groupby('Phase_Num')['Delay_Error'].apply(list)
        ax.boxplot([phase_delay_errors[i] for i in sorted(phase_delay_errors.index)],
                   labels=sorted(phase_delay_errors.index))
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Phase Number', fontsize=12)
        ax.set_ylabel('Prediction Error (Days)', fontsize=12)
        ax.set_title('Delay Prediction Error Distribution by Phase', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Cost Error by Phase
        ax = axes[1, 1]
        test_df_eval['Cost_Error'] = test_df_eval['Cost_Pred'] - test_df_eval['Cumulative_Cost_Overrun_Cr']
        phase_cost_errors = test_df_eval.groupby('Phase_Num')['Cost_Error'].apply(list)
        ax.boxplot([phase_cost_errors[i] for i in sorted(phase_cost_errors.index)],
                   labels=sorted(phase_cost_errors.index))
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Phase Number', fontsize=12)
        ax.set_ylabel('Prediction Error (Cr)', fontsize=12)
        ax.set_title('Cost Prediction Error Distribution by Phase', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gridsight_predictions.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'gridsight_predictions.png'")
        plt.close()
    
    def explain_with_shap(self, test_df, sample_size=100):
        """
        Generate SHAP explanations to identify key risk drivers.
        """
        print("\n" + "="*60)
        print("GENERATING SHAP EXPLANATIONS")
        print("="*60)
        
        # Sample for SHAP (computationally expensive for large datasets)
        X_sample = test_df[self.all_features].sample(
            n=min(sample_size, len(test_df)), 
            random_state=42
        )
        
        # SHAP for Delay Model
        print("\nCalculating SHAP values for Delay Model...")
        explainer_delay = shap.TreeExplainer(self.models['delay'])
        shap_values_delay = explainer_delay.shap_values(X_sample)
        
        # SHAP for Cost Model
        print("Calculating SHAP values for Cost Model...")
        explainer_cost = shap.TreeExplainer(self.models['cost'])
        shap_values_cost = explainer_cost.shap_values(X_sample)
        
        # Visualizations
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Delay SHAP Summary
        plt.sca(axes[0])
        shap.summary_plot(shap_values_delay, X_sample, 
                         feature_names=self.feature_names,
                         show=False, max_display=20)
        axes[0].set_title('SHAP Feature Importance: Delay Prediction', 
                         fontsize=14, fontweight='bold', pad=20)
        
        # Cost SHAP Summary
        plt.sca(axes[1])
        shap.summary_plot(shap_values_cost, X_sample, 
                         feature_names=self.feature_names,
                         show=False, max_display=20)
        axes[1].set_title('SHAP Feature Importance: Cost Overrun Prediction', 
                         fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('gridsight_shap_summary.png', dpi=300, bbox_inches='tight')
        print("SHAP summary saved as 'gridsight_shap_summary.png'")
        plt.close()
        
        # Feature importance analysis
        self.analyze_shap_insights(shap_values_delay, shap_values_cost, X_sample)
        
        return shap_values_delay, shap_values_cost, X_sample
    
    def analyze_shap_insights(self, shap_delay, shap_cost, X_sample):
        """
        Provide detailed interpretation of SHAP values.
        """
        print("\n" + "="*60)
        print("KEY INSIGHTS FROM SHAP ANALYSIS")
        print("="*60)
        
        # Calculate mean absolute SHAP values
        delay_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_SHAP': np.abs(shap_delay).mean(axis=0)
        }).sort_values('Mean_SHAP', ascending=False)
        
        cost_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_SHAP': np.abs(shap_cost).mean(axis=0)
        }).sort_values('Mean_SHAP', ascending=False)
        
        print("\nTop 10 Drivers of DELAY:")
        print("-" * 50)
        for idx, row in delay_importance.head(10).iterrows():
            print(f"  {row['Feature']:40s} | Impact: {row['Mean_SHAP']:.4f}")
        
        print("\nTop 10 Drivers of COST OVERRUN:")
        print("-" * 50)
        for idx, row in cost_importance.head(10).iterrows():
            print(f"  {row['Feature']:40s} | Impact: {row['Mean_SHAP']:.4f}")
        
        # Phase-specific analysis
        print("\n" + "="*60)
        print("PHASE-SPECIFIC RISK FACTORS")
        print("="*60)
        
        phase_features = [f for f in self.feature_names if any(
            phase in f for phase in ['ROW', 'Supply', 'Civil', 'Erection', 'Stringing', 'Testing', 'Engg']
        )]
        
        for target_name, shap_vals, importance_df in [
            ('DELAY', shap_delay, delay_importance),
            ('COST OVERRUN', shap_cost, cost_importance)
        ]:
            print(f"\nPhase Factors Most Affecting {target_name}:")
            print("-" * 50)
            
            phase_importance = importance_df[importance_df['Feature'].isin(phase_features)].head(10)
            for idx, row in phase_importance.iterrows():
                print(f"  {row['Feature']:40s} | Impact: {row['Mean_SHAP']:.4f}")
    
    def predict_single_entry(self, entry_data, show_shap=True):
        """
        Make prediction for a single project entry and provide detailed analysis.
        
        Parameters:
        -----------
        entry_data : dict or pd.DataFrame
            Single entry with all required features
        show_shap : bool
            Whether to show SHAP explanation for this prediction
        
        Returns:
        --------
        dict with predictions and metrics
        """
        print("\n" + "="*60)
        print("SINGLE ENTRY PREDICTION")
        print("="*60)
        
        # Convert to DataFrame if dict
        if isinstance(entry_data, dict):
            entry_df = pd.DataFrame([entry_data])
        else:
            entry_df = entry_data.copy()
        
        # Check if we need to prepare features
        if not all(col in entry_df.columns for col in self.all_features):
            print("\nPreparing features for the entry...")
            # Apply same transformations as in prepare_features
            for col in self.categorical_features:
                if col in entry_df.columns and col in self.label_encoders:
                    mask = entry_df[col].notna()
                    if mask.sum() > 0:
                        entry_df.loc[mask, col] = self.label_encoders[col].transform(
                            entry_df.loc[mask, col].astype(str)
                        )
                    entry_df[col] = entry_df[col].astype(float)
        
        # Extract features
        X_entry = entry_df[self.all_features]
        
        # Make predictions
        delay_pred = self.models['delay'].predict(X_entry)[0]
        cost_pred = self.models['cost'].predict(X_entry)[0]
        
        print(f"\nPredicted Delay: {delay_pred:.2f} days")
        print(f"Predicted Cost Overrun: {cost_pred:.2f} Cr")
        
        # If actual values are provided, calculate metrics
        if 'Cumulative_Delay_Days' in entry_df.columns and 'Cumulative_Cost_Overrun_Cr' in entry_df.columns:
            actual_delay = entry_df['Cumulative_Delay_Days'].iloc[0]
            actual_cost = entry_df['Cumulative_Cost_Overrun_Cr'].iloc[0]
            
            delay_error = delay_pred - actual_delay
            cost_error = cost_pred - actual_cost
            
            delay_mae = abs(delay_error)
            cost_mae = abs(cost_error)
            
            delay_mse = delay_error ** 2
            cost_mse = cost_error ** 2
            
            print("\n" + "-"*60)
            print("ACTUAL vs PREDICTED COMPARISON")
            print("-"*60)
            print(f"\nDelay:")
            print(f"  Actual: {actual_delay:.2f} days")
            print(f"  Predicted: {delay_pred:.2f} days")
            print(f"  Error: {delay_error:+.2f} days")
            print(f"  MAE: {delay_mae:.2f} days")
            print(f"  MSE: {delay_mse:.2f}")
            
            print(f"\nCost Overrun:")
            print(f"  Actual: {actual_cost:.2f} Cr")
            print(f"  Predicted: {cost_pred:.2f} Cr")
            print(f"  Error: {cost_error:+.2f} Cr")
            print(f"  MAE: {cost_mae:.2f} Cr")
            print(f"  MSE: {cost_mse:.2f}")
        
        # SHAP explanation for this prediction
        if show_shap:
            print("\n" + "="*60)
            print("SHAP EXPLANATION FOR THIS PREDICTION")
            print("="*60)
            
            # Calculate SHAP values
            explainer_delay = shap.TreeExplainer(self.models['delay'])
            explainer_cost = shap.TreeExplainer(self.models['cost'])
            
            shap_values_delay = explainer_delay.shap_values(X_entry)
            shap_values_cost = explainer_cost.shap_values(X_entry)
            
            # Get top contributing features
            delay_contributions = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Value': shap_values_delay[0],
                'Abs_SHAP': np.abs(shap_values_delay[0])
            }).sort_values('Abs_SHAP', ascending=False)
            
            cost_contributions = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Value': shap_values_cost[0],
                'Abs_SHAP': np.abs(shap_values_cost[0])
            }).sort_values('Abs_SHAP', ascending=False)
            
            print("\nTop 10 Features Contributing to DELAY Prediction:")
            print("-"*60)
            for idx, row in delay_contributions.head(10).iterrows():
                direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
                print(f"  {row['Feature']:35s} | {direction:9s} by {abs(row['SHAP_Value']):.2f} days")
            
            print("\nTop 10 Features Contributing to COST Prediction:")
            print("-"*60)
            for idx, row in cost_contributions.head(10).iterrows():
                direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
                print(f"  {row['Feature']:35s} | {direction:9s} by {abs(row['SHAP_Value']):.2f} Cr")
            
            # Visualize SHAP for this prediction
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Delay waterfall
            plt.sca(axes[0])
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_delay[0],
                    base_values=explainer_delay.expected_value,
                    data=X_entry.iloc[0].values,
                    feature_names=self.feature_names
                ),
                show=False,
                max_display=15
            )
            axes[0].set_title('Delay Prediction Explanation', fontsize=14, fontweight='bold')
            
            # Cost waterfall
            plt.sca(axes[1])
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_cost[0],
                    base_values=explainer_cost.expected_value,
                    data=X_entry.iloc[0].values,
                    feature_names=self.feature_names
                ),
                show=False,
                max_display=15
            )
            axes[1].set_title('Cost Overrun Prediction Explanation', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('single_entry_shap.png', dpi=300, bbox_inches='tight')
            print("\nSHAP visualization saved as 'single_entry_shap.png'")
            plt.close()
        
        return {
            'delay_prediction': delay_pred,
            'cost_prediction': cost_pred
        }


def main():
    """
    Main execution pipeline for GridSight.
    """
    print("="*60)
    print("GRIDSIGHT: PHASE-AWARE INFRASTRUCTURE FORECASTING ENGINE")
    print("="*60)
    
    # Initialize
    forecaster = GridSightForecaster('Data/gridsight_dataset.csv', dataset_version="v1")
    
    # Prepare features
    print("\n[1/7] Preparing features with phase-based masking...")
    df = forecaster.prepare_features()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(forecaster.all_features)}")
    
    # Split data
    print("\n[2/7] Splitting data (group-based to prevent leakage)...")
    train_df, test_df = forecaster.split_data(df, test_size=0.2, random_state=42)
    
    # Train models
    print("\n[3/7] Training XGBoost models...")
    forecaster.train_models(train_df)
    
    # Save models
    print("\n[4/7] Saving trained models...")
    forecaster.save_models(model_dir='models')
    
    # Evaluate
    print("\n[5/7] Evaluating model performance...")
    phase_metrics, test_df_eval, overall_metrics = forecaster.evaluate_models(test_df)
    
    # Create champion model bundle (for production use)
    print("\n[5.5/7] Creating champion model bundle...")
    champion_bundle = {
        'delay_model': forecaster.models['delay'],
        'cost_model': forecaster.models['cost'],
        'encoders_and_features': {
            'label_encoders': forecaster.label_encoders,
            'feature_names': forecaster.feature_names,
            'all_features': forecaster.all_features,
            'metadata_features': forecaster.metadata_features,
            'phase_features': forecaster.phase_features,
            'categorical_features': forecaster.categorical_features
        },
        'dataset_version': forecaster.dataset_version,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/champion_model.pkl', 'wb') as f:
        pickle.dump(champion_bundle, f)
    
    # Save champion metrics
    champion_metrics = {
        'overall_mae_delay': overall_metrics['delay_mae'],
        'overall_mae_cost': overall_metrics['cost_mae'],
        'overall_rmse_delay': overall_metrics['delay_rmse'],
        'overall_rmse_cost': overall_metrics['cost_rmse'],
        'overall_r2_delay': overall_metrics['delay_r2'],
        'overall_r2_cost': overall_metrics['cost_r2'],
        'dataset_version': forecaster.dataset_version,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open('models/champion_metrics.json', 'w') as f:
        json.dump(champion_metrics, f, indent=2)
    
    print("✓ Champion model bundle created: models/champion_model.pkl")
    
    # Log to MLflow
    print("\n[6/7] Logging to MLflow...")
    run_id = forecaster.log_to_mlflow(phase_metrics, overall_metrics, model_dir='models')
    
    # Visualize
    print("\n[7/7] Creating prediction visualizations...")
    forecaster.visualize_predictions(test_df_eval)
    
    # SHAP analysis
    print("\n[BONUS] Generating SHAP explainability analysis...")
    shap_delay, shap_cost, X_sample = forecaster.explain_with_shap(test_df, sample_size=100)
    
    # Demo: Single entry prediction
    print("\n" + "="*60)
    print("DEMO: SINGLE ENTRY PREDICTION")
    print("="*60)
    print("\nTesting prediction on a random test entry...")
    
    # Select a random entry from test set
    sample_entry = test_df.sample(n=1, random_state=42)
    forecaster.predict_single_entry(sample_entry, show_shap=True)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print("\nDeliverables:")
    print("  ✓ Trained XGBoost models for delay and cost prediction")
    print("  ✓ Phase-wise performance metrics")
    print("  ✓ Prediction visualizations (gridsight_predictions.png)")
    print("  ✓ SHAP explainability analysis (gridsight_shap_summary.png)")
    print("  ✓ Single entry prediction demo (single_entry_shap.png)")
    print("  ✓ Saved models in 'models/' directory")
    print(f"  ✓ MLflow tracking (Run ID: {run_id})")
    print("\nGridSight is ready for deployment!")
    print(f"\n🔗 View MLflow dashboard: http://127.0.0.1:5000")
    

if __name__ == "__main__":
    main()
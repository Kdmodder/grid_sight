"""
GridSight Retraining Pipeline
Handles model retraining based on user-validated feedback data.

This pipeline:
1. Checks if feedback data has reached threshold (100 rows)
2. Merges original training data with new feedback
3. Retrains XGBoost models using phase-aware logic
4. Compares new model performance against champion model
5. Promotes new model only if it performs better
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.xgboost

# Import GridSight forecaster
from gridsight_xgboost import GridSightForecaster


class GridSightRetrainingPipeline:
    """
    Manages the retraining lifecycle for GridSight models.
    """
    
    def __init__(
        self,
        original_data_path='Data/gridsight_dataset.csv',
        feedback_data_path='Data/new_feedback_data.csv',
        dataset_versions_path='Data/dataset_versions.json',
        champion_model_path='models/champion_model.pkl',
        models_dir='models'
    ):
        """Initialize retraining pipeline."""
        self.original_data_path = original_data_path
        self.feedback_data_path = feedback_data_path
        self.dataset_versions_path = dataset_versions_path
        self.champion_model_path = champion_model_path
        self.models_dir = models_dir
        
        # Load dataset version info
        self.dataset_versions = self._load_dataset_versions()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("GridSight_PhaseAware_Cumulative_Forecasting")
    
    def _load_dataset_versions(self):
        """Load dataset version information."""
        if os.path.exists(self.dataset_versions_path):
            with open(self.dataset_versions_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize version tracking
            return {
                'current_version': 'v1',
                'version_history': [
                    {
                        'version': 'v1',
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'num_rows': None,  # Will be set on first run
                        'description': 'Original training dataset'
                    }
                ]
            }
    
    def _save_dataset_versions(self):
        """Save updated dataset version information."""
        os.makedirs(os.path.dirname(self.dataset_versions_path), exist_ok=True)
        with open(self.dataset_versions_path, 'w') as f:
            json.dump(self.dataset_versions, f, indent=2)
        print(f"✓ Updated dataset versions saved to {self.dataset_versions_path}")
    
    def check_retraining_trigger(self, threshold=100):
        """
        Check if retraining should be triggered.
        
        Parameters:
        -----------
        threshold : int
            Minimum number of feedback rows required to trigger retraining
        
        Returns:
        --------
        bool : Whether retraining should proceed
        int : Number of feedback rows available
        """
        if not os.path.exists(self.feedback_data_path):
            print(f"⚠ No feedback data found at {self.feedback_data_path}")
            print(f"  Create this file with user-validated predictions to enable retraining.")
            return False, 0
        
        feedback_df = pd.read_csv(self.feedback_data_path)
        num_feedback_rows = len(feedback_df)
        
        print("\n" + "="*60)
        print("RETRAINING TRIGGER CHECK")
        print("="*60)
        print(f"\nFeedback rows available: {num_feedback_rows}")
        print(f"Threshold required: {threshold}")
        
        if num_feedback_rows >= threshold:
            print(f"✓ Retraining triggered! ({num_feedback_rows} >= {threshold})")
            return True, num_feedback_rows
        else:
            print(f"✗ Insufficient feedback data ({num_feedback_rows} < {threshold})")
            print(f"  Need {threshold - num_feedback_rows} more rows to trigger retraining.")
            return False, num_feedback_rows
    
    def merge_datasets(self):
        """
        Merge original training data with new feedback data.
        
        Returns:
        --------
        pd.DataFrame : Merged dataset
        str : New dataset version
        """
        print("\n" + "="*60)
        print("MERGING DATASETS")
        print("="*60)
        
        # Load original data
        original_df = pd.read_csv(self.original_data_path)
        print(f"\nOriginal data: {len(original_df)} rows")
        
        # Load feedback data
        feedback_df = pd.read_csv(self.feedback_data_path)
        print(f"Feedback data: {len(feedback_df)} rows")
        
        # Merge datasets
        merged_df = pd.concat([original_df, feedback_df], ignore_index=True)
        
        # Remove duplicates (based on Project_ID and Phase_Num)
        if 'Project_ID' in merged_df.columns and 'Phase_Num' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['Project_ID', 'Phase_Num'], keep='last')
        
        print(f"Merged data: {len(merged_df)} rows (after deduplication)")
        
        # Increment dataset version
        current_version = self.dataset_versions['current_version']
        version_num = int(current_version.replace('v', '')) + 1
        new_version = f'v{version_num}'
        
        # Update version history
        self.dataset_versions['current_version'] = new_version
        self.dataset_versions['version_history'].append({
            'version': new_version,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_rows': len(merged_df),
            'description': f'Merged with {len(feedback_df)} feedback rows',
            'feedback_rows_added': len(feedback_df)
        })
        
        self._save_dataset_versions()
        
        print(f"\n✓ Dataset version incremented: {current_version} → {new_version}")
        
        return merged_df, new_version
    
    def retrain_models(self, merged_df, new_version):
        """
        Retrain XGBoost models using the merged dataset.
        
        Parameters:
        -----------
        merged_df : pd.DataFrame
            Merged training data
        new_version : str
            New dataset version identifier
        
        Returns:
        --------
        GridSightForecaster : Trained forecaster instance
        dict : Phase metrics
        dict : Overall metrics
        str : MLflow run ID
        """
        print("\n" + "="*60)
        print("RETRAINING MODELS")
        print("="*60)
        
        # Save merged dataset temporarily
        temp_data_path = 'Data/temp_retrain_data.csv'
        merged_df.to_csv(temp_data_path, index=False)
        
        # Initialize forecaster with new version
        forecaster = GridSightForecaster(temp_data_path, dataset_version=new_version)
        
        # Prepare features
        print("\n[1/4] Preparing features...")
        df = forecaster.prepare_features()
        
        # Split data (same strategy as original training)
        print("\n[2/4] Splitting data...")
        train_df, test_df = forecaster.split_data(df, test_size=0.2, random_state=42)
        
        # Train models
        print("\n[3/4] Training XGBoost models...")
        forecaster.train_models(train_df)
        
        # Evaluate models
        print("\n[4/4] Evaluating models...")
        phase_metrics, test_df_eval, overall_metrics = forecaster.evaluate_models(test_df)
        
        # Create challenger model directory
        challenger_dir = os.path.join(self.models_dir, 'challenger')
        os.makedirs(challenger_dir, exist_ok=True)
        
        # Save challenger models
        forecaster.save_models(model_dir=challenger_dir)
        
        # Log to MLflow with retraining tag
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log all standard parameters and metrics
            forecaster.log_to_mlflow(phase_metrics, overall_metrics, model_dir=challenger_dir)
            
            # Add retraining-specific tags
            mlflow.set_tag("training_type", "retraining")
            mlflow.set_tag("retrain_timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            mlflow.set_tag("previous_version", self.dataset_versions['version_history'][-2]['version'])
            
            print(f"\n✓ Retraining logged to MLflow (Run ID: {run_id})")
        
        # Clean up temporary file
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        
        return forecaster, phase_metrics, overall_metrics, run_id
    
    def get_champion_metrics(self):
        """
        Get performance metrics of the current champion model.
        
        Returns:
        --------
        dict : Champion model metrics (or None if no champion exists)
        """
        # Try to load champion metrics from MLflow or saved file
        champion_metrics_path = os.path.join(self.models_dir, 'champion_metrics.json')
        
        if os.path.exists(champion_metrics_path):
            with open(champion_metrics_path, 'r') as f:
                champion_metrics = json.load(f)
            print("\n" + "="*60)
            print("CHAMPION MODEL METRICS")
            print("="*60)
            print(f"Version: {champion_metrics.get('dataset_version', 'unknown')}")
            print(f"Delay MAE: {champion_metrics['overall_mae_delay']:.2f} days")
            print(f"Cost MAE: {champion_metrics['overall_mae_cost']:.2f} Cr")
            return champion_metrics
        else:
            print("\n⚠ No champion model metrics found. This will be the first champion.")
            return None
    
    def compare_models(self, champion_metrics, challenger_metrics):
        """
        Compare champion and challenger model performance.
        
        Parameters:
        -----------
        champion_metrics : dict
            Current champion model metrics
        challenger_metrics : dict
            New challenger model metrics
        
        Returns:
        --------
        bool : True if challenger is better, False otherwise
        dict : Comparison results
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        if champion_metrics is None:
            print("\n✓ No existing champion. Challenger will become the new champion.")
            return True, {'reason': 'first_model'}
        
        # Compare MAE metrics (lower is better)
        delay_improvement = champion_metrics['overall_mae_delay'] - challenger_metrics['delay_mae']
        cost_improvement = champion_metrics['overall_mae_cost'] - challenger_metrics['cost_mae']
        
        print("\n" + "-"*60)
        print("DELAY PREDICTION (MAE)")
        print("-"*60)
        print(f"Champion: {champion_metrics['overall_mae_delay']:.2f} days")
        print(f"Challenger: {challenger_metrics['delay_mae']:.2f} days")
        print(f"Improvement: {delay_improvement:+.2f} days ({(delay_improvement/champion_metrics['overall_mae_delay']*100):+.2f}%)")
        
        print("\n" + "-"*60)
        print("COST OVERRUN PREDICTION (MAE)")
        print("-"*60)
        print(f"Champion: {champion_metrics['overall_mae_cost']:.2f} Cr")
        print(f"Challenger: {challenger_metrics['cost_mae']:.2f} Cr")
        print(f"Improvement: {cost_improvement:+.2f} Cr ({(cost_improvement/champion_metrics['overall_mae_cost']*100):+.2f}%)")
        
        # Decision: Challenger must be better in BOTH metrics
        challenger_better = (delay_improvement > 0) and (cost_improvement > 0)
        
        comparison_results = {
            'delay_improvement': delay_improvement,
            'cost_improvement': cost_improvement,
            'delay_improvement_pct': (delay_improvement / champion_metrics['overall_mae_delay'] * 100),
            'cost_improvement_pct': (cost_improvement / champion_metrics['overall_mae_cost'] * 100),
            'challenger_better': challenger_better
        }
        
        print("\n" + "="*60)
        if challenger_better:
            print("✓ DECISION: CHALLENGER IS BETTER")
            print("  New model will be promoted to champion")
        else:
            print("✗ DECISION: CHAMPION RETAINED")
            print("  New model does not improve both metrics")
        print("="*60)
        
        return challenger_better, comparison_results
    
    def promote_challenger(self, challenger_metrics, run_id):
        """
        Promote challenger model to champion.
        
        Parameters:
        -----------
        challenger_metrics : dict
            Challenger model metrics
        run_id : str
            MLflow run ID for the challenger
        """
        print("\n" + "="*60)
        print("PROMOTING CHALLENGER TO CHAMPION")
        print("="*60)
        
        challenger_dir = os.path.join(self.models_dir, 'challenger')
        
        # Archive old champion (if exists)
        if os.path.exists(self.champion_model_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = os.path.join(
                self.models_dir,
                f'champion_model_archived_{timestamp}.pkl'
            )
            os.rename(self.champion_model_path, archive_path)
            print(f"✓ Old champion archived: {archive_path}")
        
        # Copy challenger models to champion location
        # Save both delay and cost models as a single champion bundle
        with open(os.path.join(challenger_dir, 'delay_model.pkl'), 'rb') as f:
            delay_model = pickle.load(f)
        with open(os.path.join(challenger_dir, 'cost_model.pkl'), 'rb') as f:
            cost_model = pickle.load(f)
        with open(os.path.join(challenger_dir, 'encoders_and_features.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        champion_bundle = {
            'delay_model': delay_model,
            'cost_model': cost_model,
            'encoders_and_features': encoders,
            'dataset_version': self.dataset_versions['current_version'],
            'promoted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mlflow_run_id': run_id
        }
        
        with open(self.champion_model_path, 'wb') as f:
            pickle.dump(champion_bundle, f)
        
        print(f"✓ New champion promoted: {self.champion_model_path}")
        
        # Save champion metrics
        champion_metrics_path = os.path.join(self.models_dir, 'champion_metrics.json')
        metrics_to_save = {
            'overall_mae_delay': challenger_metrics['delay_mae'],
            'overall_mae_cost': challenger_metrics['cost_mae'],
            'overall_rmse_delay': challenger_metrics['delay_rmse'],
            'overall_rmse_cost': challenger_metrics['cost_rmse'],
            'overall_r2_delay': challenger_metrics['delay_r2'],
            'overall_r2_cost': challenger_metrics['cost_r2'],
            'dataset_version': self.dataset_versions['current_version'],
            'promoted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mlflow_run_id': run_id
        }
        
        with open(champion_metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"✓ Champion metrics saved: {champion_metrics_path}")
    
    def save_challenger(self, run_id):
        """
        Save rejected challenger model with timestamp.
        
        Parameters:
        -----------
        run_id : str
            MLflow run ID for the challenger
        """
        print("\n" + "="*60)
        print("ARCHIVING CHALLENGER MODEL")
        print("="*60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        challenger_dir = os.path.join(self.models_dir, 'challenger')
        
        # Create archive directory if it doesn't exist
        archive_dir = os.path.join(self.models_dir, 'archived_challengers')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Archive challenger
        archived_path = os.path.join(
            archive_dir,
            f'challenger_model_{timestamp}.pkl'
        )
        
        # Bundle challenger models
        with open(os.path.join(challenger_dir, 'delay_model.pkl'), 'rb') as f:
            delay_model = pickle.load(f)
        with open(os.path.join(challenger_dir, 'cost_model.pkl'), 'rb') as f:
            cost_model = pickle.load(f)
        with open(os.path.join(challenger_dir, 'encoders_and_features.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        challenger_bundle = {
            'delay_model': delay_model,
            'cost_model': cost_model,
            'encoders_and_features': encoders,
            'dataset_version': self.dataset_versions['current_version'],
            'archived_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mlflow_run_id': run_id,
            'status': 'rejected'
        }
        
        with open(archived_path, 'wb') as f:
            pickle.dump(challenger_bundle, f)
        
        print(f"✓ Challenger archived: {archived_path}")
    
    def generate_retraining_report(self, comparison_results, challenger_metrics, run_id, promoted):
        """
        Generate a markdown report documenting the retraining process.
        
        Parameters:
        -----------
        comparison_results : dict
            Results from model comparison
        challenger_metrics : dict
            Challenger model metrics
        run_id : str
            MLflow run ID
        promoted : bool
            Whether challenger was promoted
        """
        print("\n" + "="*60)
        print("GENERATING RETRAINING REPORT")
        print("="*60)
        
        report_path = 'retraining_report.md'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build report content
        report = f"""# GridSight Retraining Report

**Generated:** {timestamp}  
**MLflow Run ID:** `{run_id}`

---

## Dataset Versions

### Current Version: {self.dataset_versions['current_version']}

"""
        
        # Add version history
        report += "### Version History\n\n"
        for version_info in self.dataset_versions['version_history'][-3:]:  # Last 3 versions
            report += f"- **{version_info['version']}** ({version_info['created_at']}): "
            report += f"{version_info['description']}\n"
            if version_info.get('num_rows'):
                report += f"  - Rows: {version_info['num_rows']}\n"
        
        report += "\n---\n\n"
        
        # Add metrics comparison
        report += "## Performance Metrics\n\n"
        
        if comparison_results.get('reason') == 'first_model':
            report += "**Status:** First model trained (no previous champion)\n\n"
        else:
            report += "### Delay Prediction (MAE)\n\n"
            report += f"- **Champion:** {comparison_results.get('champion_delay_mae', 'N/A')} days\n"
            report += f"- **Challenger:** {challenger_metrics['delay_mae']:.2f} days\n"
            report += f"- **Improvement:** {comparison_results['delay_improvement']:+.2f} days "
            report += f"({comparison_results['delay_improvement_pct']:+.2f}%)\n\n"
            
            report += "### Cost Overrun Prediction (MAE)\n\n"
            report += f"- **Champion:** {comparison_results.get('champion_cost_mae', 'N/A')} Cr\n"
            report += f"- **Challenger:** {challenger_metrics['cost_mae']:.2f} Cr\n"
            report += f"- **Improvement:** {comparison_results['cost_improvement']:+.2f} Cr "
            report += f"({comparison_results['cost_improvement_pct']:+.2f}%)\n\n"
        
        report += "---\n\n"
        
        # Add decision
        report += "## Decision\n\n"
        if promoted:
            report += "**✓ CHALLENGER PROMOTED TO CHAMPION**\n\n"
            report += f"- New champion model saved to: `{self.champion_model_path}`\n"
            report += f"- Dataset version: {self.dataset_versions['current_version']}\n"
            report += f"- MLflow run ID: `{run_id}`\n"
        else:
            report += "**✗ CHAMPION RETAINED**\n\n"
            report += "The new model did not improve both delay and cost predictions.\n"
            report += f"- Challenger archived with timestamp\n"
            report += f"- MLflow run ID: `{run_id}`\n"
        
        report += "\n---\n\n"
        
        # Add MLflow link
        report += "## MLflow Tracking\n\n"
        report += f"View detailed metrics and artifacts:  \n"
        report += f"🔗 [http://127.0.0.1:5000](http://127.0.0.1:5000)\n\n"
        report += f"**Run ID:** `{run_id}`\n"
        
        report += "\n---\n\n"
        report += "*Report generated by GridSight Retraining Pipeline*\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Retraining report saved: {report_path}")
        return report_path
    
    def run(self, feedback_threshold=100):
        """
        Execute the complete retraining pipeline.
        
        Parameters:
        -----------
        feedback_threshold : int
            Minimum number of feedback rows to trigger retraining
        """
        print("\n" + "="*70)
        print("GRIDSIGHT RETRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Check retraining trigger
        should_retrain, num_feedback = self.check_retraining_trigger(feedback_threshold)
        
        if not should_retrain:
            print("\n" + "="*70)
            print("RETRAINING SKIPPED")
            print("="*70)
            return None
        
        # Step 2: Merge datasets
        merged_df, new_version = self.merge_datasets()
        
        # Step 3: Retrain models
        forecaster, phase_metrics, overall_metrics, run_id = self.retrain_models(
            merged_df, new_version
        )
        
        # Step 4: Get champion metrics
        champion_metrics = self.get_champion_metrics()
        
        # Step 5: Compare models
        promoted, comparison_results = self.compare_models(champion_metrics, overall_metrics)
        
        # Step 6: Promote or archive
        if promoted:
            self.promote_challenger(overall_metrics, run_id)
        else:
            self.save_challenger(run_id)
        
        # Step 7: Generate report
        if champion_metrics:
            comparison_results['champion_delay_mae'] = champion_metrics['overall_mae_delay']
            comparison_results['champion_cost_mae'] = champion_metrics['overall_mae_cost']
        
        report_path = self.generate_retraining_report(
            comparison_results, overall_metrics, run_id, promoted
        )
        
        # Final summary
        print("\n" + "="*70)
        print("RETRAINING PIPELINE COMPLETE")
        print("="*70)
        print("\nDeliverables:")
        print(f"  ✓ Retrained models with dataset {new_version}")
        print(f"  ✓ MLflow run ID: {run_id}")
        print(f"  ✓ Retraining report: {report_path}")
        if promoted:
            print(f"  ✓ New champion promoted: {self.champion_model_path}")
        else:
            print(f"  ✓ Champion retained (challenger archived)")
        print(f"\n🔗 View in MLflow: http://127.0.0.1:5000")
        
        return {
            'promoted': promoted,
            'run_id': run_id,
            'dataset_version': new_version,
            'report_path': report_path
        }


def main():
    """
    Execute the retraining pipeline.
    """
    pipeline = GridSightRetrainingPipeline()
    result = pipeline.run(feedback_threshold=100)
    
    if result:
        print("\n✓ Retraining completed successfully!")
    else:
        print("\n✓ Retraining skipped (insufficient feedback data)")


if __name__ == "__main__":
    main()

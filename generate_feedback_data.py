"""
Helper script to simulate feedback data for testing retraining pipeline.

This script copies N random rows from the original dataset to
new_feedback_data.csv to simulate user-validated predictions.
"""

import pandas as pd
import os

def create_sample_feedback(num_rows=120, seed=42):
    """
    Create sample feedback data for retraining testing.
    
    Parameters:
    -----------
    num_rows : int
        Number of rows to copy (default 120 to trigger retraining)
    seed : int
        Random seed for reproducibility
    """
    print("="*60)
    print("GRIDSIGHT FEEDBACK DATA GENERATOR")
    print("="*60)
    
    # Load original dataset
    original_path = 'Data/gridsight_dataset.csv'
    feedback_path = 'Data/new_feedback_data.csv'
    
    if not os.path.exists(original_path):
        print(f"\n❌ Error: {original_path} not found!")
        print("   Make sure you're running from the GridSight root directory.")
        return
    
    print(f"\nLoading original dataset: {original_path}")
    df = pd.read_csv(original_path)
    print(f"  Total rows available: {len(df)}")
    
    # Sample random rows
    if num_rows > len(df):
        print(f"\n⚠ Warning: Requested {num_rows} rows but only {len(df)} available.")
        num_rows = len(df)
    
    print(f"\nSampling {num_rows} random rows...")
    sample_df = df.sample(n=num_rows, random_state=seed)
    
    # Save to feedback file
    print(f"Saving to: {feedback_path}")
    sample_df.to_csv(feedback_path, index=False)
    
    print("\n" + "="*60)
    print("✅ FEEDBACK DATA CREATED SUCCESSFULLY")
    print("="*60)
    print(f"\nGenerated {num_rows} feedback rows")
    print(f"Saved to: {feedback_path}")
    print("\nNext steps:")
    print("  1. Review the feedback data (optional)")
    print("  2. Run: python retrain_pipeline.py")
    print("  3. Check retraining_report.md for results")
    
    # Show sample
    print("\n" + "-"*60)
    print("Sample of generated feedback (first 3 rows):")
    print("-"*60)
    print(sample_df.head(3)[['Project_ID', 'Phase_Num', 'Phase_Name', 
                              'Cumulative_Delay_Days', 'Cumulative_Cost_Overrun_Cr']].to_string())
    
    return sample_df

def clear_feedback_data():
    """
    Clear existing feedback data (reset to empty template).
    """
    feedback_path = 'Data/new_feedback_data.csv'
    original_path = 'Data/gridsight_dataset.csv'
    
    if not os.path.exists(original_path):
        print(f"❌ Error: {original_path} not found!")
        return
    
    # Load original to get headers
    df = pd.read_csv(original_path)
    
    # Create empty DataFrame with same columns
    empty_df = pd.DataFrame(columns=df.columns)
    
    # Save empty template
    empty_df.to_csv(feedback_path, index=False)
    
    print("="*60)
    print("✅ FEEDBACK DATA CLEARED")
    print("="*60)
    print(f"Reset {feedback_path} to empty template")

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("GRIDSIGHT FEEDBACK DATA UTILITY")
    print("="*60)
    print("\nOptions:")
    print("  1. Create 120 feedback rows (triggers retraining)")
    print("  2. Create 50 feedback rows (below threshold)")
    print("  3. Create 200 feedback rows (large feedback set)")
    print("  4. Clear feedback data (reset to empty)")
    print("  5. Exit")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            create_sample_feedback(num_rows=120)
        elif choice == '2':
            create_sample_feedback(num_rows=50)
        elif choice == '3':
            create_sample_feedback(num_rows=200)
        elif choice == '4':
            clear_feedback_data()
        elif choice == '5':
            print("\nExiting...")
        else:
            print("\n❌ Invalid choice. Please run again and select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

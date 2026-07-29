import os
import json
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import openpyxl

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# MINIMAL PROMPT TEMPLATE with History Accumulation logic and escaped braces
PROMPT_TEMPLATE = """Generate structured training data for the POWERGRID project named: "{project_name}".

Your task:
- Search web sources (POWERGRID website, tenders, EPC contracts, clearance reports, GIS data, historical patterns).
- Extract ANY real data available.
- If any field is missing, infer a realistic value based on:
  • project type
  • location
  • similar POWERGRID projects
  • vendor patterns
  • weather and terrain norms
- NEVER refuse, NEVER leave blank fields.
- ALWAYS output STRICT VALID JSON.

OUTPUT REQUIREMENTS:
- Output ONLY a JSON object (no markdown, no commentary, no text).
- The JSON must contain EXACTLY the 32 fields below.

HISTORY ACCUMULATION RULE (CRITICAL):For each object in the array where Phase_Num is $N$, you MUST include all "Actual" and "Planned" data fields for all phases from $1$ to $N$.This ensures that the record for Phase 3 contains the history of Phase 1 and Phase 2.Fields for future phases ($>N$) should be set to null

Features

Metadata
1. Project_ID: string (use project name as ID)
2. Project_Type: "Substation" | "Line" (transmission line or substation)
3. Voltage_Level: "400kV" | "765kV"
4. Project_Location: "State, District" (e.g., "Karnataka, Raichur")
5. Project_Size: number (MVA for substation, Ckm for line)
   - Small: <100 MVA or <100 Ckm
   - Medium: 100-500 MVA or 100-300 Ckm
   - Large: >500 MVA or >300 Ckm
6. Contract_Type: "EPC" | "TBCB"
   - EPC: Engineering, Procurement, Construction
   - TBCB: Tariff Based Competitive Bidding
7. Planned_Duration: number (total months, typically 18-36)
8. Total_Budget_Cr: number (total budget in Crores, typically 100-500 Cr)
9. Start_Date: "YYYY-MM-DD" (project start date)

PHASE 1: Planning/Engineering
------------------------------
- Engg_%Complete: 100
- Engg_Planned_Days: 60-120 (typical engineering phase)
- Engg_Actual_Days: >= Engg_Planned_Days - 5 (slight early finish possible)
- Engg_Planned_Cost_Cr: ~10 percent of Total_Budget_Cr
- Engg_Actual_Cost_Cr: 0.9x to 1.3x of Engg_Planned_Cost_Cr
- DPR_Status: "Approved" | "Pending" | "Revised"
- Design_Changes: 0 | 1 (20 percent chance of 1)

PHASE 2: Approvals & ROW
-------------------------
- ROW_%Complete: 100
- ROW_Planned_Days: 120-180 (longest phase, approvals take time)
- ROW_Actual_Days: >= ROW_Planned_Days (rarely finishes early)
- ROW_Planned_Cost_Cr: ~15 percent  of Total_Budget_Cr
- ROW_Actual_Cost_Cr: 0.9x to 1.8x (can have major overruns)
- Forest_Clearance: 0 | 1 (depends on terrain)
- Statutory_Approvals: "Complete" | "Partial" | "Pending"

PHASE 3: Procurement/Supply
----------------------------
- Supply_%Complete: 100
- Supply_Planned_Days: 45-90
- Supply_Actual_Days: >= Supply_Planned_Days - 5
- Supply_Planned_Cost_Cr: ~25 percent of Total_Budget_Cr
- Supply_Actual_Cost_Cr: 0.95x to 1.4x (material price fluctuations)
- Orders_Placed: 0 | 1
- Equip_Delivery_Status: "OnTime" | "Delayed" | "Critical"

PHASE 4: Site Preparation/Civil
--------------------------------
- Civil_%Complete: 100
- Civil_Planned_Days: 60-120
- Civil_Actual_Days: >= Civil_Planned_Days (weather-dependent)
- Civil_Planned_Cost_Cr: ~15 percent of Total_Budget_Cr
- Civil_Actual_Cost_Cr: 0.9x to 1.5x (terrain impacts cost)
- Foundations_Complete: 85-100 (foundation completion %)
- Site_Access: 0 | 1

PHASE 5: Erection/Construction
-------------------------------
- Erection_%Complete: 100
- Erection_Planned_Days: 90-150 (labor-intensive)
- Erection_Actual_Days: >= Erection_Planned_Days
- Erection_Planned_Cost_Cr: ~20 percent of Total_Budget_Cr
- Erection_Actual_Cost_Cr: 0.9x to 1.4x
- Towers_Erected_Pct: 90-100
- Structures_Complete: 0 | 1

PHASE 6: Stringing/Cabling
---------------------------
- Stringing_%Complete: 100
- Stringing_Planned_Days: 60-100
- Stringing_Actual_Days: >= Stringing_Planned_Days - 5
- Stringing_Planned_Cost_Cr: ~10 percent of Total_Budget_Cr
- Stringing_Actual_Cost_Cr: 0.9x to 1.3x
- Conductor_Strung_Pct: 95-100
- Terrain_Type: "Flat" | "Hilly" | "Forest"

PHASE 7: Testing & Commissioning
---------------------------------
- Testing_%Complete: 100
- Testing_Planned_Days: 30-60
- Testing_Actual_Days: >= Testing_Planned_Days - 5
- Testing_Planned_Cost_Cr: ~5 percent of Total_Budget_Cr
- Testing_Actual_Cost_Cr: 0.95x to 1.2x
- Trial_Run_Status: "Successful" | "Failed" | "Pending"
- DOCO_Date: "YYYY-MM-DD" (calculate from Start_Date + all phases)

CUMULATIVE TARGETS (Calculate from phase data):
------------------------------------------------
For EACH phase N (1 to 7), calculate:

Cumulative_Delay_Days = SUM of (Actual_Days - Planned_Days) for phases 1 to N
Cumulative_Cost_Overrun_Cr = SUM of (Actual_Cost_Cr - Planned_Cost_Cr) for phases 1 to N

CRITICAL: These MUST be monotonically non-decreasing!


JSON SCHEMA (FILL ALL):


[
  {{
    "Project_ID": "string",
    "Project_Type": "Line|Substation",
    "Voltage_Level": "400kV|765kV",
    "Project_Location": "string",
    "Project_Size": number,
    "Contract_Type": "EPC|TBCB",
    "Planned_Duration": number,
    "Total_Budget_Cr": number,
    "Start_Date": "DD-MM-YYYY",
    
    "Phase_Num": 1,
    "Phase_Name": "Planning/Engineering",
    "Engg_%Complete": 100,
    "Engg_Planned_Days": number,
    "Engg_Actual_Days": number,
    "Engg_Planned_Cost_Cr": number,
    "Engg_Actual_Cost_Cr": number,
    "DPR_Status": "string",
    "Design_Changes": 0|1,
    
    "Cumulative_Delay_Days": number,
    "Cumulative_Cost_Overrun_Cr": number
  }},
  {{
    "Project_ID": "string",
    ... (same metadata)
    
    "Phase_Num": 2,
    "Phase_Name": "Approvals & ROW",
    "ROW_%Complete": 100,
    "ROW_Planned_Days": number,
    "ROW_Actual_Days": number,
    "ROW_Planned_Cost_Cr": number,
    "ROW_Actual_Cost_Cr": number,
    "Forest_Clearance": 0|1,
    "Statutory_Approvals": "string",
    
    "Cumulative_Delay_Days": number,
    "Cumulative_Cost_Overrun_Cr": number
  }},
  ... (phases 3-7 follow same pattern)
]

(Continue for phases 3-7 with monotonically increasing cumulative values)
All fields labeled Actual_Days and Actual_Cost_Cr must represent only the time and money spent within that specific phase. They are NOT cumulative sums.

Before outputting JSON, verify:

✓ Cumulative_Delay_Days[N] >= Cumulative_Delay_Days[N-1]
✓ Cumulative_Cost_Overrun_Cr[N] >= Cumulative_Cost_Overrun_Cr[N-1]
✓ SUM of all phase Planned_Cost_Cr = Total_Budget_Cr (within 1%)
✓ All Actual_Days >= Planned_Days - 5 (small early finish allowed)
✓ Final Cumulative_Delay_Days <= Planned_Duration * 30 * 0.5 (max 50 percent delay)
✓ Final Cumulative_Cost_Overrun_Cr <= Total_Budget_Cr * 0.4 (max 40 percent overrun)
✓ All completion % = 100 (we only track completed phases)
✓ All numeric fields >= 0

Apply domain knowledge:

IF Forest_Clearance = 1:
  → Phase 2 (ROW) should have 30-50 percent delay
  → ROW_Actual_Cost_Cr should be 1.3-1.8x planned

IF Terrain_Type = "Hilly" OR "Forest":
  → Phase 4 (Civil) should have 20-40 percent delay
  → Civil_Actual_Cost_Cr should be 1.2-1.5x planned
  → Phase 5 (Erection) should also have delays

IF Contract_Type = "TBCB":
  → Generally better cost control (1.0-1.2x planned costs)
  → Phase execution more disciplined

IF Voltage_Level = "765kV":
  → Higher complexity, more delays in Phase 5-7
  → Equipment delays more likely in Phase 3

Overall distribution (realistic):
- 60% projects: final delay < 15%, cost overrun < 10%
- 30% projects: final delay 15-30%, cost overrun 10-25%
- 10% projects: final delay > 30%, cost overrun > 25%


RULES:
- Use DD-MM-YYYY format for all dates.
- Do not include percentage sign, keep it numerical between 0 to 100.
- Numeric fields must NOT include commas.
- JSON must be clean, valid and machine-parsable.
- DO NOT reuse data from previous examples.
- DO NOT output placeholders; fill each field with the best available or inferred value.
- If you cannot find exact ddata, 
 Using only verified project milestones from official sources
 POWERGRID's published project reports
 anonymized historical data from completed projects rather than fabricating it
"""

def extract_json_array(response_content):
    """Aggressively parse JSON array from model response."""
    try:
        content = response_content.strip()
        
        # Strategy 1: Look for JSON array
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx+1]
            try:
                result = json.loads(json_str)
                # Validate all items are dictionaries
                if isinstance(result, list):
                    validated = [item for item in result if isinstance(item, dict)]
                    if validated:
                        return validated
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON object (single object instead of array)
        start_obj = content.find('{')
        end_obj = content.rfind('}')
        
        if start_obj != -1 and end_obj != -1:
            json_str = content[start_obj:end_obj+1]
            try:
                obj = json.loads(json_str)
                # Wrap single object in array only if it's a dict
                return [obj] if isinstance(obj, dict) else []
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Remove markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    validated = [item for item in result if isinstance(item, dict)]
                    return validated if validated else []
                elif isinstance(result, dict):
                    return [result]
                else:
                    return []
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Try parsing the entire content
        try:
            result = json.loads(content)
            if isinstance(result, list):
                validated = [item for item in result if isinstance(item, dict)]
                return validated if validated else []
            elif isinstance(result, dict):
                return [result]
            else:
                return []
        except json.JSONDecodeError:
            pass
        
        # All strategies failed - print response and return empty
        logger.error("❌ JSON Parsing failed with all strategies")
        print("\n" + "="*80)
        print("RAW API RESPONSE (JSON parsing failed):")
        print("="*80)
        print(response_content)
        print("="*80 + "\n")
        return []
        
    except Exception as e:
        logger.error(f"JSON Parsing failed: {e}")
        print("\n" + "="*80)
        print("RAW API RESPONSE (Exception occurred):")
        print("="*80)
        print(response_content)
        print("="*80 + "\n")
        return []

def get_project_data(project_name):
    """Fetch project data from Perplexity API."""
    prompt = PROMPT_TEMPLATE.format(project_name=project_name)
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        content = response.choices[0].message.content
        return extract_json_array(content)
        
    except Exception as e:
        logger.error(f"API Error for {project_name}: {e}")
        return []

def main():
    # Read project names from Excel file
    excel_path = r"C:\Users\Tejas\Desktop\GridSight\Failed Projects.xlsx"
    
    try:
        # Read Excel file and extract project names
        df_projects = pd.read_excel(excel_path)
        # Assuming project names are in the first column
        # Adjust column name if different
        project_col = df_projects.columns[0]
        projects = df_projects[project_col].dropna().tolist()
        logger.info(f"📋 Loaded {len(projects)} projects from Excel file")
    except FileNotFoundError:
        logger.error(f"Excel file not found at: {excel_path}")
        projects = []
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        projects = []
    
    if not projects:
        logger.error("No projects to process. Exiting.")
        return
    
    # Batch processing parameters
    BATCH_SIZE = 30
    total_projects = len(projects)
    completed_projects = []
    failed_projects = []
    remaining_projects = projects.copy()
    
    batch_num = 1
    start_idx = 0
    
    while start_idx < total_projects:
        end_idx = min(start_idx + BATCH_SIZE, total_projects)
        batch_projects = projects[start_idx:end_idx]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 Processing Batch {batch_num} ({len(batch_projects)} projects)")
        logger.info(f"📊 Progress: {start_idx}/{total_projects} completed")
        logger.info(f"{'='*80}\n")
        
        batch_rows = []
        batch_failed = []
        
        for project in tqdm(batch_projects, desc=f"Batch {batch_num}"):
            data = get_project_data(project)
            if data:
                # Extra validation: ensure all items are dictionaries
                valid_data = [item for item in data if isinstance(item, dict)]
                if valid_data:
                    batch_rows.extend(valid_data)
                    completed_projects.append(project)
                else:
                    batch_failed.append(project)
                    failed_projects.append(project)
                    logger.warning(f"⚠️ Invalid data format for: {project}")
            else:
                batch_failed.append(project)
                failed_projects.append(project)
                logger.warning(f"⚠️ JSON parsing failed for: {project}")
            time.sleep(2)  # Rate limiting
        
        # Update remaining projects
        remaining_projects = projects[end_idx:]
        
        # Save batch results
        if batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            
            # Sort and process batch data
            df_batch = df_batch.sort_values(['Project_ID', 'Phase_Num'])
            df_batch = df_batch.groupby('Project_ID', group_keys=False).apply(lambda x: x.ffill())
            
            # Organize columns
            cols = ["Project_ID", "Phase_Num", "Phase_Name", "Cumulative_Delay_Days", "Cumulative_Cost_Overrun_Cr"]
            other_cols = [c for c in df_batch.columns if c not in cols]
            df_batch = df_batch[cols + other_cols]
            
            # Save draft for review
            draft_filename = f'draft_batch_{batch_num}.csv'
            df_batch.to_csv(draft_filename, index=False)
            logger.info(f"\n✅ Batch {batch_num} complete!")
            logger.info(f"📄 Draft saved to: {draft_filename}")
            logger.info(f"📊 Batch stats: {len(batch_rows)} rows from {len(batch_projects) - len(batch_failed)} successful projects")
        else:
            logger.warning(f"\n⚠️ No data collected in Batch {batch_num}")
        
        # Save failed projects for this batch
        if batch_failed:
            failed_filename = f'failed_projects_batch_{batch_num}.txt'
            with open(failed_filename, 'w', encoding='utf-8') as f:
                f.write(f"Failed Projects from Batch {batch_num}:\n")
                f.write("="*80 + "\n")
                for proj in batch_failed:
                    f.write(f"{proj}\n")
            logger.warning(f"❌ {len(batch_failed)} failed projects saved to: {failed_filename}")
        
        # Show status
        logger.info(f"\n{'='*80}")
        logger.info(f"📈 CURRENT STATUS:")
        logger.info(f"  ✅ Completed: {len(completed_projects)}/{total_projects}")
        logger.info(f"  ❌ Failed: {len(failed_projects)}")
        logger.info(f"  ⏳ Remaining: {len(remaining_projects)}")
        logger.info(f"{'='*80}\n")
        
        # Check if there are more projects
        if end_idx >= total_projects:
            logger.info("🎉 All projects processed!")
            break
        
        # Ask user to continue
        print("\n" + "="*80)
        print(f"✋ CHECKPOINT - Batch {batch_num} Complete")
        print(f"📊 Processed: {end_idx}/{total_projects} projects")
        print(f"📄 Review draft file: {draft_filename}")
        print("="*80)
        user_input = input("\nDo you want to APPEND this batch to gridsight_dataset.csv and continue? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            # Append to main CSV
            if batch_rows:
                main_csv = 'gridsight_dataset.csv'
                if os.path.exists(main_csv):
                    # Append to existing
                    df_batch.to_csv(main_csv, mode='a', header=False, index=False)
                    logger.info(f"✅ Batch {batch_num} appended to {main_csv}")
                else:
                    # Create new
                    df_batch.to_csv(main_csv, index=False)
                    logger.info(f"✅ New {main_csv} created with Batch {batch_num}")
            
            # Continue to next batch
            start_idx = end_idx
            batch_num += 1
            logger.info(f"\n➡️ Continuing to next batch...\n")
        else:
            # Stop execution
            logger.info("\n🛑 Execution stopped by user.")
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 FINAL SUMMARY:")
            logger.info(f"  ✅ Completed Projects ({len(completed_projects)}):")
            for proj in completed_projects:
                logger.info(f"     • {proj}")
            logger.info(f"\n  ⏳ Remaining Projects ({len(remaining_projects)}):")
            for proj in remaining_projects:
                logger.info(f"     • {proj}")
            if failed_projects:
                logger.info(f"\n  ❌ Failed Projects ({len(failed_projects)}):")
                for proj in failed_projects:
                    logger.info(f"     • {proj}")
            logger.info(f"{'='*80}\n")
            break
    
    # Final summary of all failed projects
    if failed_projects:
        all_failed_filename = 'all_failed_projects.txt'
        with open(all_failed_filename, 'w', encoding='utf-8') as f:
            f.write(f"All Failed Projects (Total: {len(failed_projects)}):\n")
            f.write("="*80 + "\n")
            for proj in failed_projects:
                f.write(f"{proj}\n")
        logger.info(f"\n📝 Complete list of failed projects saved to: {all_failed_filename}")

if __name__ == "__main__":
    main()
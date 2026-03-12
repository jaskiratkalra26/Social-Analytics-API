import os
import json
import glob

RESULTS_DIR = "test_results"

def analyze_results():
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    print(f"Found {len(result_files)} result files.")
    
    summary = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "errors": []
    }
    
    for filepath in result_files:
        summary["total"] += 1
        filename = os.path.basename(filepath)
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            if "error" in data:
                summary["failed"] += 1
                summary["errors"].append(f"{filename}: {data['error']}")
                continue
                
            # key validation
            required_keys = ["metadata", "hook_analysis", "pacing_analysis", "lighting_analysis", "text_analysis", "content_classification"]
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                summary["failed"] += 1
                summary["errors"].append(f"{filename}: Missing keys {missing}")
                continue
            
            # Deep check - check for empty analysis where it should likely not be empty
            # e.g. metadata duration should be > 0
            if data.get("metadata", {}).get("duration", 0) <= 0:
                 summary["failed"] += 1
                 summary["errors"].append(f"{filename}: Duration is 0 or missing")
                 continue
                 
            # Content Classification Check
            content_classification = data.get("content_classification", {})
            if not content_classification or "error" in content_classification:
                 # Check if it was an error or just empty
                 if "error" in content_classification:
                     summary["failed"] += 1
                     summary["errors"].append(f"{filename}: Content Classification Error: {content_classification['error']}")
                     continue
            
            # Hook Analysis Check
            hook = data.get("hook_analysis", {})
            if not hook or "error" in hook:
                 summary["errors"].append(f"{filename}: Hook Analysis missing or error")

            summary["success"] += 1
            
            # Print a concise summary of the content for inspection
            print(f"\n--- {filename} ---")
            print(f"duration: {data['metadata'].get('duration', 'N/A')}s, fps: {data['metadata'].get('fps', 'N/A')}")
            print(f"hook_score: {hook.get('hook_efficacy_score', hook.get('hook_score', 'N/A'))}")
            print(f"pace: {data.get('pacing_analysis', {}).get('pace_description', data.get('pacing_analysis', {}).get('pace_category', 'N/A'))}")
            print(f"lighting: {data.get('lighting_analysis', {}).get('lighting_quality', data.get('lighting_analysis', {}).get('lighting_category', 'N/A'))}")
            print(f"text_overlay: {len(data.get('text_analysis', {}).get('text_content', []))} items detected")
            print(f"classification: {content_classification.get('virality_score_prediction', 'N/A')}")


        except json.JSONDecodeError:
             summary["failed"] += 1
             summary["errors"].append(f"{filename}: Invalid JSON")
        except Exception as e:
             summary["failed"] += 1
             summary["errors"].append(f"{filename}: Check failed - {str(e)}")

    print("-" * 30)
    print(f"Total Files Analyzed: {summary['total']}")
    print(f"Successful: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    if summary["errors"]:
        print("\nErrors Found:")
        for err in summary["errors"]:
            print(f" - {err}")
    else:
        print("\nAll files passed validation check.")
    print("-" * 30)

if __name__ == "__main__":
    analyze_results()

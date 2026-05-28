import json
import time
import os

class ResearchExperimentTracker:
    """
    Automatically saves configurations, metrics, runtime, FPS, failure counts, and experiment IDs.
    """
    def __init__(self, base_dir: str = "results/evaluation/experiment_logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.exp_id = f"exp_{int(time.time())}"
        self.log_dir = os.path.join(self.base_dir, self.exp_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def save_config(self, config: dict):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
    def save_metrics(self, metrics: dict, suffix: str = ""):
        # Save JSON
        with open(os.path.join(self.log_dir, f"metrics{suffix}.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Save Markdown Summary
        summary = metrics.get("summary", {})
        md_content = f"# Experiment {self.exp_id} Summary{suffix}\n\n"
        md_content += "## Core Stability Metrics\n"
        md_content += f"- **Identity Stability Score:** {summary.get('identity_stability_score', 0):.4f}\n"
        md_content += f"- **Total Visible Switches:** {summary.get('total_visible_switches', 0)}\n"
        md_content += f"- **Duplicate Box Frames:** {summary.get('duplicate_box_frames', 0)}\n"
        md_content += f"- **Teleportation Events:** {summary.get('teleportation_events', 0)}\n"
        md_content += f"- **Fragmentation Count:** {summary.get('fragmentation_count', 0)}\n\n"
        md_content += "---\n*Generated automatically by SELFWATCH Evaluation Suite.*\n"
        
        with open(os.path.join(self.log_dir, f"summary{suffix}.md"), "w") as f:
            f.write(md_content)
            
    def log_runtime_stats(self, fps: float, total_time: float):
        stats = {
            "fps": fps,
            "total_runtime_seconds": total_time
        }
        with open(os.path.join(self.log_dir, "runtime_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

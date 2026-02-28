import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class TrainingReportGenerator:
    """Generate comprehensive training reports."""
    
    def __init__(self, run_id: str, output_dir: str = "./reports"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.report_data = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "sections": {}
        }
        
    def add_section(self, name: str, data: Dict[str, Any]):
        """Add a section to the report."""
        self.report_data["sections"][name] = data
        
    def add_training_config(self, config: Dict[str, Any]):
        """Add training configuration."""
        self.report_data["training_config"] = config
        
    def add_data_statistics(self, stats: Dict[str, Any]):
        """Add data statistics."""
        self.report_data["data_statistics"] = stats
        
    def add_training_metrics(self, metrics: Dict[str, Any]):
        """Add training metrics."""
        self.report_data["training_metrics"] = metrics
        
    def add_model_info(self, model_info: Dict[str, Any]):
        """Add model information."""
        self.report_data["model_info"] = model_info
        
    def add_safety_performance(self, safety_stats: Dict[str, Any]):
        """Add safety performance metrics."""
        self.report_data["safety_performance"] = safety_stats
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = {}
        
        if "data_statistics" in self.report_data:
            ds = self.report_data["data_statistics"]
            summary["total.get("total_samples_samples"] = ds", 0)
            summary["crisis_samples"] = ds.get("crisis_samples", 0)
            summary["crisis_ratio"] = ds.get("crisis_ratio", 0)
            summary["subreddits"] = len(ds.get("subreddit_distribution", {}))
            
        if "training_metrics" in self.report_data:
            tm = self.report_data["training_metrics"]
            summary["final_train_loss"] = tm.get("avg_train_loss", 0)
            summary["final_val_loss"] = tm.get("avg_val_loss", 0)
            summary["crisis_accuracy"] = tm.get("crisis_accuracy", 0)
            summary["safety_score"] = tm.get("safety_score", 0)
            
        if "training_config" in self.report_data:
            tc = self.report_data["training_config"]
            summary["epochs"] = tc.get("num_epochs", 0)
            summary["batch_size"] = tc.get("batch_size", 0)
            summary["learning_rate"] = tc.get("learning_rate", 0)
            
        return summary
        
    def save_report(self, filename: Optional[str] = None):
        """Save the complete report."""
        if filename is None:
            filename = f"report_{self.run_id}.json"
            
        self.report_data["summary"] = self.generate_summary()
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
            
        print(f"Report saved to: {output_path}")
        return output_path
        
    def save_markdown(self, filename: Optional[str] = None) -> str:
        """Save report as Markdown."""
        if filename is None:
            filename = f"report_{self.run_id}.md"
            
        output_path = self.output_dir / filename
        
        md_content = self._generate_markdown()
        
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        print(f"Markdown report saved to: {output_path}")
        return str(output_path)
        
    def _generate_markdown(self) -> str:
        """Generate Markdown report content."""
        md = []
        md.append(f"# Mental Health Support Model - Training Report")
        md.append(f"")
        md.append(f"**Run ID:** {self.run_id}")
        md.append(f"**Generated:** {self.report_data['generated_at']}")
        md.append(f"")
        
        summary = self.report_data.get("summary", {})
        
        md.append(f"## Executive Summary")
        md.append(f"")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        
        for key, value in summary.items():
            if isinstance(value, float):
                md.append(f"| {key} | {value:.4f} |")
            else:
                md.append(f"| {key} | {value} |")
                
        md.append(f"")
        
        if "data_statistics" in self.report_data:
            md.append(f"## Data Statistics")
            md.append(f"")
            ds = self.report_data["data_statistics"]
            
            md.append(f"- **Total Samples:** {ds.get('total_samples', 0):,}")
            md.append(f"- **Posts:** {ds.get('total_posts', 0):,}")
            md.append(f"- **Comments:** {ds.get('total_comments', 0):,}")
            md.append(f"- **Crisis Samples:** {ds.get('crisis_samples', 0):,}")
            md.append(f"- **Crisis Ratio:** {ds.get('crisis_ratio', 0):.2%}")
            md.append(f"")
            
            md.append(f"### Response Types")
            md.append(f"")
            resp_types = ds.get("response_types", {})
            for resp_type, count in resp_types.items():
                md.append(f"- {resp_type}: {count:,}")
            md.append(f"")
            
            md.append(f"### Subreddit Distribution")
            md.append(f"")
            sub_dist = ds.get("subreddit_distribution", {})
            sorted_subs = sorted(sub_dist.items(), key=lambda x: x[1], reverse=True)[:15]
            for sub, count in sorted_subs:
                md.append(f"- {sub}: {count:,}")
            md.append(f"")
            
        if "training_metrics" in self.report_data:
            md.append(f"## Training Metrics")
            md.append(f"")
            tm = self.report_data["training_metrics"]
            
            md.append(f"| Metric | Value |")
            md.append(f"|--------|-------|")
            md.append(f"| Avg Train Loss | {tm.get('avg_train_loss', 0):.4f} |")
            md.append(f"| Avg Val Loss | {tm.get('avg_val_loss', 0):.4f} |")
            md.append(f"| Crisis Accuracy | {tm.get('crisis_accuracy', 0):.4f} |")
            md.append(f"| Quality Correlation | {tm.get('quality_correlation', 0):.4f} |")
            md.append(f"| Safety Score | {tm.get('safety_score', 0):.4f} |")
            md.append(f"")
            
        if "training_config" in self.report_data:
            md.append(f"## Training Configuration")
            md.append(f"")
            tc = self.report_data["training_config"]
            
            md.append(f"| Parameter | Value |")
            md.append(f"|-----------|-------|")
            for key, value in tc.items():
                md.append(f"| {key} | {value} |")
            md.append(f"")
            
        if "model_info" in self.report_data:
            md.append(f"## Model Information")
            md.append(f"")
            mi = self.report_data["model_info"]
            
            md.append(f"- **Architecture:** {mi.get('architecture', 'N/A')}")
            md.append(f"- **Parameters:** {mi.get('num_parameters', 'N/A'):,}")
            md.append(f"- **Encoders:** {', '.join(mi.get('encoders', []))}")
            md.append(f"")
            
        if "safety_performance" in self.report_data:
            md.append(f"## Safety Performance")
            md.append(f"")
            sp = self.report_data["safety_performance"]
            
            md.append(f"| Metric | Value |")
            md.append(f"|--------|-------|")
            for key, value in sp.items():
                if isinstance(value, float):
                    md.append(f"| {key} | {value:.4f} |")
                else:
                    md.append(f"| {key} | {value} |")
            md.append(f"")
            
        md.append(f"---")
        md.append(f"*Report generated by Mental Health Support Model Training System*")
        
        return "\n".join(md)


class JSONLogger:
    """JSON logger for training runs."""
    
    def __init__(self, log_dir: str = "./logs", run_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = run_id
        
        self.log_file = self.log_dir / f"{run_id}.json"
        
        self.log_data = {
            "run_id": run_id,
            "status": "initialized",
            "start_time": datetime.now().isoformat(),
            "events": [],
            "metrics": {},
            "errors": []
        }
        
        self._save()
        
    def log_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Log an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data or {}
        }
        
        self.log_data["events"].append(event)
        self._save()
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if name not in self.log_data["metrics"]:
            self.log_data["metrics"][name] = []
            
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "value": value
        }
        
        if step is not None:
            metric_entry["step"] = step
            
        self.log_data["metrics"][name].append(metric_entry)
        self._save()
        
    def log_error(self, error: str, details: Optional[Dict] = None):
        """Log an error."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "details": details or {}
        }
        
        self.log_data["errors"].append(error_entry)
        self._save()
        
    def update_status(self, status: str):
        """Update run status."""
        self.log_data["status"] = status
        self.log_data["last_update"] = datetime.now().isoformat()
        self._save()
        
    def _save(self):
        """Save log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
    def get_log_path(self) -> str:
        """Get the log file path."""
        return str(self.log_file)
    
    def finalize(self, final_metrics: Optional[Dict] = None):
        """Finalize the log."""
        self.log_data["status"] = "completed"
        self.log_data["end_time"] = datetime.now().isoformat()
        
        if final_metrics:
            self.log_data["final_metrics"] = final_metrics
            
        self._save()


def create_report_from_logs(
    log_dir: str,
    run_id: str,
    output_dir: str = "./reports"
) -> str:
    """Create a report from existing logs."""
    
    log_file = Path(log_dir) / f"{run_id}.json"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
        
    with open(log_file) as f:
        log_data = json.load(f)
        
    generator = TrainingReportGenerator(run_id, output_dir)
    
    generator.log_data = log_data
    
    report_path = generator.save_report()
    generator.save_markdown()
    
    return report_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        create_report_from_logs("./logs", run_id, "./reports")
    else:
        print("Usage: python report_generator.py <run_id>")

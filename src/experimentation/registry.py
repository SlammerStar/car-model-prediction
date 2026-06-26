import json
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from src.utils import MODELS_DIR, logger

class ModelRegistry:
    """
    Manages model persistence, archiving, and promoting to production.
    Maintains registry.json manifest.
    """
    def __init__(self):
        self.production_dir = MODELS_DIR / "production"
        self.candidates_dir = MODELS_DIR / "candidates"
        self.archive_dir = MODELS_DIR / "archive"
        self.registry_path = MODELS_DIR / "registry.json"
        
        for d in [self.production_dir, self.candidates_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self._init_registry()

    def _init_registry(self):
        if not self.registry_path.exists():
            with open(self.registry_path, 'w') as f:
                json.dump({
                    "production_model": None,
                    "history": []
                }, f, indent=4)

    def _load_registry(self):
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save_registry(self, registry):
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=4)

    def save_candidate(self, experiment_id: str, model_name: str, pipeline, metadata: dict) -> Path:
        exp_dir = self.candidates_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        model_path = exp_dir / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        
        with open(exp_dir / f"{model_name}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved candidate {model_name} to {model_path}")
        return model_path

    def promote_to_production(self, candidate_path: Path, metadata: dict):
        # Archive current production if exists
        registry = self._load_registry()
        prod_model_path = self.production_dir / "pipeline.pkl"
        prod_meta_path = self.production_dir / "metadata.json"
        
        if prod_model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.archive_dir / f"pipeline_{timestamp}.pkl"
            shutil.move(str(prod_model_path), str(archive_path))
            if prod_meta_path.exists():
                shutil.move(str(prod_meta_path), str(self.archive_dir / f"metadata_{timestamp}.json"))
            logger.info(f"Archived previous production model to {archive_path}")

        # Promote new
        shutil.copy(str(candidate_path), str(prod_model_path))
        with open(prod_meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Update registry
        entry = {
            "model_name": metadata.get("model_name"),
            "timestamp": datetime.now().isoformat(),
            "metrics": metadata.get("metrics"),
            "dataset_version": metadata.get("dataset_version", "unknown"),
            "feature_version": metadata.get("feature_version", "v1.0"),
            "supported_features": metadata.get("features", [])
        }
        
        registry["production_model"] = entry
        registry["history"].append(entry)
        self._save_registry(registry)
        
        logger.info(f"Promoted {metadata.get('model_name')} to production.")

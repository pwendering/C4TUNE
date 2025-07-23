
import logging
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import subprocess


class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.timestamp}"
        self.run_dir = Path(config.paths.run_dir) / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()
        self._save_git_commit()
        self._setup_logging()

        # Optional: make the run dir accessible via config
        self.config.paths.checkpoint_dir = str(self.checkpoint_dir)

    def __enter__(self):
            return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_logging()
    
    def _save_config(self):
            config_path = self.run_dir / "config.yaml"
            OmegaConf.save(self.config, config_path)

    def _save_git_commit(self):
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
        except Exception:
            commit = "N/A"
        (self.run_dir / "commit.txt").write_text(commit)

    def _setup_logging(self):
        
        log_file = self.run_dir / "training.log"
        
        # remove any previous handlers
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        
        # file handler
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        logging.info(f"Experiment directory: {self.run_dir}")
        
    def _stop_logging(self):
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        loggers.append(logging.getLogger())
    
        for logger in loggers:
            handlers = logger.handlers[:]
            for handler in handlers:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except Exception:
                    pass
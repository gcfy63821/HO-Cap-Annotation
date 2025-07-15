from .common_imports import Union, Path, sys, logging, Optional, json, os
from omegaconf import OmegaConf, DictConfig


def add_path(path: Union[str, Path]) -> None:
    """
    Add a directory to the system path if it's not already present.

    Args:
        path (Union[str, Path]): The directory path to add to the system path.
    """
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def get_logger(
    log_name: str = "HOCapToolkit",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a logger with console and optional file output."""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s:%(funcName)s] [%(levelname).3s] %(message)s",
        datefmt="%Y%m%d;%H:%M:%S",
    )
    if not logger.hasHandlers():
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def load_config(cfg_file: Path) -> DictConfig:
    """Load configuration from file and return as an OmegaConf DictConfig object."""
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    try:
        # with cfg_file.open("r", encoding="utf-8") as f:
        #     config_dict = json.load(f)

        # # Convert the dictionary to OmegaConf's DictConfig
        # cfg = OmegaConf.create(config_dict)
        cfg = OmegaConf.load(cfg_file)

        # Update max_workers based on available CPU cores
        max_workers = os.cpu_count()  or 1
        if cfg.get("max_workers", -1) == -1:
            cfg.max_workers = max_workers // 2
        else:
            cfg.max_workers = min(cfg.max_workers, max_workers)

    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON config: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

    return cfg

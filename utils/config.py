"""
Configuration loader and manager
"""
import yaml
import torch
from typing import Any, Dict
from pathlib import Path
import sys


class Config:
    """Configuration manager with dot notation access"""
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration
        Args:
            config_dict: Dictionary containing configuration
        """
        self.config = config_dict
        self.parse_nested(config_dict)
    
    def parse_nested(self, d: Dict[str, Any], prefix: str = ""):
        """
        Parse nested dictionary and create attributes for dot notation access

        Args:
            d: Dictionary to parse
            prefix: Prefix for nested keys (used internally)
        """
        for key, value in d.items():
            if isinstance(value, dict):
                # Create nested Config object for dictionary values
                setattr(self, key, Config(value))
            else:
                # Set as regular attribute
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default values

        Args:
            key: Key Path
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary

        Returns:
            Dictionary representation of config
        """
        return self.config
    
    def __repr__(self):
        return f"Config({list(self.config.keys())})"

def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, searches for configs/config.yaml
        
    Returns:
        Config object with loaded configuration
    """
    if config_path is None:
        # Search for config.yaml in standard locations
        current = Path(__file__).resolve()
        
        # Try parent directories
        for parent in [current.parent.parent, current.parent.parent.parent]:
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Configuration file not found. "
                "Create configs/config.yaml in project root."
            )
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Resolve paths relative to project root
    project_root = config_path.parent.parent
    config_dict = resolve_paths(config_dict, project_root)
    
    # Create directories
    create_directories(config_dict, project_root)
    
    print("Configuration loaded successfully")
    
    return Config(config_dict)

def get_device(config: Config) -> torch.device:
    """
    Get torch device from configuration with automatic fallback
    
    Args:
        config: Configuration object
        
    Returns:
        torch.device object
    """
    device_str = config.system.device
    fallback = config.system.device_fallback
    
    # Check CUDA
    if device_str == "cuda":
        if torch.cuda.is_available():
            print(f"Device: cuda")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Memory: {memory_gb:.1f} GB")
            return torch.device("cuda")
        else:
            print(f"CUDA not available, falling back to {fallback}")
            return torch.device(fallback)
    
    # Check MPS (Apple Silicon)
    elif device_str == "mps":
        if torch.backends.mps.is_available():
            print(f"Device: mps")
            print(f"GPU: Apple Silicon")
            return torch.device("mps")
        else:
            print(f"MPS not available, falling back to {fallback}")
            return torch.device(fallback)
    
    # CPU
    else:
        print(f"Device: cpu")
        return torch.device("cpu")


def resolve_paths(config_dict: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    Resolve relative paths to absolute paths
    
    Args:
        config_dict: Configuration dictionary
        project_root: Project root directory
        
    Returns:
        Updated configuration dictionary
    """
    # Storage paths
    if 'storage' in config_dict:
        storage = config_dict['storage']
        for key in storage:
            if '_path' in key and isinstance(storage[key], str):
                storage[key] = str(project_root / storage[key])
    
    # Logging path
    if 'logging' in config_dict and config_dict['logging'].get('log_to_file'):
        log_path = config_dict['logging']['log_file_path']
        config_dict['logging']['log_file_path'] = str(project_root / log_path)
    
    return config_dict


def create_directories(config_dict: Dict[str, Any], project_root: Path):
    """
    Create necessary directories
    
    Args:
        config_dict: Configuration dictionary
        project_root: Project root directory
    """
    dirs_to_create = []
    
    # Storage directories
    if 'storage' in config_dict:
        storage = config_dict['storage']
        dirs_to_create.extend([
            Path(storage['face_database_path']).parent,
            Path(storage['privacy_profile_path']).parent,
            Path(storage['provenance_logs_path'])
        ])
    
    # Data directories
    dirs_to_create.extend([
        project_root / "data" / "uploads",
        project_root / "data" / "outputs"
    ])
    
    # Logs directory
    if 'logging' in config_dict and config_dict['logging'].get('log_to_file'):
        log_path = Path(config_dict['logging']['log_file_path'])
        dirs_to_create.append(log_path.parent)
    
    # Create all directories
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


def get_risk_color(config: Config, risk_level: str) -> str:
    """
    Get color code for risk level
    
    Args:
        config: Configuration object
        risk_level: Risk level ('critical', 'high', 'medium', 'low')
        
    Returns:
        Hex color code
    """
    return config.get(f'risk_levels.{risk_level}.color', default='#808080')


def validate_config(config: Config) -> list:
    """
    Validate configuration
    
    Args:
        config: Configuration object
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ['system', 'models', 'storage', 'pipeline', 'logging']
    for section in required_sections:
        if not hasattr(config, section):
            errors.append(f"Missing required section: {section}")
    
    # Check risk levels
    if hasattr(config, 'risk_levels'):
        required_levels = ['critical', 'high', 'medium', 'low']
        for level in required_levels:
            if not hasattr(config.risk_levels, level):
                errors.append(f"Missing risk level: {level}")
    else:
        errors.append("Missing risk_levels section")
    
    return errors

# Testing 

def test_config():
    """Test configuration loading and access"""
    print("="*60)
    print("Testing Configuration System")
    print("="*60 + "\n")
    
    # Test 1: Load configuration
    print("Test 1: Loading configuration...")
    try:
        config = load_config()
        print("Configuration loaded\n")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return False
    
    # Test 2: Dot notation access
    print("Test 2: Dot notation access...")
    try:
        assert config.system.device in ['cuda', 'cpu', 'mps']
        assert config.models.detection.face_detector == 'mtcnn'
        assert config.pipeline.default_mode == 'hybrid'
        print("Dot notation working\n")
    except Exception as e:
        print(f"Dot notation failed: {e}")
        return False
    
    # Test 3: get() method
    print("Test 3: get() method with defaults...")
    try:
        threshold = config.get('models.detection.confidence_threshold', default=0.5)
        assert threshold == 0.7
        
        missing = config.get('nonexistent.key', default='default_value')
        assert missing == 'default_value'
        print("get() method working\n")
    except Exception as e:
        print(f"get() method failed: {e}")
        return False
    
    # Test 4: Device detection
    print("Test 4: Device detection...")
    try:
        device = get_device(config)
        assert isinstance(device, torch.device)
        print(f"Device: {device}\n")
    except Exception as e:
        print(f"Device detection failed: {e}")
        return False
    
    # Test 5: Helper functions
    print("Test 5: Helper functions...")
    try:
        color = get_risk_color(config, 'critical')
        assert color == '#FF0000'
        print(f"Risk color (critical): {color}\n")
    except Exception as e:
        print(f"Helper functions failed: {e}")
        return False
    
    # Test 6: Configuration validation
    print("Test 6: Configuration validation...")
    errors = validate_config(config)
    if errors:
        print(f"Validation errors: {errors}")
        return False
    else:
        print("Configuration valid\n")
    
    # Test 7: Access nested values
    print("Test 7: Nested value access...")
    try:
        print(f"System device: {config.system.device}")
        print(f"Face detector: {config.models.detection.face_detector}")
        print(f"Threshold: {config.models.detection.confidence_threshold}")
        print(f"Max image size: {config.pipeline.max_image_size}")
        print(f"Learning enabled: {config.learning.enabled}")
        print(f"Encryption: {config.storage.encryption_enabled}")
        print("All nested values accessible\n")
    except Exception as e:
        print(f"Nested access failed: {e}")
        return False
    
    # Test 8: to_dict() method
    print("Test 8: to_dict() conversion...")
    try:
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'system' in config_dict
        print("to_dict() working\n")
    except Exception as e:
        print(f"to_dict() failed: {e}")
        return False
    
    print("="*60)
    print("All tests passed! Configuration system working correctly.")
    print("="*60)
    return True


if __name__ == "__main__":
    print("\nPrivacy Guard - Configuration Module Test\n")
    success = test_config()
    if success:
        print("\nSuccess")
        sys.exit(0)
    else:
        print("\nTests failed!")
        sys.exit(1)
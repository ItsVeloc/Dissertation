import os
import yaml
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the teacher-tester system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from a YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default config.
        """
        if config_path is None:
            # Use default config path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(dir_path, "default_config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated path to config value
            default: Default value if key not found
            
        Returns:
            The configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key: Dot-separated path to config value
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save config. If None, overwrites the loaded config.
        """
        if config_path is None:
            # Use default config path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(dir_path, "default_config.yaml")
            
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Singleton config instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
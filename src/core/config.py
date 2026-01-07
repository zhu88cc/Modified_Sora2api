"""Configuration management"""
import tomli
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Application configuration"""

    def __init__(self):
        self._config = self._load_config()
        self._admin_username: Optional[str] = None
        self._admin_password: Optional[str] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from setting.toml"""
        config_path = Path(__file__).parent.parent.parent / "config" / "setting.toml"
        with open(config_path, "rb") as f:
            return tomli.load(f)

    def reload_config(self):
        """Reload configuration from file"""
        self._config = self._load_config()

    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config
    
    @property
    def admin_username(self) -> str:
        # If admin_username is set from database, use it; otherwise fall back to config file
        if self._admin_username is not None:
            return self._admin_username
        return self._config["global"]["admin_username"]

    @admin_username.setter
    def admin_username(self, value: str):
        self._admin_username = value
        self._config["global"]["admin_username"] = value

    def set_admin_username_from_db(self, username: str):
        """Set admin username from database"""
        self._admin_username = username

    @property
    def sora_base_url(self) -> str:
        return self._config["sora"]["base_url"]
    
    @property
    def sora_timeout(self) -> int:
        return self._config["sora"]["timeout"]
    
    @property
    def sora_max_retries(self) -> int:
        return self._config["sora"]["max_retries"]
    
    @property
    def poll_interval(self) -> float:
        return self._config["sora"]["poll_interval"]
    
    @property
    def max_poll_attempts(self) -> int:
        return self._config["sora"]["max_poll_attempts"]
    
    @property
    def server_host(self) -> str:
        return self._config["server"]["host"]
    
    @property
    def server_port(self) -> int:
        return self._config["server"]["port"]

    @property
    def debug_enabled(self) -> bool:
        return self._config.get("debug", {}).get("enabled", False)

    @property
    def debug_log_requests(self) -> bool:
        return self._config.get("debug", {}).get("log_requests", True)

    @property
    def debug_log_responses(self) -> bool:
        return self._config.get("debug", {}).get("log_responses", True)

    @property
    def debug_mask_token(self) -> bool:
        return self._config.get("debug", {}).get("mask_token", True)

    # Mutable properties for runtime updates
    @property
    def api_key(self) -> str:
        return self._config["global"]["api_key"]

    @api_key.setter
    def api_key(self, value: str):
        self._config["global"]["api_key"] = value

    @property
    def admin_password(self) -> str:
        # If admin_password is set from database, use it; otherwise fall back to config file
        if self._admin_password is not None:
            return self._admin_password
        return self._config["global"]["admin_password"]

    @admin_password.setter
    def admin_password(self, value: str):
        self._admin_password = value
        self._config["global"]["admin_password"] = value

    def set_admin_password_from_db(self, password: str):
        """Set admin password from database"""
        self._admin_password = password

    def set_debug_enabled(self, enabled: bool):
        """Set debug mode enabled/disabled"""
        if "debug" not in self._config:
            self._config["debug"] = {}
        self._config["debug"]["enabled"] = enabled

    @property
    def cache_timeout(self) -> int:
        """Get cache timeout in seconds"""
        return self._config.get("cache", {}).get("timeout", 7200)

    def set_cache_timeout(self, timeout: int):
        """Set cache timeout in seconds"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["timeout"] = timeout

    @property
    def cache_base_url(self) -> str:
        """Get cache base URL"""
        return self._config.get("cache", {}).get("base_url", "")

    def set_cache_base_url(self, base_url: str):
        """Set cache base URL"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["base_url"] = base_url

    @property
    def cache_enabled(self) -> bool:
        """Get cache enabled status"""
        return self._config.get("cache", {}).get("enabled", False)

    def set_cache_enabled(self, enabled: bool):
        """Set cache enabled status"""
        if "cache" not in self._config:
            self._config["cache"] = {}
        self._config["cache"]["enabled"] = enabled

    @property
    def image_timeout(self) -> int:
        """Get image generation timeout in seconds"""
        return self._config.get("generation", {}).get("image_timeout", 300)

    def set_image_timeout(self, timeout: int):
        """Set image generation timeout in seconds"""
        if "generation" not in self._config:
            self._config["generation"] = {}
        self._config["generation"]["image_timeout"] = timeout

    @property
    def video_timeout(self) -> int:
        """Get video generation timeout in seconds"""
        return self._config.get("generation", {}).get("video_timeout", 3000)

    def set_video_timeout(self, timeout: int):
        """Set video generation timeout in seconds"""
        if "generation" not in self._config:
            self._config["generation"] = {}
        self._config["generation"]["video_timeout"] = timeout

    @property
    def watermark_free_enabled(self) -> bool:
        """Get watermark-free mode enabled status"""
        return self._config.get("watermark_free", {}).get("watermark_free_enabled", False)

    def set_watermark_free_enabled(self, enabled: bool):
        """Set watermark-free mode enabled/disabled"""
        if "watermark_free" not in self._config:
            self._config["watermark_free"] = {}
        self._config["watermark_free"]["watermark_free_enabled"] = enabled

    @property
    def watermark_free_parse_method(self) -> str:
        """Get watermark-free parse method"""
        return self._config.get("watermark_free", {}).get("parse_method", "third_party")

    @property
    def watermark_free_custom_url(self) -> str:
        """Get custom parse server URL"""
        return self._config.get("watermark_free", {}).get("custom_parse_url", "")

    @property
    def watermark_free_custom_token(self) -> str:
        """Get custom parse server access token"""
        return self._config.get("watermark_free", {}).get("custom_parse_token", "")

    @property
    def at_auto_refresh_enabled(self) -> bool:
        """Get AT auto refresh enabled status"""
        return self._config.get("token_refresh", {}).get("at_auto_refresh_enabled", False)

    def set_at_auto_refresh_enabled(self, enabled: bool):
        """Set AT auto refresh enabled/disabled"""
        if "token_refresh" not in self._config:
            self._config["token_refresh"] = {}
        self._config["token_refresh"]["at_auto_refresh_enabled"] = enabled

    # ==================== Cloudflare Configuration ====================
    
    @property
    def cf_enabled(self) -> bool:
        """Get Cloudflare solver enabled status"""
        cf_config = self._config.get("cloudflare", {})
        # Support both 'enabled' and legacy 'solver_enabled'
        return cf_config.get("enabled", cf_config.get("solver_enabled", False))

    def set_cf_enabled(self, enabled: bool):
        """Set Cloudflare solver enabled/disabled"""
        if "cloudflare" not in self._config:
            self._config["cloudflare"] = {}
        self._config["cloudflare"]["enabled"] = enabled

    @property
    def cf_api_key(self) -> str:
        """Get Cloudflare solver API key"""
        return self._config.get("cloudflare", {}).get("api_key", "")

    def set_cf_api_key(self, key: str):
        """Set Cloudflare solver API key"""
        if "cloudflare" not in self._config:
            self._config["cloudflare"] = {}
        self._config["cloudflare"]["api_key"] = key

    @property
    def cf_api_url(self) -> str:
        """Get Cloudflare solver base URL (e.g., http://localhost:8000)"""
        cf_config = self._config.get("cloudflare", {})
        # Support both 'api_url' and legacy 'solver_api_url'
        return cf_config.get("api_url", cf_config.get("solver_api_url", "http://localhost:8000"))

    def set_cf_api_url(self, url: str):
        """Set Cloudflare solver base URL"""
        if "cloudflare" not in self._config:
            self._config["cloudflare"] = {}
        self._config["cloudflare"]["api_url"] = url

    @property
    def cf_global_enabled(self) -> bool:
        """Get Cloudflare global mode enabled (use CF for all requests)"""
        return self._config.get("cloudflare", {}).get("global_enabled", False)

    def set_cf_global_enabled(self, enabled: bool):
        """Set Cloudflare global mode enabled/disabled"""
        if "cloudflare" not in self._config:
            self._config["cloudflare"] = {}
        self._config["cloudflare"]["global_enabled"] = enabled

    @property
    def cf_api_only_enabled(self) -> bool:
        """Get Cloudflare API-only mode enabled (use CF only for API requests)"""
        return self._config.get("cloudflare", {}).get("api_only_enabled", True)

    def set_cf_api_only_enabled(self, enabled: bool):
        """Set Cloudflare API-only mode enabled/disabled"""
        if "cloudflare" not in self._config:
            self._config["cloudflare"] = {}
        self._config["cloudflare"]["api_only_enabled"] = enabled

    # ==================== Database Configuration ====================
    
    @property
    def db_type(self) -> str:
        """Get database type: sqlite or mysql"""
        return self._config.get("database", {}).get("type", "sqlite")

    @property
    def sqlite_path(self) -> str:
        """Get SQLite database path"""
        return self._config.get("database", {}).get("sqlite_path", "data/hancat.db")

    @property
    def mysql_host(self) -> str:
        """Get MySQL host"""
        return self._config.get("database", {}).get("mysql_host", "localhost")

    @property
    def mysql_port(self) -> int:
        """Get MySQL port"""
        return self._config.get("database", {}).get("mysql_port", 3306)

    @property
    def mysql_user(self) -> str:
        """Get MySQL user"""
        return self._config.get("database", {}).get("mysql_user", "root")

    @property
    def mysql_password(self) -> str:
        """Get MySQL password"""
        return self._config.get("database", {}).get("mysql_password", "")

    @property
    def mysql_database(self) -> str:
        """Get MySQL database name"""
        return self._config.get("database", {}).get("mysql_database", "sora2api")

    @property
    def mysql_pool_size(self) -> int:
        """Get MySQL connection pool size"""
        return self._config.get("database", {}).get("mysql_pool_size", 10)

    # ==================== Redis Configuration ====================
    
    @property
    def redis_enabled(self) -> bool:
        """Get Redis enabled status"""
        return self._config.get("redis", {}).get("enabled", False)

    @property
    def redis_host(self) -> str:
        """Get Redis host"""
        return self._config.get("redis", {}).get("host", "localhost")

    @property
    def redis_port(self) -> int:
        """Get Redis port"""
        return self._config.get("redis", {}).get("port", 6379)

    @property
    def redis_password(self) -> str:
        """Get Redis password"""
        return self._config.get("redis", {}).get("password", "")

    @property
    def redis_db(self) -> int:
        """Get Redis database number"""
        return self._config.get("redis", {}).get("db", 0)

    @property
    def redis_lock_timeout(self) -> int:
        """Get Redis lock timeout in seconds"""
        return self._config.get("redis", {}).get("lock_timeout", 300)

    # Legacy aliases for backward compatibility
    @property
    def cloudflare_solver_enabled(self) -> bool:
        """Legacy alias for cf_enabled"""
        return self.cf_enabled

    @property
    def cloudflare_solver_api_url(self) -> str:
        """Legacy alias for cf_api_url"""
        return self.cf_api_url

# Global config instance
config = Config()

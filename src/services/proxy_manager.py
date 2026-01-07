"""Proxy management module"""
from typing import Optional, List, Dict
from pathlib import Path
import asyncio
from datetime import datetime
from ..core.database import Database
from ..core.models import ProxyConfig

class ProxyManager:
    """Proxy configuration manager with pool rotation support"""
    def __init__(self, db: Database):
        self.db = db
        self._proxy_pool: List[str] = []
        self._pool_index: int = 0
        self._pool_lock = asyncio.Lock()
        self._proxy_file_path = Path(__file__).parent.parent.parent / "data" / "proxy.txt"
        self._proxy_status: Dict[str, dict] = {}  # 代理状态缓存

    def _split_concatenated_proxies(self, text: str) -> List[str]:
        """Split concatenated proxies like 'socks5://...socks5://...' into separate lines

        Handles cases where multiple proxies are pasted together without newlines.
        Also handles 'st5 ' prefix format.
        """
        import re
        # Split by protocol prefixes or 'st5 ' prefix, keeping the delimiter
        # This handles: socks5://...socks5://... or http://...socks5://... or st5 ...st5 ...
        # Note: Use re.IGNORECASE flag instead of inline (?i) to avoid "global flags not at the start" error
        parts = re.split(r'(?=https?://)|(?=socks5h?://)|(?=[sS][tT]5\s+)', text)
        result = []
        for part in parts:
            part = part.strip()
            if part:
                result.append(part)
        return result

    def _load_proxy_pool(self) -> List[str]:
        """Load proxy list from data/proxy.txt"""
        proxies = []
        if self._proxy_file_path.exists():
            try:
                with open(self._proxy_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Split concatenated proxies (e.g., socks5://...socks5://...)
                            split_proxies = self._split_concatenated_proxies(line)
                            for proxy_line in split_proxies:
                                # Convert to standard proxy URL format
                                proxy_url = self._parse_proxy_line(proxy_line)
                                if proxy_url:
                                    proxies.append(proxy_url)
            except Exception as e:
                print(f"⚠️ Failed to load proxy pool: {e}")
        return proxies

    def _parse_proxy_line(self, line: str) -> Optional[str]:
        """Parse proxy line and convert to standard URL format

        Supported formats:
        - http://user:pass@host:port (标准HTTP格式)
        - https://user:pass@host:port (标准HTTPS格式)
        - socks5://user:pass@host:port (标准SOCKS5格式)
        - socks5h://user:pass@host:port (标准SOCKS5H格式)
        - socks5://host:port:user:pass (SOCKS5简化格式)
        - st5 host:port:user:pass (ST5简写格式)
        - host:port (无认证HTTP)
        - host:port:user:pass (HTTP简化格式)
        """
        import re
        line = line.strip()
        if not line:
            return None

        # 1. Handle st5 prefix format: "st5 ip:port:user:pass" -> "socks5://user:pass@ip:port"
        st5_match = re.match(r'^st5\s+(.+)$', line, re.IGNORECASE)
        if st5_match:
            rest = st5_match.group(1)
            # st5 格式可能带@或不带@
            if "@" in rest:
                # st5 user:pass@host:port -> socks5://user:pass@host:port
                return f"socks5://{rest}"
            else:
                # st5 host:port:user:pass -> socks5://user:pass@host:port
                parts = rest.split(":")
                if len(parts) >= 4:
                    host = parts[0]
                    port = parts[1]
                    user = parts[2]
                    password = ":".join(parts[3:])
                    if port.isdigit():
                        return f"socks5://{user}:{password}@{host}:{port}"
            print(f"⚠️ Invalid st5 proxy format: {line}")
            return None

        # 2. Check if it's a URL format with protocol
        if line.startswith(("http://", "https://", "socks5://", "socks5h://")):
            # Already in standard format with @ (user:pass@host:port)
            if "@" in line:
                return line

            # Handle simplified format: protocol://host:port:user:pass
            # e.g., socks5://38.134.216.107:13611:helX01iJa8:jbMXPCMoja
            try:
                protocol_end = line.index("://") + 3
                protocol = line[:protocol_end]  # e.g., "socks5://"
                rest = line[protocol_end:]  # e.g., "38.134.216.107:13611:helX01iJa8:jbMXPCMoja"

                parts = rest.split(":")
                if len(parts) >= 4:
                    # host:port:user:pass format
                    host = parts[0]
                    port = parts[1]
                    user = parts[2]
                    password = ":".join(parts[3:])  # Password might contain colons
                    if port.isdigit():
                        return f"{protocol}{user}:{password}@{host}:{port}"
                elif len(parts) == 2:
                    # Just host:port, no auth
                    return line
            except Exception as e:
                print(f"⚠️ Failed to parse proxy URL: {line}, error: {e}")

            return line

        # 3. No protocol prefix - determine format by structure

        # Check if it has @ (user:pass@host:port without protocol)
        if "@" in line:
            # Assume http:// for bare user:pass@host:port
            return f"http://{line}"

        # Count colons to determine format
        colon_count = line.count(":")

        # Format: host:port (exactly 1 colon, no auth)
        if colon_count == 1:
            host, port = line.split(":")
            if port.isdigit():
                return f"http://{host}:{port}"
            print(f"⚠️ Invalid proxy format (port not numeric): {line}")
            return None

        # Format: host:port:user:pass (3+ colons)
        if colon_count >= 3:
            parts = line.split(":")
            if len(parts) >= 4:
                host = parts[0]
                port = parts[1]
                user = parts[2]
                password = ":".join(parts[3:])  # Password might contain colons
                if port.isdigit():
                    return f"http://{user}:{password}@{host}:{port}"

        # Unknown format
        print(f"⚠️ Unknown proxy format: {line}")
        return None

    async def get_proxy_url(self, token_id: Optional[int] = None) -> Optional[str]:
        """Get proxy URL if enabled, with pool rotation support

        Args:
            token_id: Optional token ID for token-specific proxy (reserved for future use)
        """
        config = await self.db.get_proxy_config()

        if not config.proxy_enabled:
            return None

        # If proxy pool is enabled, rotate through proxies
        if config.proxy_pool_enabled:
            async with self._pool_lock:
                # Reload proxy pool if empty
                if not self._proxy_pool:
                    self._proxy_pool = self._load_proxy_pool()

                if self._proxy_pool:
                    # Get current proxy and rotate index
                    proxy = self._proxy_pool[self._pool_index]
                    self._pool_index = (self._pool_index + 1) % len(self._proxy_pool)
                    return proxy
                else:
                    # Fallback to single proxy if pool is empty
                    return config.proxy_url if config.proxy_url else None

        # Use single proxy
        return config.proxy_url if config.proxy_url else None

    async def update_proxy_config(self, enabled: bool, proxy_url: Optional[str], proxy_pool_enabled: bool = False):
        """Update proxy configuration"""
        await self.db.update_proxy_config(enabled, proxy_url, proxy_pool_enabled)

        # Reset proxy pool when config changes
        async with self._pool_lock:
            self._proxy_pool = []
            self._pool_index = 0
    async def get_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration"""
        return await self.db.get_proxy_config()

    async def get_proxy_pool_count(self) -> int:
        """Get the number of proxies in the pool"""
        proxies = self._load_proxy_pool()
        return len(proxies)

    async def reload_proxy_pool(self):
        """Force reload proxy pool from file"""
        async with self._pool_lock:
            self._proxy_pool = self._load_proxy_pool()
            self._pool_index = 0
            if self._proxy_pool:
                print(f"✅ Proxy pool reloaded: {len(self._proxy_pool)} proxies")
                # Print first proxy as example (masked for security)
                first_proxy = self._proxy_pool[0]
                if "@" in first_proxy:
                    # Mask credentials
                    parts = first_proxy.split("@")
                    print(f"   Example format: {parts[0][:15]}...@{parts[1]}")
                else:
                    print(f"   Example format: {first_proxy}")
        return len(self._proxy_pool)

    async def test_single_proxy(self, proxy_url: str, timeout: int = 10) -> dict:
        """Test a single proxy by connecting to sora.chatgpt.com

        Uses curl_cffi which natively supports SOCKS5/SOCKS5H proxies.

        Args:
            proxy_url: Proxy URL to test (will be parsed to standard format)
            timeout: Request timeout in seconds

        Returns:
            dict with valid, latency, error fields
        """
        from curl_cffi.requests import AsyncSession

        # Parse proxy URL to standard format
        parsed_proxy = self._parse_proxy_line(proxy_url)
        if not parsed_proxy:
            return {
                "valid": False,
                "latency": None,
                "error": f"无效的代理格式: {proxy_url}"
            }

        start_time = datetime.now()
        test_url = "https://sora.chatgpt.com/"

        try:
            async with AsyncSession() as session:
                response = await session.get(
                    test_url,
                    proxy=parsed_proxy,
                    timeout=timeout,
                    allow_redirects=True,
                    impersonate="chrome120"
                )
                latency = (datetime.now() - start_time).total_seconds() * 1000
                # 200 或 403 都表示代理可以连接到目标
                if response.status_code in [200, 403]:
                    return {
                        "valid": True,
                        "latency": round(latency, 2),
                        "error": None
                    }
                else:
                    return {
                        "valid": False,
                        "latency": None,
                        "error": f"HTTP {response.status_code}"
                    }
        except asyncio.TimeoutError:
            return {
                "valid": False,
                "latency": None,
                "error": "连接超时"
            }
        except Exception as e:
            error_msg = str(e)
            # 简化错误信息
            if "proxy" in error_msg.lower():
                return {
                    "valid": False,
                    "latency": None,
                    "error": f"代理连接失败: {error_msg[:100]}"
                }
            return {
                "valid": False,
                "latency": None,
                "error": error_msg[:100]
            }

    async def test_all_proxies(self, remove_invalid: bool = False) -> dict:
        """Test all proxies in the pool

        Args:
            remove_invalid: If True, remove invalid proxies from the pool file

        Returns:
            dict with total, valid, invalid counts and details
        """
        proxies = self._load_proxy_pool()
        if not proxies:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "removed": 0,
                "results": []
            }

        results = []
        valid_proxies = []

        for proxy in proxies:
            result = await self.test_single_proxy(proxy)
            result["proxy"] = self._mask_proxy(proxy)
            result["proxy_full"] = proxy  # 保留完整代理用于后续处理
            results.append(result)

            if result["valid"]:
                valid_proxies.append(proxy)

            # 更新状态缓存
            self._proxy_status[proxy] = {
                "valid": result["valid"],
                "latency": result["latency"],
                "error": result["error"],
                "tested_at": datetime.now().isoformat()
            }

        removed_count = 0
        if remove_invalid and len(valid_proxies) < len(proxies):
            # 重写代理池文件，只保留有效代理
            removed_count = len(proxies) - len(valid_proxies)
            await self._save_proxy_pool(valid_proxies)
            # 重新加载代理池
            await self.reload_proxy_pool()

        # 清理结果中的完整代理信息
        for r in results:
            del r["proxy_full"]

        return {
            "total": len(proxies),
            "valid": len(valid_proxies),
            "invalid": len(proxies) - len(valid_proxies),
            "removed": removed_count,
            "results": results
        }

    def _mask_proxy(self, proxy_url: str) -> str:
        """Mask proxy credentials for display"""
        if "@" in proxy_url:
            parts = proxy_url.split("@")
            protocol_and_creds = parts[0]
            host_part = parts[1]
            # 只显示协议和主机部分
            if "://" in protocol_and_creds:
                protocol = protocol_and_creds.split("://")[0]
                return f"{protocol}://***@{host_part}"
            return f"***@{host_part}"
        return proxy_url

    async def _save_proxy_pool(self, proxies: List[str]):
        """Save proxy list to file"""
        try:
            with open(self._proxy_file_path, "w", encoding="utf-8") as f:
                f.write("# 代理池配置文件\n")
                f.write("# 每行一个代理地址，支持 HTTP 和 SOCKS5 格式\n")
                f.write("# 以 # 开头的行为注释\n\n")
                for proxy in proxies:
                    f.write(f"{proxy}\n")
            print(f"✅ Proxy pool saved: {len(proxies)} proxies")
        except Exception as e:
            print(f"⚠️ Failed to save proxy pool: {e}")

    def get_proxy_status(self) -> Dict[str, dict]:
        """Get cached proxy status"""
        return self._proxy_status

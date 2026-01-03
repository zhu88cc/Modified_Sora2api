"""Cloudflare Solver - Unified Cloudflare challenge handling with global state"""
import asyncio
import threading
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from ..core.config import config


class CloudflareState:
    """Global Cloudflare state manager
    
    Maintains globally shared cf_clearance cookies and user_agent.
    All requests use the same credentials until a new 429 challenge or expiration.
    
    Features:
    - Thread-safe (uses threading.RLock to avoid deadlocks)
    - Credentials valid for 10 minutes, auto-expire
    - Auto-invalidate on 429/403
    - Read operations use snapshots to avoid long lock holding
    """
    
    # Credential TTL (seconds)
    CREDENTIAL_TTL = 600  # 10 minutes
    
    def __init__(self):
        self._cookies: Dict[str, str] = {}
        self._user_agent: Optional[str] = None
        self._last_updated: Optional[datetime] = None
        self._is_valid: bool = False
        # Use RLock to avoid same-thread reentry deadlock
        self._lock = threading.RLock()
    
    @property
    def cookies(self) -> Dict[str, str]:
        """Get current Cloudflare cookies"""
        with self._lock:
            if not self._check_validity():
                return {}
            return self._cookies.copy()
    
    @property
    def user_agent(self) -> Optional[str]:
        """Get current User-Agent"""
        with self._lock:
            if not self._check_validity():
                return None
            return self._user_agent
    
    @property
    def is_valid(self) -> bool:
        """Check if valid Cloudflare credentials exist"""
        with self._lock:
            return self._check_validity()
    
    @property
    def last_updated(self) -> Optional[datetime]:
        """Get last update time"""
        with self._lock:
            return self._last_updated
    
    @property
    def expires_at(self) -> Optional[datetime]:
        """Get credential expiration time"""
        with self._lock:
            if self._last_updated:
                return self._last_updated + timedelta(seconds=self.CREDENTIAL_TTL)
            return None
    
    @property
    def remaining_seconds(self) -> int:
        """Get remaining valid time (seconds)"""
        with self._lock:
            if not self._last_updated or not self._is_valid:
                return 0
            expires = self._last_updated + timedelta(seconds=self.CREDENTIAL_TTL)
            remaining = (expires - datetime.now()).total_seconds()
            return max(0, int(remaining))
    
    def _check_validity(self) -> bool:
        """Check if credentials are valid (internal method, no lock)"""
        if not self._is_valid or not self._cookies or not self._user_agent:
            return False
        if not self._last_updated:
            return False
        # Check expiration
        expires = self._last_updated + timedelta(seconds=self.CREDENTIAL_TTL)
        if datetime.now() > expires:
            self._is_valid = False
            return False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status info"""
        with self._lock:
            is_valid = self._check_validity()
            remaining = 0
            if is_valid and self._last_updated:
                expires = self._last_updated + timedelta(seconds=self.CREDENTIAL_TTL)
                remaining = max(0, int((expires - datetime.now()).total_seconds()))
            return {
                "is_valid": is_valid,
                "has_credentials": bool(self._cookies) and bool(self._user_agent),
                "last_updated": self._last_updated.isoformat() if self._last_updated else None,
                "expires_at": (self._last_updated + timedelta(seconds=self.CREDENTIAL_TTL)).isoformat() if self._last_updated else None,
                "remaining_seconds": remaining,
                "cookies_count": len(self._cookies),
                "user_agent": self._user_agent[:50] + "..." if self._user_agent and len(self._user_agent) > 50 else self._user_agent,
            }
    
    def update(self, cookies: Dict[str, str], user_agent: str):
        """Update Cloudflare credentials (sync method)"""
        with self._lock:
            self._cookies = cookies.copy()
            self._user_agent = user_agent
            self._last_updated = datetime.now()
            self._is_valid = True
            print(f"✅ 全局 Cloudflare 凭据已更新 (cookies: {list(cookies.keys())}, ua: {user_agent[:50]}...)")
    
    async def update_async(self, cookies: Dict[str, str], user_agent: str):
        """Update Cloudflare credentials (async method)"""
        self.update(cookies, user_agent)
    
    def invalidate(self):
        """Mark credentials as invalid (called on 429/403)"""
        with self._lock:
            self._is_valid = False
            print("⚠️ Cloudflare 凭据已标记为无效")
    
    def clear(self):
        """Clear Cloudflare credentials (sync method)"""
        with self._lock:
            self._cookies = {}
            self._user_agent = None
            self._last_updated = None
            self._is_valid = False
            print("🗑️ 全局 Cloudflare 凭据已清除")
    
    async def clear_async(self):
        """Clear Cloudflare credentials (async method)"""
        self.clear()
    
    def apply_to_session(self, session, domain: str = ".sora.chatgpt.com"):
        """Apply cookies to session"""
        with self._lock:
            if not self._check_validity():
                return
            for name, value in self._cookies.items():
                session.cookies.set(name, value, domain=domain)
    
    def get_headers_update(self) -> Dict[str, str]:
        """Get headers to update"""
        with self._lock:
            if self._check_validity() and self._user_agent:
                return {"User-Agent": self._user_agent}
            return {}


# Global per-token state
_cf_states: Dict[str, CloudflareState] = {}
_cf_solving_lock: asyncio.Lock = None  # 全局单一锁
_cf_refreshing: bool = False  # 全局刷新标记

MIN_CF_REFRESH_INTERVAL = 30  # seconds


def _get_global_lock() -> asyncio.Lock:
    """获取全局 CF 解决锁"""
    global _cf_solving_lock
    if _cf_solving_lock is None:
        _cf_solving_lock = asyncio.Lock()
    return _cf_solving_lock


def is_cf_refreshing(token_id: Optional[int] = None, token: Optional[str] = None) -> bool:
    """Check if another request is refreshing CF credentials"""
    return _cf_refreshing


def get_cloudflare_state(token_id: Optional[int] = None, token: Optional[str] = None) -> CloudflareState:
    """Get Cloudflare state manager - 使用全局状态"""
    # 使用全局状态，不按 token 分开
    global _cf_states
    if "global" not in _cf_states:
        _cf_states["global"] = CloudflareState()
    return _cf_states["global"]


async def solve_cloudflare_challenge(
    proxy_url: Optional[str] = None, max_retries: int = 1, force_refresh: bool = False,
    token_id: Optional[int] = None, token: Optional[str] = None, bypass_cooldown: bool = False,
    timeout: int = 30
) -> Optional[Dict[str, Any]]:
    """Solve Cloudflare challenge and update global state
    
    使用全局锁防止并发调用：如果有请求正在获取凭据，其他请求会等待结果。
    """
    global _cf_refreshing
    import concurrent.futures
    import urllib.request
    import json
    import socket
    
    if not config.cf_enabled or not config.cf_api_url:
        print("⚠️ Cloudflare Solver API 未配置或未启用")
        return None
    
    cf_state = get_cloudflare_state()
    lock = _get_global_lock()

    # 检查是否有其他请求正在刷新
    is_waiting = lock.locked()
    if is_waiting:
        print(f"⏳ 等待其他请求获取 CF 凭据...")
    
    # 使用全局锁防止并发 CF Solver 调用
    async with lock:
        # 如果凭据仍然有效且不是强制刷新，直接返回
        if not force_refresh and cf_state.is_valid:
            if not is_waiting:
                print("✅ 使用现有有效的 Cloudflare 凭据")
            return {"cookies": cf_state.cookies, "user_agent": cf_state.user_agent}
        
        # 如果是等待后进入，再次检查凭据是否已被其他请求更新
        if is_waiting and cf_state.is_valid:
            print("✅ 其他请求已获取 CF 凭据")
            return {"cookies": cf_state.cookies, "user_agent": cf_state.user_agent}
        
        # 标记正在刷新
        _cf_refreshing = True
        
        try:
            refresh_msg = "（强制刷新）" if force_refresh else ""
            print(f"🔄 开始获取 Cloudflare 凭据...{refresh_msg}")
            
            # Build full API URL
            base_url = config.cf_api_url.rstrip('/')
            api_url = f"{base_url}/v1/challenge"
            if force_refresh:
                api_url = f"{api_url}?skip_cache=true"
            
            socket_timeout = timeout
            
            def _sync_request():
                """Sync request function, executed in separate thread"""
                try:
                    req = urllib.request.Request(api_url)
                    req.add_header('User-Agent', 'Mozilla/5.0')
                    if config.cf_api_key:
                        req.add_header('Authorization', f'Bearer {config.cf_api_key}')
                    with urllib.request.urlopen(req, timeout=socket_timeout) as response:
                        status_code = response.getcode()
                        data = json.loads(response.read().decode('utf-8'))
                        return {"status_code": status_code, "data": data}
                except urllib.error.URLError as e:
                    print(f"⚠️ CF Solver URL 错误: {e.reason}")
                    return None
                except socket.timeout:
                    print(f"⚠️ CF Solver 超时 ({socket_timeout}秒)")
                    return None
                except Exception as e:
                    print(f"⚠️ CF Solver 异常: {type(e).__name__}: {e}")
                    return None
            
            for attempt in range(1, max_retries + 1):
                try:
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        try:
                            result = await asyncio.wait_for(
                                loop.run_in_executor(executor, _sync_request),
                                timeout=timeout + 5
                            )
                        except asyncio.TimeoutError:
                            print(f"⚠️ CF Solver 请求超时 ({timeout}秒)")
                            return None
                    
                    if result is None:
                        print("⚠️ CF Solver 请求失败")
                        return None
                    
                    if result["status_code"] == 200:
                        data = result["data"]
                        if data.get("success"):
                            cookies = data.get("cookies", {})
                            user_agent = data.get("user_agent")
                            elapsed = data.get("elapsed_seconds", 0)
                            print(f"✅ CF 凭据获取成功，耗时 {elapsed:.2f}s")
                            if cookies and user_agent:
                                cf_state.update(cookies, user_agent)
                            return {"cookies": cookies, "user_agent": user_agent}
                        else:
                            print(f"⚠️ CF Solver 返回失败: {data.get('error')}")
                    else:
                        print(f"⚠️ CF Solver 请求失败: {result['status_code']}")
                
                except Exception as e:
                    print(f"⚠️ CF Solver 调用失败: {type(e).__name__}: {e}")
                
                if attempt < max_retries:
                    await asyncio.sleep(2)
            
            print("❌ CF 凭据获取失败")
            return None
        finally:
            _cf_refreshing = False


def is_cloudflare_challenge(status_code: int, headers: dict, response_text: str) -> bool:
    """Detect if response is a Cloudflare challenge"""
    if status_code not in [429, 403]:
        return False
    return (
        "cf-mitigated" in str(headers)
        or "Just a moment" in response_text
        or "challenge-platform" in response_text
    )

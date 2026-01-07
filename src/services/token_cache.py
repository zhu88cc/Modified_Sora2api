"""Token cache for reducing database queries"""
import asyncio
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from ..core.models import Token


class TokenCache:
    """In-memory cache for tokens to reduce database queries
    
    高并发优化：
    - 缓存活跃 Token 列表，避免每次请求都查询数据库
    - 使用 TTL 自动过期，保证数据一致性
    - 读写分离，读操作无锁
    """
    
    # 缓存 TTL（秒）
    CACHE_TTL = 30
    
    def __init__(self):
        self._active_tokens: List[Token] = []
        self._all_tokens: List[Token] = []
        self._token_by_id: Dict[int, Token] = {}
        self._last_refresh: Optional[datetime] = None
        self._refresh_lock = asyncio.Lock()
        self._dirty = True  # 标记缓存是否需要刷新
    
    @property
    def is_stale(self) -> bool:
        """Check if cache is stale"""
        if self._dirty:
            return True
        if self._last_refresh is None:
            return True
        return datetime.now() - self._last_refresh > timedelta(seconds=self.CACHE_TTL)
    
    def invalidate(self):
        """Mark cache as dirty (needs refresh)"""
        self._dirty = True
    
    async def refresh(self, db) -> None:
        """Refresh cache from database
        
        Args:
            db: Database instance
        """
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if not self.is_stale:
                return
            
            # Fetch all tokens
            all_tokens = await db.get_all_tokens()
            
            # Build active tokens list
            now = datetime.now()
            active_tokens = []
            token_by_id = {}
            
            for token in all_tokens:
                token_by_id[token.id] = token
                
                # Check if token is active
                if not token.is_active:
                    continue
                if token.cooled_until and token.cooled_until > now:
                    continue
                if token.expiry_time and token.expiry_time <= now:
                    continue
                
                active_tokens.append(token)
            
            # Update cache atomically
            self._all_tokens = all_tokens
            self._active_tokens = active_tokens
            self._token_by_id = token_by_id
            self._last_refresh = datetime.now()
            self._dirty = False
    
    def get_active_tokens(self) -> List[Token]:
        """Get cached active tokens (no lock, read-only)"""
        return self._active_tokens.copy()
    
    def get_all_tokens(self) -> List[Token]:
        """Get cached all tokens (no lock, read-only)"""
        return self._all_tokens.copy()
    
    def get_token(self, token_id: int) -> Optional[Token]:
        """Get token by ID from cache"""
        return self._token_by_id.get(token_id)
    
    def update_token(self, token: Token):
        """Update a single token in cache"""
        self._token_by_id[token.id] = token
        
        # Update in lists
        for i, t in enumerate(self._all_tokens):
            if t.id == token.id:
                self._all_tokens[i] = token
                break
        
        # Rebuild active tokens
        now = datetime.now()
        self._active_tokens = [
            t for t in self._all_tokens
            if t.is_active
            and (not t.cooled_until or t.cooled_until <= now)
            and (not t.expiry_time or t.expiry_time > now)
        ]
    
    def remove_token(self, token_id: int):
        """Remove token from cache"""
        if token_id in self._token_by_id:
            del self._token_by_id[token_id]
        
        self._all_tokens = [t for t in self._all_tokens if t.id != token_id]
        self._active_tokens = [t for t in self._active_tokens if t.id != token_id]


# Global cache instance
_token_cache: Optional[TokenCache] = None


def get_token_cache() -> TokenCache:
    """Get the global token cache"""
    global _token_cache
    if _token_cache is None:
        _token_cache = TokenCache()
    return _token_cache

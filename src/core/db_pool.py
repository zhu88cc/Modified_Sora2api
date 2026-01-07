"""Database connection pool for SQLite - High Concurrency Version"""
import asyncio
import aiosqlite
from typing import Optional, List, Tuple, Any
from contextlib import asynccontextmanager
from collections import deque


class WriteQueue:
    """Write operation queue for batching database writes
    
    批量写入队列，将多个写操作合并执行，减少锁竞争
    """
    
    def __init__(self, max_batch_size: int = 100, flush_interval: float = 0.1):
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval
        self._queue: deque = deque()
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the flush task"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """Stop the flush task"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
    
    async def _flush_loop(self):
        """Background task to flush writes periodically"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            # Flush is handled by the pool
    
    async def add(self, sql: str, params: tuple = None) -> asyncio.Future:
        """Add a write operation to the queue"""
        future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((sql, params, future))
        return future
    
    async def get_batch(self) -> List[Tuple[str, tuple, asyncio.Future]]:
        """Get a batch of write operations"""
        async with self._lock:
            batch = []
            while self._queue and len(batch) < self.max_batch_size:
                batch.append(self._queue.popleft())
            return batch


class DatabasePool:
    """SQLite connection pool optimized for high concurrency (1000+)
    
    优化策略：
    1. WAL 模式 - 读写分离，读操作不阻塞
    2. 大读连接池 - 支持大量并发读
    3. 写操作队列 - 批量执行写操作减少锁竞争
    4. 内存缓存 - 减少数据库访问
    5. 优化 PRAGMA 设置
    """
    
    def __init__(self, db_path: str, read_pool_size: int = 20):
        self.db_path = db_path
        self.read_pool_size = read_pool_size
        self._read_pool: asyncio.Queue = asyncio.Queue()
        self._write_conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()
        self._write_queue = WriteQueue(max_batch_size=50, flush_interval=0.05)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the connection pool"""
        async with self._init_lock:
            if self._initialized:
                return
            
            # Create write connection with optimized settings
            self._write_conn = await aiosqlite.connect(
                self.db_path,
                timeout=60.0,  # 60秒超时
                isolation_level=None  # 自动提交模式，手动控制事务
            )
            
            # 高并发优化 PRAGMA 设置
            await self._write_conn.execute("PRAGMA journal_mode=WAL")
            await self._write_conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和安全
            await self._write_conn.execute("PRAGMA cache_size=-64000")  # 64MB 缓存
            await self._write_conn.execute("PRAGMA temp_store=MEMORY")
            await self._write_conn.execute("PRAGMA mmap_size=268435456")  # 256MB 内存映射
            await self._write_conn.execute("PRAGMA busy_timeout=60000")  # 60秒等待锁
            await self._write_conn.execute("PRAGMA wal_autocheckpoint=1000")  # WAL 检查点
            
            # Create read connections pool
            for _ in range(self.read_pool_size):
                conn = await aiosqlite.connect(
                    self.db_path,
                    timeout=60.0
                )
                await conn.execute("PRAGMA query_only=ON")
                await conn.execute("PRAGMA cache_size=-32000")  # 32MB 缓存
                await conn.execute("PRAGMA mmap_size=134217728")  # 128MB 内存映射
                await conn.execute("PRAGMA busy_timeout=60000")
                await self._read_pool.put(conn)
            
            # Start write queue flush task
            self._flush_task = asyncio.create_task(self._flush_write_queue())
            
            self._initialized = True
            print(f"✅ Database pool initialized (read pool: {self.read_pool_size}, WAL mode, high concurrency optimized)")
    
    async def _flush_write_queue(self):
        """Background task to flush write queue"""
        while True:
            try:
                await asyncio.sleep(0.05)  # 50ms 刷新间隔
                batch = await self._write_queue.get_batch()
                if batch:
                    async with self._write_lock:
                        await self._write_conn.execute("BEGIN IMMEDIATE")
                        try:
                            for sql, params, future in batch:
                                try:
                                    if params:
                                        cursor = await self._write_conn.execute(sql, params)
                                    else:
                                        cursor = await self._write_conn.execute(sql)
                                    future.set_result(cursor.lastrowid)
                                except Exception as e:
                                    future.set_exception(e)
                            await self._write_conn.execute("COMMIT")
                        except Exception as e:
                            await self._write_conn.execute("ROLLBACK")
                            # Set exception for all pending futures
                            for _, _, future in batch:
                                if not future.done():
                                    future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Write queue flush error: {e}")
    
    async def close(self):
        """Close all connections"""
        # Stop flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        if self._write_conn:
            await self._write_conn.close()
            self._write_conn = None
        
        while not self._read_pool.empty():
            conn = await self._read_pool.get()
            await conn.close()
        
        self._initialized = False
    
    @asynccontextmanager
    async def read_connection(self):
        """Get a read connection from pool"""
        if not self._initialized:
            await self.initialize()
        
        conn = await self._read_pool.get()
        try:
            conn.row_factory = aiosqlite.Row
            yield conn
        finally:
            await self._read_pool.put(conn)
    
    @asynccontextmanager
    async def write_connection(self):
        """Get the write connection with lock"""
        if not self._initialized:
            await self.initialize()
        
        async with self._write_lock:
            self._write_conn.row_factory = aiosqlite.Row
            yield self._write_conn
    
    async def execute_write(self, sql: str, params: tuple = None, max_retries: int = 5):
        """Execute a write operation with retry logic
        
        Args:
            sql: SQL statement
            params: SQL parameters
            max_retries: Maximum retry attempts for locked database
        """
        for attempt in range(max_retries):
            try:
                async with self.write_connection() as conn:
                    if params:
                        cursor = await conn.execute(sql, params)
                    else:
                        cursor = await conn.execute(sql)
                    await conn.commit()
                    return cursor
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 0.2 + (attempt * 0.1)  # 递增等待
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def queue_write(self, sql: str, params: tuple = None) -> Any:
        """Queue a write operation for batch execution
        
        Use this for non-critical writes that can be batched.
        Returns the lastrowid when the write is executed.
        """
        if not self._initialized:
            await self.initialize()
        
        future = await self._write_queue.add(sql, params)
        return await future
    
    async def execute_read(self, sql: str, params: tuple = None):
        """Execute a read operation"""
        async with self.read_connection() as conn:
            if params:
                cursor = await conn.execute(sql, params)
            else:
                cursor = await conn.execute(sql)
            return cursor


# Global pool instance
_pool: Optional[DatabasePool] = None


def get_pool() -> Optional[DatabasePool]:
    """Get the global database pool"""
    return _pool


async def init_pool(db_path: str, read_pool_size: int = 20):
    """Initialize the global database pool
    
    Args:
        db_path: Path to database file
        read_pool_size: Number of read connections (default 20 for high concurrency)
    """
    global _pool
    if _pool is None:
        _pool = DatabasePool(db_path, read_pool_size)
        await _pool.initialize()
    return _pool


async def close_pool():
    """Close the global database pool"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_db_connection(db_path: str, readonly: bool = False):
    """Get a database connection with proper settings
    
    This is a standalone helper for code that doesn't use the pool yet.
    Includes WAL mode and busy timeout settings.
    """
    conn = await aiosqlite.connect(
        db_path,
        timeout=60.0
    )
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA busy_timeout=60000")
        await conn.execute("PRAGMA cache_size=-64000")
        await conn.execute("PRAGMA mmap_size=268435456")
        if readonly:
            await conn.execute("PRAGMA query_only=ON")
        conn.row_factory = aiosqlite.Row
        yield conn
    finally:
        await conn.close()

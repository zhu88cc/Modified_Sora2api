"""Cloudflare Turnstile Challenge Solver using DrissionPage"""
import time
import threading
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CloudflareSolution:
    """Cloudflare challenge solution result"""
    cf_clearance: str
    cookies: Dict[str, str]
    user_agent: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, max_age_seconds: int = 1800) -> bool:
        """检查 cookie 是否过期（默认30分钟）"""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > max_age_seconds


class CloudflareSolver:
    """
    Cloudflare Turnstile Challenge solver using DrissionPage.
    
    DrissionPage 使用真实浏览器绕过 Cloudflare 检测。
    """
    
    # 缓存的解决方案（按域名）
    _solutions: Dict[str, CloudflareSolution] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        proxy: Optional[str] = None,
        headless: bool = True,
        timeout: int = 60
    ):
        """
        Initialize CloudflareSolver.
        
        Args:
            proxy: 代理地址，格式 "ip:port" 或 "http://ip:port"
            headless: 是否无头模式（默认 True）
            timeout: 等待 Cloudflare 验证超时时间（秒）
        """
        self.proxy = proxy
        self.headless = headless
        self.timeout = timeout
    
    def _create_page(self):
        """创建浏览器页面"""
        from DrissionPage import ChromiumPage, ChromiumOptions
        
        options = ChromiumOptions()
        
        # 设置代理
        if self.proxy:
            proxy_addr = self.proxy
            if not proxy_addr.startswith("http"):
                proxy_addr = f"http://{proxy_addr}"
            options.set_proxy(proxy_addr)
        
        # 无头模式
        if self.headless:
            options.headless()
        
        # 反检测设置
        options.set_argument("--disable-blink-features=AutomationControlled")
        options.set_argument("--no-sandbox")
        options.set_argument("--disable-dev-shm-usage")
        options.set_argument("--disable-gpu")
        
        return ChromiumPage(options)
    
    @classmethod
    def get_cached_solution(cls, domain: str) -> Optional[CloudflareSolution]:
        """获取缓存的解决方案"""
        with cls._lock:
            solution = cls._solutions.get(domain)
            if solution and not solution.is_expired():
                return solution
            return None
    
    @classmethod
    def cache_solution(cls, domain: str, solution: CloudflareSolution):
        """缓存解决方案"""
        with cls._lock:
            cls._solutions[domain] = solution
    
    def solve(self, website_url: str, force_refresh: bool = False) -> CloudflareSolution:
        """
        解决 Cloudflare Turnstile challenge.
        
        Args:
            website_url: 目标页面 URL
            force_refresh: 强制刷新，忽略缓存
        
        Returns:
            CloudflareSolution 包含 cf_clearance cookie
            
        Raises:
            CloudflareError: 如果解决失败
        """
        from urllib.parse import urlparse
        domain = urlparse(website_url).netloc
        
        # 检查缓存
        if not force_refresh:
            cached = self.get_cached_solution(domain)
            if cached:
                print(f"✅ 使用缓存的 Cloudflare cookie (剩余 {1800 - (datetime.now() - cached.created_at).total_seconds():.0f}s)")
                return cached
        
        page = self._create_page()
        
        try:
            print(f"正在访问: {website_url}")
            page.get(website_url)
            
            # 等待 Cloudflare 验证完成
            cf_clearance = self._wait_for_clearance(page)
            
            # 获取所有 cookies
            cookies = {}
            for cookie in page.cookies():
                cookies[cookie["name"]] = cookie["value"]
            
            # 获取 user agent
            user_agent = page.run_js("return navigator.userAgent")
            
            solution = CloudflareSolution(
                cf_clearance=cf_clearance,
                cookies=cookies,
                user_agent=user_agent
            )
            
            # 缓存解决方案
            self.cache_solution(domain, solution)
            
            return solution
            
        finally:
            page.quit()
    
    def _wait_for_clearance(self, page) -> str:
        """等待 cf_clearance cookie 出现"""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise CloudflareError(f"等待 Cloudflare 验证超时 ({self.timeout}s)")
            
            # 检查是否有 cf_clearance cookie
            for cookie in page.cookies():
                if cookie["name"] == "cf_clearance":
                    print(f"✅ Cloudflare 验证通过，耗时 {elapsed:.1f}s")
                    return cookie["value"]
            
            # 检查页面是否还在验证中
            title = page.title.lower() if page.title else ""
            if "just a moment" in title or "checking" in title:
                print(f"等待 Cloudflare 验证中... ({elapsed:.1f}s)")
            else:
                # 页面标题变了，可能已经通过，再检查一次 cookie
                for cookie in page.cookies():
                    if cookie["name"] == "cf_clearance":
                        print(f"✅ Cloudflare 验证通过，耗时 {elapsed:.1f}s")
                        return cookie["value"]
            
            time.sleep(1)


class CloudflareError(Exception):
    """Cloudflare solving error"""
    pass


# 便捷函数
def solve_cloudflare(url: str, proxy: str = None, headless: bool = True) -> Dict[str, str]:
    """
    便捷函数：解决 Cloudflare 并返回 cookies 字典
    
    Args:
        url: 目标 URL
        proxy: 代理地址
        headless: 是否无头模式
    
    Returns:
        cookies 字典，包含 cf_clearance
    """
    solver = CloudflareSolver(proxy=proxy, headless=headless)
    solution = solver.solve(url)
    return solution.cookies

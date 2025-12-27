"""Test Cloudflare Turnstile Challenge Solver using DrissionPage"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.cloudflare_solver import CloudflareSolver, CloudflareError


def test_cloudflare_solver():
    """测试 Cloudflare Turnstile challenge 解决"""
    # 配置
    proxy = "127.0.0.1:7897"
    target_url = "https://sora.chatgpt.com"
    
    print("Testing Cloudflare solver with DrissionPage...")
    print(f"Target URL: {target_url}")
    print(f"Proxy: {proxy}")
    print("-" * 50)
    
    solver = CloudflareSolver(
        proxy=proxy,
        headless=False,  # 设为 True 可无头运行
        timeout=60
    )
    
    try:
        solution = solver.solve(target_url)
        
        print("\n✅ Challenge solved successfully!")
        print(f"cf_clearance: {solution.cf_clearance[:50]}..." if len(solution.cf_clearance) > 50 else f"cf_clearance: {solution.cf_clearance}")
        print(f"userAgent: {solution.user_agent}")
        print(f"cookies: {list(solution.cookies.keys())}")
        
        return solution
        
    except CloudflareError as e:
        print(f"\n❌ Failed to solve challenge: {e}")
        return None


def test_sora_cloudflare():
    """测试 sora.chatgpt.com 的 Cloudflare 解决"""
    proxy = "127.0.0.1:7897"
    
    print("Testing Cloudflare solver for sora.chatgpt.com...")
    print(f"Proxy: {proxy}")
    print("-" * 50)
    
    solver = CloudflareSolver(
        proxy=proxy,
        headless=False,
        timeout=60
    )
    
    try:
        solution = solver.solve("https://sora.chatgpt.com")
        
        print("\n✅ Sora Cloudflare challenge solved!")
        print(f"cf_clearance: {solution.cf_clearance}")
        print(f"\n可以在 API 请求中使用这些 cookies:")
        for name, value in solution.cookies.items():
            print(f"  {name}: {value[:30]}..." if len(value) > 30 else f"  {name}: {value}")
        
        return solution
        
    except CloudflareError as e:
        print(f"\n❌ Failed: {e}")
        return None


if __name__ == "__main__":
    test_sora_cloudflare()

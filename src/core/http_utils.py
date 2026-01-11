"""HTTP utilities - Common HTTP headers and request helpers"""
import base64
import json
import os
import random
from typing import Optional, Dict

# 手机指纹列表（curl_cffi 只支持这些手机指纹）
MOBILE_FINGERPRINTS = [
    "safari17_2_ios",
    "safari18_0_ios",
]

# Sora App UA (Android)
SORA_APP_USER_AGENT = "Sora/1.2026.007 (Android 15; 24122RKC7C; build 2600700)"

# 手机 UA 列表 (Sora App)
MOBILE_USER_AGENTS = [
    SORA_APP_USER_AGENT,
]

# Sora App 请求头模板
CHROME_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://sora.chatgpt.com",
    "Referer": "https://sora.chatgpt.com/",
}

# 默认 UA
DEFAULT_USER_AGENT = SORA_APP_USER_AGENT


def get_random_fingerprint() -> str:
    """获取随机手机指纹"""
    return random.choice(MOBILE_FINGERPRINTS)


def get_random_user_agent() -> str:
    """获取随机手机 UA"""
    return random.choice(MOBILE_USER_AGENTS)


def generate_device_id() -> str:
    """生成随机的 oai-device-id (UUID v4 格式)"""
    import uuid
    return str(uuid.uuid4())


def generate_id() -> str:
    """生成随机 UUID，用于 openai-sentinel-token 请求/响应"""
    import uuid
    return str(uuid.uuid4())


def b64_like(n_bytes: int, suffix: str = "", urlsafe: bool = False) -> str:
    """生成类似 base64 的随机字符串（用于 pow token mock）"""
    raw = os.urandom(n_bytes)
    if urlsafe:
        s = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    else:
        s = base64.b64encode(raw).decode("ascii")
    return s + suffix


# ============================================================================
# PoW (Proof of Work) 相关常量和函数
# ============================================================================

# PoW 最大迭代次数
POW_MAX_ITERATION = 500000

# 浏览器环境模拟常量
POW_CORES = [8, 16, 24, 32]
POW_SCRIPTS = [
    "https://cdn.oaistatic.com/_next/static/cXh69klOLzS0Gy2joLDRS/_ssgManifest.js?dpl=453ebaec0d44c2decab71692e1bfe39be35a24b3"
]
POW_DPL = ["prod-f501fe933b3edf57aea882da888e1a544df99840"]
POW_NAVIGATOR_KEYS = [
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "webdriver−false",
    "vendor−Google Inc.",
    "mediaDevices−[object MediaDevices]",
    "cookieEnabled−true",
    "product−Gecko",
    "productSub−20030107",
    "hardwareConcurrency−32",
    "onLine−true",
]
POW_DOCUMENT_KEYS = ["_reactListeningo743lnnpvdg", "location"]
POW_WINDOW_KEYS = [
    "0", "window", "self", "document", "name", "location",
    "navigator", "screen", "innerWidth", "innerHeight",
    "localStorage", "sessionStorage", "crypto", "performance",
    "fetch", "setTimeout", "setInterval", "console",
]


def get_pow_parse_time() -> str:
    """生成 PoW 用的时间字符串 (EST 时区)"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=-5)))
    return now.strftime("%a %b %d %Y %H:%M:%S") + " GMT-0500 (Eastern Standard Time)"


def get_pow_config(user_agent: str) -> list:
    """生成 PoW 配置数组
    
    注意: config[3] 和 config[9] 会在 PoW 计算中动态修改
    """
    import time
    import uuid
    
    return [
        random.choice([1920 + 1080, 2560 + 1440, 1920 + 1200, 2560 + 1600]),  # [0] screen size
        get_pow_parse_time(),  # [1] 时间字符串
        4294705152,  # [2] jsHeapSizeLimit
        0,  # [3] 迭代次数 (动态)
        user_agent,  # [4] UA
        random.choice(POW_SCRIPTS) if POW_SCRIPTS else "",  # [5] script
        random.choice(POW_DPL) if POW_DPL else None,  # [6] dpl
        "en-US",  # [7] language
        "en-US,es-US,en,es",  # [8] languages
        0,  # [9] 迭代次数/2 (动态)
        random.choice(POW_NAVIGATOR_KEYS),  # [10] navigator key
        random.choice(POW_DOCUMENT_KEYS),  # [11] document key
        random.choice(POW_WINDOW_KEYS),  # [12] window key
        time.perf_counter() * 1000,  # [13] perf time
        str(uuid.uuid4()),  # [14] UUID
        "",  # [15] empty
        random.choice(POW_CORES),  # [16] cores
        time.time() * 1000 - (time.perf_counter() * 1000),  # [17] time origin
    ]


def solve_pow(seed: str, difficulty: str, config: list) -> tuple:
    """执行真正的 PoW 计算
    
    Args:
        seed: 来自 sentinel/req 响应的 seed
        difficulty: 来自 sentinel/req 响应的 difficulty (十六进制字符串)
        config: 环境配置数组
        
    Returns:
        (answer, success): answer 是 base64 编码的结果，success 表示是否找到有效答案
    """
    import hashlib
    
    diff_len = len(difficulty) // 2  # 十六进制转字节长度
    seed_encoded = seed.encode()
    target_diff = bytes.fromhex(difficulty)
    
    # 预计算静态部分
    static_part1 = (json.dumps(config[:3], separators=(',', ':'), ensure_ascii=False)[:-1] + ',').encode()
    static_part2 = (',' + json.dumps(config[4:9], separators=(',', ':'), ensure_ascii=False)[1:-1] + ',').encode()
    static_part3 = (',' + json.dumps(config[10:], separators=(',', ':'), ensure_ascii=False)[1:]).encode()
    
    for i in range(POW_MAX_ITERATION):
        # config[3] = i, config[9] = i >> 1
        dynamic_i = str(i).encode()
        dynamic_j = str(i >> 1).encode()
        
        final_json = static_part1 + dynamic_i + static_part2 + dynamic_j + static_part3
        b64_encoded = base64.b64encode(final_json)
        
        # SHA3-512 哈希
        hash_value = hashlib.sha3_512(seed_encoded + b64_encoded).digest()
        
        # 检查是否满足难度要求
        if hash_value[:diff_len] <= target_diff:
            return b64_encoded.decode(), True
    
    # 失败时返回错误标记
    error_token = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + base64.b64encode(f'"{seed}"'.encode()).decode()
    return error_token, False


def get_pow_token(user_agent: Optional[str] = None) -> str:
    """生成初始 PoW token (用于首次请求 sentinel/req)
    
    这个 token 用于获取 seed 和 difficulty，之后需要用 solve_pow 计算真正的答案
    """
    ua = user_agent or get_random_user_agent()
    config = get_pow_config(ua)
    
    # 生成一个随机 seed 用于 requirements token
    seed = format(random.random())
    difficulty = "0fffff"  # 默认难度
    
    solution, _ = solve_pow(seed, difficulty, config)
    return "gAAAAAC" + solution


def get_pow_token_mock(user_agent: Optional[str] = None) -> str:
    """生成 PoW token (兼容旧接口)
    
    实际调用 get_pow_token
    """
    return get_pow_token(user_agent)


def post_sentinel_req(
    base_url: str,
    flow: str,
    pow_token: str,
    auth_token: Optional[str] = None,
) -> Dict:
    """请求 /backend-api/sentinel/req 获取 token 响应"""
    import requests

    url = f"{base_url.rstrip('/')}/backend-api/sentinel/req"
    payload = {"p": pow_token, "flow": flow, "id": generate_id()}
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    r = requests.post(url, json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def build_openai_sentinel_token(flow: str, resp: Dict, pow_token: str, user_agent: Optional[str] = None) -> str:
    """构建 openai-sentinel-token 字符串
    
    如果响应中包含 proofofwork 要求，会执行真正的 PoW 计算
    """
    final_pow_token = pow_token
    
    # 检查是否需要执行 PoW
    proofofwork = resp.get("proofofwork", {})
    if proofofwork.get("required"):
        seed = proofofwork.get("seed", "")
        difficulty = proofofwork.get("difficulty", "")
        if seed and difficulty:
            # 执行真正的 PoW 计算
            ua = user_agent or get_random_user_agent()
            config = get_pow_config(ua)
            solution, success = solve_pow(seed, difficulty, config)
            if success:
                final_pow_token = "gAAAAAB" + solution
            else:
                # PoW 失败，使用错误标记
                final_pow_token = "gAAAAAB" + solution
    
    token_payload = {
        "p": final_pow_token,
        "t": resp.get("turnstile", {}).get("dx", ""),
        "c": resp.get("token", ""),
        "id": generate_id(),
        "flow": flow
    }
    return json.dumps(token_payload, ensure_ascii=False, separators=(",", ":"))


def build_sora_headers(
    token: str,
    user_agent: Optional[str] = None,
    content_type: Optional[str] = None,
    sentinel_token: Optional[str] = None,
    device_id: Optional[str] = None
) -> dict:
    """构建 Sora API 请求头
    
    Args:
        token: Access token
        user_agent: 自定义 User-Agent (默认随机选择)
        content_type: Content-Type (默认 application/json)
        sentinel_token: openai-sentinel-token (仅生成请求需要)
        device_id: oai-device-id (默认生成随机 UUID)
    
    Returns:
        完整的请求头字典
    """
    headers = {
        **CHROME_HEADERS,
        "Authorization": f"Bearer {token}",
        "User-Agent": user_agent or get_random_user_agent(),
        "oai-device-id": device_id or generate_device_id(),
    }
    
    if content_type:
        headers["Content-Type"] = content_type
    
    if sentinel_token:
        headers["openai-sentinel-token"] = sentinel_token
    
    return headers


def build_simple_headers(token: str) -> dict:
    """构建简单的 API 请求头
    
    Args:
        token: Access token
    
    Returns:
        简单请求头字典
    """
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Origin": "https://sora.chatgpt.com",
        "Referer": "https://sora.chatgpt.com/",
        "User-Agent": get_random_user_agent(),
    }

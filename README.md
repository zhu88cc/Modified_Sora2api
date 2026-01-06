# 本项目二次开发基于 [sora2api](https://github.com/TheSmallHanCat/sora2api)

##### 原作者: [TheSmallHanCat](https://github.com/TheSmallHanCat)

# 该项目是二开的项目请谨慎使用

---

## 项目介绍

Sora2API 是一个 OpenAI 兼容的 Sora API 服务，支持文生图、文生视频、图生视频、视频 Remix、角色创建等功能。

**✅ 已兼容 [new-api](https://github.com/Calcium-Ion/new-api) sora2 渠道对接格式**

### 主要功能

- 文生图片 / 图生图片
- 文生视频 / 图生视频
- 视频 Remix（基于已有视频二次创作）
- 视频分镜（Storyboard）
- 角色卡创建与引用
- Token 池管理与自动轮询
- 代理配置（支持单代理和代理池轮询）
- 代理池检测（自动检测并移除无效代理）
- 无水印模式
- 管理后台

### 批量操作功能

- 批量添加 Token（支持重复检测）
- 批量测试 Token（自动启用/禁用）
- 批量激活 Sora2（使用邀请码）
- 批量启用/禁用 Token
- 批量删除禁用 Token

### 性能优化

- 自适应轮询机制（根据进度动态调整间隔）
- 停滞检测（避免无效请求）
- 并发控制（批量操作限流）
- Token 缓存（减少数据库查询）

---

## new-api 对接说明

本项目的 `/v1/videos` 接口已完全兼容 new-api 的 sora2 渠道格式。

### 配置方式

在 new-api 中添加渠道：
- **类型**: Sora
- **Base URL**: `http://your-sora2api-server:8000`
- **密钥**: 你的 API Key（默认 `han1234`）
- **模型**: `sora-2`, `sora-2-pro`

### 支持的接口

| 接口 | 描述 |
|------|------|
| `POST /v1/videos` | 创建视频生成任务 |
| `GET /v1/videos/{id}` | 获取任务状态 |
| `GET /v1/videos/{id}/content` | 获取视频直链（302 重定向） |
| `POST /v1/videos/{id}/remix` | 视频 Remix |

### 响应格式

```json
{
  "id": "sora-2-abc123def456",
  "object": "video",
  "model": "sora-2",
  "status": "in_progress",
  "progress": 50,
  "created_at": 1702388400,
  "completed_at": 1702388500,
  "seconds": "10",
  "size": "1280x720",
  "error": null
}
```

### 状态值

| 状态 | 描述 |
|------|------|
| `queued` | 排队中 |
| `in_progress` | 处理中 |
| `completed` | 成功 |
| `failed` | 失败 |
| `cancelled` | Client disconnected |

Note: `cancelled` indicates the client disconnected before completion. `request_logs.status_code` is set to 499.

---

## 快速开始

```bash
# Docker 部署
docker-compose up -d

# 本地部署
pip install -r requirements.txt
python main.py
```

**管理后台**: http://localhost:8000 (默认账号: admin/admin)

**默认 API Key**: `han1234`

---

## 项目结构

```
├── config/                 # 配置文件
│   ├── setting.toml       # 主配置文件
│   └── setting_warp.toml  # Warp 配置
├── data/                   # 数据目录
│   ├── hancat.db          # SQLite 数据库
│   └── proxy.txt          # 代理池配置（每行一个代理地址）
├── docs/                   # API 文档
│   └── API_V1_DOCUMENTATION.md # v1 API 文档
├── src/                    # 源代码
│   ├── api/               # API 路由
│   │   ├── admin.py       # 管理接口
│   │   ├── openai_compat.py # OpenAI 兼容接口
│   │   ├── public.py      # 公共接口
│   │   ├── routes.py      # 路由注册
│   │   └── sora_compat.py # Sora 兼容接口
│   ├── core/              # 核心模块
│   │   ├── auth.py        # 认证
│   │   ├── config.py      # 配置管理
│   │   ├── database.py    # 数据库
│   │   ├── logger.py      # 日志
│   │   └── models.py      # 数据模型
│   └── services/          # 业务服务
│       ├── generation_handler.py # 生成处理
│       ├── proxy_manager.py      # 代理管理
│       ├── sora_client.py        # Sora 客户端
│       └── token_manager.py      # Token 管理
├── static/                 # 静态文件
│   ├── login.html         # 登录页面
│   └── manage.html        # 管理后台
├── tests/                  # 测试脚本
├── docker-compose.yml      # Docker 配置
├── Dockerfile             # Docker 镜像
├── main.py                # 入口文件
└── requirements.txt       # 依赖
```

---

## 代理池配置

在 `data/proxy.txt` 中配置代理列表，每行一个：

```
# 支持格式
http://ip:port
http://user:pass@ip:port
socks5://ip:port
ip:port
ip:port:user:pass
```

**使用逻辑：**
1. 在管理后台启用 `代理` 和 `代理池` 开关
2. 每次请求 Sora API 时，自动轮询使用下一个代理
3. 代理池为空时，回退到单代理配置
4. 修改 `proxy.txt` 后，在管理后台点击"重载代理池"生效

---

## API 文档

详细 API 文档请参考：
- [v1 接口文档](docs/API_V1_DOCUMENTATION.md) - 完整的 v1 API 接口文档（new-api 兼容）

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/chat/completions` | POST | 统一聊天补全接口（支持流式） |
| `/v1/videos` | POST | 创建视频生成任务（new-api 兼容） |
| `/v1/videos/{id}` | GET | 获取视频任务状态（new-api 兼容） |
| `/v1/videos/{id}/content` | GET | 获取视频直链（302 重定向） |
| `/v1/videos/{id}/remix` | POST | 视频 Remix（new-api 兼容） |
| `/v1/images/generations` | POST | 图片生成 |
| `/v1/characters` | POST | 角色创建 |
| `/v1/stats` | GET | 系统统计 |
| `/v1/feed` | GET | 公共 Feed |
| `/api/tokens` | GET/POST | Token 管理 |
| `/api/login` | POST | 管理员登录 |

---

## 许可证

MIT License

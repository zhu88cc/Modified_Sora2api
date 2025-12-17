# 本项目二次开发基于 [sora2api](https://github.com/TheSmallHanCat/sora2api)
####
# 原作者: [TheSmallHanCat](https://github.com/TheSmallHanCat)
# 该项目是二开的项目请谨慎使用
# Sora2API

OpenAI 兼容的 Sora API 服务

## 快速开始

```bash
# Docker 部署
docker-compose up -d

# 本地部署
pip install -r requirements.txt
python main.py
```

**管理后台**: http://localhost:8000 (默认账号: admin/admin)

---

## API 说明

### 基本信息

- **接口端点**: `POST /v1/chat/completions`
- **身份验证**: `Authorization: Bearer YOUR_API_KEY`
- **默认 API Key**: `han1234`

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称 |
| `messages` | array | 是 | 消息数组 |
| `stream` | boolean | 否 | 是否流式输出，默认 false |
| `image` | string | 否 | Base64 编码的图片（图生视频） |
| `video` | string | 否 | Base64 编码的视频或视频 URL（角色创建） |
| `remix_target_id` | string | 否 | Sora 分享链接 ID（Remix 模式） |
| `style_id` | string | 否 | 视频风格 |
| `character_options` | object | 否 | 角色创建选项 |

### character_options 参数

角色创建时的可选配置：

| 参数 | 类型 | 说明 |
|------|------|------|
| `timestamps` | string | 视频时间戳，用于提取角色帧，格式如 `"0,3"` |
| `username` | string | 自定义角色用户名 |
| `display_name` | string | 自定义角色显示名称 |
| `instruction_set` | string | 角色指令集 |
| `safety_instruction_set` | string | 安全指令集 |

### 支持的模型

**图片模型**

| 模型名称 | 尺寸 |
|------|------|
| `sora-image` | 360x360 |
| `sora-image-landscape` | 540x360 |
| `sora-image-portrait` | 360x540 |

**视频模型**

| 模型名称 | 时长 | 方向 |
|------|------|------|
| `sora-video-10s` | 10秒 | 方形 |
| `sora-video-15s` | 15秒 | 方形 |
| `sora-video-landscape-10s` | 10秒 | 横屏 |
| `sora-video-landscape-15s` | 15秒 | 横屏 |
| `sora-video-portrait-10s` | 10秒 | 竖屏 |
| `sora-video-portrait-15s` | 15秒 | 竖屏 |

### 视频风格 (style_id)

| 风格 | 值 |
|------|------|
| 节日 | `festive` |
| 复古 | `retro` |
| 新闻 | `news` |
| 自拍 | `selfie` |
| 手持 | `handheld` |
| 动漫 | `anime` |
| 漫画 | `comic` |
| 金色 | `golden` |
| 怀旧 | `vintage` |

---

## 流式响应格式说明

### reasoning_content 结构

流式响应中的 `reasoning_content` 字段为结构化 JSON 对象，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage` | string | 当前处理阶段 |
| `status` | string | 当前状态 |
| `progress` | number | 进度百分比 (0-100)，可选 |
| `message` | string | 人类可读的状态消息 |
| `details` | object | 额外详情，可选 |
| `timestamp` | number | Unix 时间戳 |

### stage 阶段值

| 值 | 说明 |
|------|------|
| `upload` | 上传媒体文件 |
| `generation` | 生成图片/视频 |
| `cache` | 缓存文件 |
| `character_creation` | 创建角色 |
| `remix` | 混剪视频 |
| `watermark_free` | 去水印处理 |
| `storyboard` | 故事板模式 |
| `error` | 错误状态 |
| `processing` | 通用处理中 |

### status 状态值

| 值 | 说明 |
|------|------|
| `started` | 阶段开始 |
| `processing` | 处理中 |
| `completed` | 阶段完成 |
| `error` | 发生错误 |

### content 结果格式

最终结果通过 `content` 字段返回，为 JSON 字符串格式。

#### 图片生成结果

```json
{
  "type": "image",
  "urls": ["http://localhost:8000/tmp/xxx.png"],
  "count": 1,
  "data": [
    {"url": "http://localhost:8000/tmp/xxx.png"}
  ]
}
```

> `data` 字段兼容 OpenAI Images API 格式

#### 视频生成结果

```json
{
  "type": "video",
  "url": "http://localhost:8000/tmp/xxx.mp4",
  "data": [
    {
      "url": "http://localhost:8000/tmp/xxx.mp4",
      "revised_prompt": null
    }
  ]
}
```

> `data` 字段兼容 OpenAI Sora API 格式

#### 角色创建结果

```json
{
  "type": "character",
  "username": "mycharacter123",
  "display_name": "我的角色",
  "cameo_id": "cameo_xxx",
  "character_id": "char_xxx",
  "data": {
    "username": "mycharacter123",
    "display_name": "我的角色",
    "cameo_id": "cameo_xxx",
    "character_id": "char_xxx"
  }
}
```

#### 错误结果

```json
{
  "type": "error",
  "error": "Content policy violation: ...",
  "data": {
    "error": "Content policy violation: ..."
  }
}
```

### 解析示例 (Python)

```python
import json

def parse_stream_response(line: str):
    """解析流式响应行"""
    if not line.startswith("data: "):
        return None
    
    data_str = line[6:]  # 移除 "data: " 前缀
    if data_str == "[DONE]":
        return {"done": True}
    
    data = json.loads(data_str)
    delta = data["choices"][0]["delta"]
    
    # 解析 reasoning_content (进度信息)
    reasoning = delta.get("reasoning_content")
    if reasoning:
        stage = reasoning.get("stage")
        status = reasoning.get("status")
        progress = reasoning.get("progress")
        message = reasoning.get("message")
        details = reasoning.get("details")
        
        print(f"[{stage}] {status}: {message}")
        if progress is not None:
            print(f"  Progress: {progress}%")
        if details:
            print(f"  Details: {details}")
    
    # 解析 content (最终结果 - JSON 格式)
    content = delta.get("content")
    if content:
        result = json.loads(content)
        result_type = result.get("type")
        
        if result_type == "image":
            print(f"图片生成成功: {result['count']} 张")
            for item in result["data"]:
                print(f"  URL: {item['url']}")
        
        elif result_type == "video":
            print(f"视频生成成功: {result['url']}")
        
        elif result_type == "character":
            print(f"角色创建成功: @{result['username']} ({result['display_name']})")
            print(f"  cameo_id: {result['cameo_id']}")
            print(f"  character_id: {result['character_id']}")
        
        elif result_type == "error":
            print(f"生成失败: {result['error']}")
    
    return data

# 使用示例
for line in response.iter_lines():
    if line:
        parse_stream_response(line.decode())
```

### 解析示例 (JavaScript)

```javascript
async function parseStreamResponse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      
      const dataStr = line.slice(6);
      if (dataStr === '[DONE]') {
        console.log('Stream completed');
        continue;
      }
      
      const data = JSON.parse(dataStr);
      const delta = data.choices[0].delta;
      
      // 解析 reasoning_content (进度信息)
      const reasoning = delta.reasoning_content;
      if (reasoning) {
        console.log(`[${reasoning.stage}] ${reasoning.status}: ${reasoning.message}`);
        if (reasoning.progress !== undefined) {
          console.log(`  Progress: ${reasoning.progress}%`);
        }
        if (reasoning.details) {
          console.log('  Details:', reasoning.details);
        }
      }
      
      // 解析 content (最终结果 - JSON 格式)
      if (delta.content) {
        const result = JSON.parse(delta.content);
        
        switch (result.type) {
          case 'image':
            console.log(`图片生成成功: ${result.count} 张`);
            result.data.forEach(item => console.log(`  URL: ${item.url}`));
            break;
          case 'video':
            console.log(`视频生成成功: ${result.url}`);
            break;
          case 'character':
            console.log(`角色创建成功: @${result.username} (${result.display_name})`);
            console.log(`  cameo_id: ${result.cameo_id}`);
            console.log(`  character_id: ${result.character_id}`);
            break;
          case 'error':
            console.log(`生成失败: ${result.error}`);
            break;
        }
      }
    }
  }
}
```

---

## 接口示例

### 1. 文生图片

**请求**
```json
{
  "model": "sora-image",
  "messages": [
    {
      "role": "user",
      "content": "一只可爱的小猫"
    }
  ],
  "stream": true
}
```

**响应 (流式)**

流式响应包含两种类型的数据：
- `reasoning_content`: 结构化的处理进度信息 (JSON 对象)
- `content`: 最终生成结果

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"sora","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":{"stage":"generation","status":"started","message":"Initializing generation request...","timestamp":1234567890},"tool_calls":null},"finish_reason":null,"native_finish_reason":null}],"usage":{"prompt_tokens":0}}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"sora","choices":[{"index":0,"delta":{"content":null,"reasoning_content":{"stage":"generation","status":"processing","progress":50,"message":"Image generation in progress: 50% completed...","timestamp":1234567890},"tool_calls":null},"finish_reason":null,"native_finish_reason":null}],"usage":{"prompt_tokens":0}}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"sora","choices":[{"index":0,"delta":{"content":"![Generated Image](http://localhost:8000/tmp/xxx.png)","reasoning_content":null,"tool_calls":null},"finish_reason":"STOP","native_finish_reason":"STOP"}],"usage":{"prompt_tokens":0,"completion_tokens":1,"total_tokens":1}}

data: [DONE]
```

**curl**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-image",
    "messages": [{"role": "user", "content": "一只可爱的小猫"}],
    "stream": true
  }'
```

---

### 2. 文生视频

**请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": "一只猫在弹钢琴"
    }
  ],
  "stream": true,
  "style_id": "anime"
}
```

**curl**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-video-landscape-10s",
    "messages": [{"role": "user", "content": "一只猫在弹钢琴"}],
    "stream": true,
    "style_id": "anime"
  }'
```

---

### 3. 图生视频

**请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "让图片动起来"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo..."
          }
        }
      ]
    }
  ],
  "stream": true
}
```

**说明**
- 图片可以通过 base64 编码传入
- 也支持 URL 形式: `{"url": "https://example.com/image.png"}`

---

### 4. 视频 Remix（基于已有视频继续创作）

提示词内包含 Sora 分享链接或 ID 即可触发 Remix 模式。

**请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": "https://sora.chatgpt.com/p/s_68e3a06dcd888191b150971da152c1f5 改成水墨画风格"
    }
  ],
  "stream": true
}
```

**curl**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-video-landscape-10s",
    "messages": [
      {
        "role": "user",
        "content": "https://sora.chatgpt.com/p/s_68e3a06dcd888191b150971da152c1f5 改成水墨画风格"
      }
    ]
  }'
```

**说明**
- 支持完整分享链接: `https://sora.chatgpt.com/p/s_xxx`
- 支持分享 ID: `s_xxx`
- Remix 会基于原视频进行二次创作

---

### 5. 视频分镜（Storyboard）

使用 `[时长]提示词` 格式触发分镜模式，可以精确控制每个片段的时长和内容。

**请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": "[5.0s]猫猫从飞机上跳伞 [5.0s]猫猫降落 [10.0s]猫猫在田野奔跑"
    }
  ],
  "stream": true
}
```

**curl**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-video-landscape-10s",
    "messages": [
      {
        "role": "user",
        "content": "[5.0s]猫猫从飞机上跳伞 [5.0s]猫猫降落 [10.0s]猫猫在田野奔跑"
      }
    ]
  }'
```

**说明**
- 格式: `[时长s]提示词`，例如 `[5.0s]场景描述`
- 支持多行格式或空格分隔
- 每个片段可以设置不同的时长 (5s, 10s, 15s, 20s)

---

### 6. 创建角色

通过上传视频创建角色卡，可用于后续视频生成中引用角色。

**基础请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": {
            "url": "data:video/mp4;base64,..."
          }
        }
      ]
    }
  ],
  "stream": true,
  "character_options": {
    "username": "my_character",
    "display_name": "我的角色"
  }
}
```

**完整参数请求**
```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": {
            "url": "https://example.com/video.mp4"
          }
        }
      ]
    }
  ],
  "stream": true,
  "character_options": {
    "timestamps": "0,3",
    "username": "cat_character",
    "display_name": "可爱的猫咪",
    "instruction_set": "A cute orange cat with fluffy fur",
    "safety_instruction_set": "family-friendly content only"
  }
}
```

**curl**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-video-landscape-10s",
    "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}]}],
    "stream": true,
    "character_options": {
      "timestamps": "0,3",
      "username": "my_cat",
      "display_name": "我的猫咪"
    }
  }'
```

**character_options 参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `timestamps` | string | 视频时间戳（秒），用于提取角色帧，格式 `"起始,结束"`，如 `"0,3"` |
| `username` | string | 角色用户名，用于在提示词中引用，如 `@my_cat` |
| `display_name` | string | 角色显示名称 |
| `instruction_set` | string | 角色描述/指令集，描述角色外观特征 |
| `safety_instruction_set` | string | 安全指令集，限制生成内容类型 |

**使用创建的角色**

创建角色后，可以在提示词中使用 `@username` 引用：

```json
{
  "model": "sora-video-landscape-10s",
  "messages": [
    {
      "role": "user",
      "content": "@my_cat 在草地上奔跑"
    }
  ],
  "stream": true
}
```

---

### 7. 获取模型列表

**请求**
```bash
curl -X GET "http://localhost:8000/v1/models" \
  -H "Authorization: Bearer han1234"
```

**响应**
```json
{
  "object": "list",
  "data": [
    {"id": "sora-image", "object": "model", "owned_by": "openai"},
    {"id": "sora-video-landscape-10s", "object": "model", "owned_by": "openai"}
  ]
}
```

---

### 7. Token 管理

**获取 Token 列表**
```bash
curl -X GET "http://localhost:8000/api/tokens" \
  -H "Authorization: Bearer han1234"
```

**添加 Token**
```bash
curl -X POST "http://localhost:8000/api/tokens" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{"token": "your_sora_token_here"}'
```

**删除 Token**
```bash
curl -X DELETE "http://localhost:8000/api/tokens/1" \
  -H "Authorization: Bearer han1234"
```

---

### 8. 搜索用户/角色

**请求**
```bash
curl -X GET "http://localhost:8000/v1/characters/search?username=test&intent=cameo&limit=10" \
  -H "Authorization: Bearer han1234"
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `username` | string | 否 | 搜索的用户名关键字 |
| `intent` | string | 否 | `users` (所有用户，默认) 或 `cameo` (可用于视频生成的角色) |
| `token_id` | int | 否 | 指定使用的 Token ID |
| `limit` | int | 否 | 返回数量，默认 10 |

---

### 9. 获取公共 Feed

**请求**
```bash
curl -X GET "http://localhost:8000/v1/feed?limit=8&cut=nf2_latest" \
  -H "Authorization: Bearer han1234"
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `limit` | int | 否 | 返回数量，默认 8 |
| `cut` | string | 否 | `nf2_latest` (最新，默认) 或 `nf2_top` (热门) |
| `cursor` | string | 否 | 分页游标 |
| `token_id` | int | 否 | 指定使用的 Token ID |

---

### 10. 获取 Token 发布内容

**请求**
```bash
curl -X GET "http://localhost:8000/v1/tokens/1/profile-feed?limit=12" \
  -H "Authorization: Bearer han1234"
```

---

## Python 示例

```python
import requests
import base64

API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "han1234"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 文生视频
response = requests.post(API_URL, headers=headers, json={
    "model": "sora-video-landscape-10s",
    "messages": [{"role": "user", "content": "一只猫在跳舞"}],
    "stream": True,
    "style_id": "anime"
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode())

# 图生视频
with open("image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(API_URL, headers=headers, json={
    "model": "sora-video-landscape-10s",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "让图片动起来"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
    }],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

---

## OpenAI 标准格式 API

除了 `/v1/chat/completions` 端点外，还提供以下 OpenAI 标准格式的独立端点。

**所有端点均为非流式输出，返回标准 JSON 结果。**

> 📖 **详细文档**: [docs/OPENAI_API.md](docs/OPENAI_API.md)

### 端点概览

| 端点 | 功能 | 请求格式 |
|------|------|----------|
| `POST /v1/videos` | 视频生成 | form-data / JSON |
| `POST /v1/images/generations` | 图片生成 | form-data / JSON |
| `POST /v1/characters` | 角色卡创建 | form-data / JSON |

### 快速示例

**视频生成:**

```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking in the garden", "model": "sora-video-10s"}'
```

```json
{
  "id": "video-xxx",
  "object": "video",
  "created": 1702388400,
  "model": "sora-video-10s",
  "data": [{"url": "http://localhost:8000/tmp/xxx.mp4", "revised_prompt": "A cat walking in the garden"}]
}
```

**图片生成:**

```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset", "model": "sora-image-landscape"}'
```

```json
{
  "created": 1702388400,
  "data": [{"url": "http://localhost:8000/tmp/xxx.png", "revised_prompt": "A beautiful sunset"}]
}
```

**角色卡创建:**

```bash
curl -X POST "http://localhost:8000/v1/characters" \
  -H "Authorization: Bearer han1234" \
  -F video="@video.mp4;type=video/mp4" \
  -F timestamps="0,3" \
  -F username="my_cat"
```

```json
{
  "id": "char_xxx",
  "object": "character",
  "created": 1702388400,
  "model": "sora-video-10s",
  "data": {"cameo_id": "ch_xxx", "username": "my_cat", "message": "Character creation completed"}
}
```

**使用角色生成视频 (在 prompt 中使用 @username):**

```bash
curl -X POST "http://localhost:8000/v1/videos" \
  -H "Authorization: Bearer han1234" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "@my_cat running in the park", "model": "sora-video-10s"}'
```

---

## 测试脚本

在 `tests/` 目录下提供了测试脚本：

```bash
# 测试搜索角色 API
python tests/test_search_characters.py

# 测试公共 Feed API
python tests/test_public_feed.py

# 测试 Token 发布内容 API
python tests/test_token_feed.py
```

---

## 许可证

MIT License

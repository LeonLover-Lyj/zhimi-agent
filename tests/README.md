# 知觅Agent测试说明

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # pytest配置和fixtures
├── test_agent.py            # 自动化测试脚本
├── manual_test_guide.md     # 手动测试指南
└── README.md                # 本文件
```

## 运行测试

### 自动化测试

```bash
# 在虚拟环境中运行所有测试
python -m pytest tests/test_agent.py -v

# 运行特定测试类
python -m pytest tests/test_agent.py::TestSearchTools -v

# 运行特定测试方法
python -m pytest tests/test_agent.py::TestSearchTools::test_simple_keyword_search_tool_build -v
```

### 手动测试

1. 启动应用：运行 `run.bat`（Windows）或 `run.sh`（Linux/Mac）
2. 按照 `manual_test_guide.md` 中的测试用例逐一验证
3. 记录测试结果

## 测试覆盖

### 已实现的测试

1. **搜索工具测试**（TestSearchTools）
   - ✅ 工具构建测试
   - ✅ 无索引时的防御式处理
   - ✅ 有索引时的功能测试

2. **工具选择测试**（TestToolSelection）
   - ✅ 工具描述清晰度测试

### 需要环境配置的测试

以下测试需要完整的LangChain环境和API key，在导入agent模块时会跳过：

1. **对话历史管理测试**（TestConversationHistory）
   - 需要修复agent.py中的导入问题后才能运行

2. **Agent集成测试**（TestAgentIntegration）
   - 需要API key和完整的LangChain环境

## 测试结果

### 最新测试结果

```
7 passed, 7 deselected, 4 warnings
```

- ✅ 搜索工具相关测试全部通过
- ⚠️ 对话历史测试因导入问题跳过（需要修复agent.py）
- ⚠️ Agent集成测试需要API key，已标记为skip

## 已知问题

1. **agent.py导入问题**：`create_tool_calling_agent` 在当前LangChain版本中可能不存在或路径不同
2. **BM25Retriever API变化**：已修复，使用兼容性处理

## 下一步

1. 修复agent.py中的导入问题，使对话历史测试可以运行
2. 配置API key后运行Agent集成测试
3. 添加更多端到端测试用例


# GitHub 仓库上传指南

## 当前状态

✅ Git 仓库已初始化
✅ 文件已添加到暂存区
✅ 初始提交已完成

## 下一步：连接到 GitHub

### 方法1：在 GitHub 上创建新仓库（推荐）

1. **访问 GitHub**：https://github.com/new
2. **创建新仓库**：
   - Repository name: `zhimi-agent`（或你喜欢的名称）
   - Description: "知觅Agent - 基于LangChain和Qwen的本地知识问答系统"
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
   - 点击 "Create repository"

3. **添加远程仓库并推送**：
   ```bash
   # 替换 YOUR_USERNAME 为你的GitHub用户名
   git remote add origin https://github.com/LeonLover-Lyj/zhimi-agent.git
   git branch -M main
   git push -u origin main
   ```

### 方法2：如果仓库已存在

如果你已经在GitHub上创建了仓库，直接运行：

```bash
# 替换为你的实际仓库URL
git remote add origin https://github.com/YOUR_USERNAME/zhimi-agent.git
git branch -M main
git push -u origin main
```

## 验证

推送成功后，访问你的GitHub仓库页面，应该能看到所有文件。

## 后续更新

以后更新代码时：

```bash
git add .
git commit -m "描述你的更改"
git push
```

## 注意事项

1. **不要提交敏感信息**：
   - `.env` 文件已在 `.gitignore` 中
   - 确保 `.env` 文件不会被提交

2. **大文件**：
   - `memory/faiss_index/` 目录已被忽略（包含大文件）
   - 如果需要版本控制索引，考虑使用 Git LFS

3. **API Keys**：
   - 确保 `.env` 文件不会被提交
   - 使用 `.env.example` 作为模板

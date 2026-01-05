import argparse
import time
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 使用新的导入方式
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "memory/faiss_index"
# 使用更轻量级的模型，减少加载时间和内存使用
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 约300MB，速度更快

def load_docs(data_dir: Path):
    """加载指定目录下的文档"""
    docs = []
    print("🔍 正在扫描文档...")
    
    # 获取所有支持的文件
    supported_files = []
    for p in data_dir.rglob("*"):
        if p.suffix in [".txt", ".pdf", ".md", ".markdown"]:
            supported_files.append(p)
    
    print(f"📂 找到 {len(supported_files)} 个支持的文档文件")
    
    # 加载文档
    for i, p in enumerate(supported_files, 1):
        print(f"📄 正在加载 ({i}/{len(supported_files)}): {p.name}")
        try:
            if p.suffix == ".txt":
                docs += TextLoader(str(p), encoding="utf-8").load()
            elif p.suffix == ".pdf":
                docs += PyPDFLoader(str(p)).load()
            elif p.suffix in [".md", ".markdown"]:
                docs += UnstructuredMarkdownLoader(str(p)).load()
        except Exception as e:
            print(f"  ⚠️  加载失败: {e}")
    
    print(f"✅ 成功加载 {len(docs)} 个文档")
    return docs

def main(data_dir: str):
    """主函数：构建文档索引"""
    print("=" * 50)
    print("📚 开始构建文档向量索引")
    print("=" * 50)
    
    # 1. 加载文档
    raw_docs = load_docs(Path(data_dir))
    if not raw_docs:
        print("❌ 没有找到可处理的文档，请检查目录路径")
        return
    
    # 2. 分割文档
    print("\n✂️ 正在分割文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，"]
    )
    docs = splitter.split_documents(raw_docs)
    print(f"📝 文档分割完成: {len(raw_docs)} → {len(docs)} 个片段")
    
    # 3. 加载嵌入模型（使用新的 HuggingFaceEmbeddings）
    print("\n🤖 正在加载嵌入模型...")
    print("   ⏳ 首次使用需要下载模型，请耐心等待（约300MB）")
    start_time = time.time()
    
    # 使用新的 HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},  # 使用CPU，如需GPU可改为 "cuda"
        encode_kwargs={
            "normalize_embeddings": True,  # 归一化向量
            "show_progress_bar": True      # 显示编码进度条
        }
    )
    
    load_time = time.time() - start_time
    print(f"   ✅ 模型加载完成，耗时: {load_time:.1f}秒")
    
    # 4. 构建向量索引
    print("\n🔧 正在构建向量索引...")
    index_start_time = time.time()
    vs = FAISS.from_documents(docs, embeddings)
    index_time = time.time() - index_start_time
    print(f"   ✅ 索引构建完成，耗时: {index_time:.1f}秒")
    
    # 5. 保存索引
    print("\n💾 正在保存索引...")
    vs.save_local(INDEX_PATH)
    
    # 统计信息
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("🎉 索引构建完成!")
    print("=" * 50)
    print(f"📊 统计信息:")
    print(f"   📂 文档目录: {data_dir}")
    print(f"   📄 原始文档: {len(raw_docs)} 个")
    print(f"   📝 文本片段: {len(docs)} 个")
    print(f"   🤖 嵌入模型: {EMBED_MODEL}")
    print(f"   📁 索引路径: {INDEX_PATH}")
    print(f"   ⏱️  总耗时: {total_time:.1f}秒")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建本地文档向量索引")
    parser.add_argument("--dir", required=True, help="包含文档的目录路径")
    args = parser.parse_args()
    main(args.dir)
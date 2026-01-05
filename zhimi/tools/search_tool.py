from pathlib import Path
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field

INDEX_PATH = "memory/faiss_index"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

def load_retrievers():
    if not Path(INDEX_PATH).exists():
        return None, None
    faiss = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    bm25 = BM25Retriever.from_documents(list(faiss.docstore._dict.values()))
    bm25.k = 2
    return faiss, bm25

faiss, bm25 = load_retrievers()

def simple_keyword_search(query: str) -> str:
    """对本地文档进行简单的关键词匹配检索
    
    适用于明确的术语、名称、具体关键词查询。
    通过文本匹配查找包含查询关键词的文档片段。
    """
    if faiss is None:
        return "⚠️ 本地知识库尚未构建，请先构建索引。"
    
    # 从FAISS索引中提取所有文档
    all_docs = list(faiss.docstore._dict.values())
    
    if not all_docs:
        return "未找到相关本地信息。"
    
    # 提取查询关键词（简单分词，去除常见停用词）
    query_lower = query.lower()
    query_terms = [term for term in query_lower.split() if len(term) > 1]
    
    # 如果查询词太短，直接使用整个查询
    if not query_terms:
        query_terms = [query_lower]
    
    # 对每个文档进行关键词匹配
    matched_docs = []
    for doc in all_docs:
        content_lower = doc.page_content.lower()
        # 计算匹配的关键词数量
        match_count = sum(1 for term in query_terms if term in content_lower)
        if match_count > 0:
            matched_docs.append((match_count, doc))
    
    # 按匹配数量排序，取前3个
    matched_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in matched_docs[:3]]
    
    if not top_docs:
        return "未找到包含相关关键词的本地信息。"
    
    # 返回匹配的文档内容
    results = [doc.page_content for doc in top_docs]
    return "\n\n---\n\n".join(results)

def hybrid_search(query: str) -> str:
    """使用向量检索和关键词检索的混合方法
    
    适用于需要理解语义、上下文、概念的问题。
    结合FAISS向量相似度检索和BM25关键词检索，提供更准确的搜索结果。
    """
    if faiss is None or bm25 is None:
        return "⚠️ 本地知识库尚未构建，请先构建索引。"
    
    # FAISS向量检索
    faiss_docs = faiss.similarity_search(query, k=2)
    
    # BM25关键词检索（使用invoke方法，兼容新版本API）
    try:
        bm25_docs = bm25.invoke(query) if hasattr(bm25, 'invoke') else bm25.get_relevant_documents(query)
    except AttributeError:
        # 如果都没有，尝试直接调用
        bm25_docs = []
    
    docs = faiss_docs + (bm25_docs if isinstance(bm25_docs, list) else [])
    uniq = {d.page_content: d for d in docs}
    return "\n\n---\n\n".join(uniq.keys()) or "未找到相关本地信息。"

class SearchInput(BaseModel):
    query: str = Field(description="用户问题或查询关键词")

def build_simple_search_tool():
    """构建简单关键词检索工具"""
    return Tool.from_function(
        func=simple_keyword_search,
        name="simple_keyword_search",
        description="简单关键词检索工具。适用于明确的术语、名称、具体关键词查询。当用户询问具体的名称、术语、关键词时使用此工具。例如：'知觅是什么'、'如何安装'、'配置文件位置'等。",
        args_schema=SearchInput,
    )

def build_search_tool():
    """构建混合检索工具"""
    return Tool.from_function(
        func=hybrid_search,
        name="hybrid_search",
        description="混合检索工具（向量检索+关键词检索）。适用于需要理解语义、上下文、概念的问题。当用户询问需要理解含义、上下文关系、概念解释的问题时使用此工具。例如：'解释一下工作原理'、'它们之间的关系是什么'、'这个概念如何应用'等。",
        args_schema=SearchInput,
    )


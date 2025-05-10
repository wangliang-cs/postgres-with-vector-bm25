import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from psycopg2 import extensions

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "database": "compass",
    "user": "nju_common",
    "password": "opensource"
}

# 注册向量类型适配器
def adapt_vector(vector):
    return extensions.QuotedString(f"[{','.join(map(str, vector))}]")

# 连接到数据库并注册适配器
conn = psycopg2.connect(**DB_CONFIG)
extensions.register_adapter(np.ndarray, adapt_vector)
extensions.register_adapter(list, adapt_vector)
conn.close()

def setup_database():
    """创建数据库表和索引"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;  -- 确保向量扩展已安装
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding VECTOR(384),  -- 使用384维向量
        metadata JSONB
    );
    """)

    # 创建全文搜索索引
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_documents_content_search ON documents
    USING gin(to_tsvector('english', content));
    """)

    # 创建向量索引 (使用IVFFlat或HNSW)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents
    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("数据库表和索引创建完成")

def insert_sample_data():
    """插入样例数据"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 样例文档数据
    sample_documents = [
        {
            "title": "人工智能概述 Artificial Intelligence",
            "content": "Artificial Intelligence 人工智能 是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
            "metadata": {"category": "technology", "author": "张三"}
        },
        {
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个分支，它通过算法使计算机能够从数据中学习并做出决策或预测。",
            "metadata": {"category": "technology", "author": "李四"}
        },
        {
            "title": "深度学习进展",
            "content": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的工作方式。",
            "metadata": {"category": "technology", "author": "王五"}
        },
        {
            "title": "数据库系统原理",
            "content": "数据库系统是计算机系统中存储、管理和处理数据的核心组件，包括关系型和非关系型数据库。",
            "metadata": {"category": "database", "author": "赵六"}
        },
        {
            "title": "PostgreSQL高级特性",
            "content": "PostgreSQL是一个功能强大的开源关系数据库系统，支持扩展如向量搜索和全文检索。",
            "metadata": {"category": "database", "author": "钱七"}
        }
    ]

    # 为每个文档生成随机向量 (实际应用中应使用真实嵌入模型)
    for doc in sample_documents:
        # 生成384维随机向量 (实际应用中应使用模型生成)
        embedding = np.random.rand(384).astype(np.float32).tolist()
        doc["embedding"] = embedding
        # 将metadata字典转换为JSON字符串
        doc["metadata_json"] = psycopg2.extras.Json(doc["metadata"])

    # 插入数据
    execute_values(
        cursor,
        """
        INSERT INTO documents (title, content, embedding, metadata)
        VALUES %s
        """,
        [
            (doc["title"], doc["content"], doc["embedding"], doc["metadata_json"])
            for doc in sample_documents
        ]
    )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"插入了 {len(sample_documents)} 条样例数据")

def hybrid_search(query_text, query_vector=None, vector_weight=0.0, text_weight=1.0, top_k=5):
    """
    执行混合检索 (向量 + BM25)

    参数:
        query_text: 查询文本 (用于BM25)
        query_vector: 查询向量 (如果为None，则生成随机向量)
        vector_weight: 向量相似度权重
        text_weight: 文本相似度权重
        top_k: 返回结果数量
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 如果没有提供查询向量，生成一个随机向量 (实际应用中应使用模型生成)
    if query_vector is None:
        query_vector = np.random.rand(384).astype(np.float32).tolist()

    # 执行混合查询，不再需要显式类型转换
    cursor.execute("""
    SELECT
        id,
        title,
        content,
        metadata,
        (1 - (embedding <=> %s)) AS vector_score,
        ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS text_score,
        %s * (1 - (embedding <=> %s)) + %s * ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS combined_score
    FROM documents
    ORDER BY combined_score DESC
    LIMIT %s;
    """, (query_vector,
          query_text,
          vector_weight, query_vector,
          text_weight, query_text,
          top_k))

    results = cursor.fetchall()

    print(f"\n混合检索结果 (向量权重: {vector_weight}, 文本权重: {text_weight}):")
    print("="*80)
    for i, row in enumerate(results):
        print(f"\n结果 {i+1} (综合得分: {row[6]:.4f})")
        print(f"ID: {row[0]}")
        print(f"标题: {row[1]}")
        print(f"内容: {row[2][:100]}...")  # 只显示前100个字符
        print(f"元数据: {row[3]}")
        print(f"向量得分: {row[4]:.4f}, 文本得分: {row[5]:.4f}")

    cursor.close()
    conn.close()
    return results

if __name__ == "__main__":
    # 初始化数据库
    setup_database()
    insert_sample_data()

    # 执行混合检索示例
    query_text = "人工智能"
    # query_text = "Artificial Intelligence"

    # 为示例生成一个固定的查询向量 (实际应用中应使用模型生成)
    np.random.seed(42)  # 为了示例可重复性
    example_query_vector = np.random.rand(384).astype(np.float32).tolist()

    # 执行查询
    hybrid_search(query_text, example_query_vector)

    # 尝试不同的权重组合
    # print("\n尝试不同的权重组合:")
    # hybrid_search(query_text, example_query_vector, vector_weight=0.9, text_weight=0.1)
    # hybrid_search(query_text, example_query_vector, vector_weight=0.1, text_weight=0.9)

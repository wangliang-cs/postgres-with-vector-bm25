import psycopg2
import numpy as np
import json
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
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    extensions.register_adapter(np.ndarray, adapt_vector)
    extensions.register_adapter(list, adapt_vector)
    return conn

def setup_database():
    """创建数据库表和索引"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 创建表
            cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS pg_search;

            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(384)
            );
            """)

            # 创建文本搜索GIN索引
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_text_search ON documents
            USING gin (to_tsvector('simple', title || ' ' || content));
            """)

            # 创建向量索引
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents
            USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
            """)

        conn.commit()
    print("数据库表和索引创建完成")

def insert_sample_data():
    """插入样例数据"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 样例文档数据
            sample_documents = [
                {
                    "title": "人工智能概述",
                    "content": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
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

            # 为每个文档生成随机向量
            for doc in sample_documents:
                embedding = np.random.rand(384).astype(np.float32).tolist()
                doc["embedding"] = embedding

            # 插入数据，将metadata转换为JSON字符串
            execute_values(
                cursor,
                """
                INSERT INTO documents (title, content, embedding, metadata)
                VALUES %s
                """,
                [
                    (doc["title"], doc["content"], doc["embedding"], json.dumps(doc["metadata"]))
                    for doc in sample_documents
                ]
            )

        conn.commit()
    print(f"插入了 {len(sample_documents)} 条样例数据")

def hybrid_search(query_text, query_vector=None, vector_weight=0.6, bm25_weight=0.4, top_k=10, metadata_filter=None):
    """
    执行混合检索 (BM25 + 向量)

    参数:
        query_text: 查询文本
        query_vector: 查询向量
        vector_weight: 向量相似度权重
        bm25_weight: BM25权重
        top_k: 返回结果数量
        metadata_filter: 元数据过滤条件 (例如: "category:technology")
    """
    if query_vector is None:
        query_vector = np.random.rand(384).astype(np.float32).tolist()

    # 构建SQL查询
    base_sql = """
    SELECT
        id,
        title,
        content,
        metadata,
        (%s * ts_rank_cd(to_tsvector('simple', title || ' ' || content), plainto_tsquery('simple', %s))) AS bm25_score,
        (%s * (1 - (embedding <=> %s))) AS vector_score,
        (%s * ts_rank_cd(to_tsvector('simple', title || ' ' || content), plainto_tsquery('simple', %s))) +
        (%s * (1 - (embedding <=> %s))) AS combined_score
    FROM
        documents
    WHERE
        to_tsvector('simple', title || ' ' || content) @@ plainto_tsquery('simple', %s)
    """
    
    # 添加元数据过滤条件
    if metadata_filter:
        key, value = metadata_filter.split(':')
        base_sql += f" AND metadata @> '{{\"{key}\": \"{value}\"}}'::jsonb"
    
    # 添加排序和限制
    base_sql += """
    ORDER BY
        combined_score DESC
    LIMIT %s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 执行混合查询
            if metadata_filter:
                cursor.execute(base_sql, (
                    bm25_weight, query_text, vector_weight, query_vector,
                    bm25_weight, query_text, vector_weight, query_vector,
                    query_text, top_k
                ))
            else:
                cursor.execute(base_sql, (
                    bm25_weight, query_text, vector_weight, query_vector,
                    bm25_weight, query_text, vector_weight, query_vector,
                    query_text, top_k
                ))

            results = cursor.fetchall()

    print(f"\n混合检索结果 (向量权重: {vector_weight}, BM25权重: {bm25_weight}):")
    print("="*80)
    for i, row in enumerate(results):
        print(f"\n结果 {i+1} (综合得分: {row[6]:.4f})")
        print(f"ID: {row[0]}")
        print(f"标题: {row[1]}")
        print(f"内容: {row[2][:100]}...")
        print(f"元数据: {row[3]}")
        print(f"BM25得分: {row[4]:.4f}, 向量得分: {row[5]:.4f}")

    return results

if __name__ == "__main__":
    # 初始化数据库
    setup_database()
    insert_sample_data()

    # 执行混合检索示例
    query_text = "人工智能"
    np.random.seed(42)  # 为了示例可重复性
    example_query_vector = np.random.rand(384).astype(np.float32).tolist()

    # 执行基础查询
    hybrid_search(query_text, example_query_vector)

    # 尝试不同的权重组合
    print("\n尝试不同的权重组合:")
    hybrid_search(query_text, example_query_vector, vector_weight=0.8, bm25_weight=0.2)
    hybrid_search(query_text, example_query_vector, vector_weight=0.2, bm25_weight=0.8)

    # 执行带元数据过滤的查询
    print("\n执行带元数据过滤的查询:")
    hybrid_search(query_text, example_query_vector, metadata_filter="category:database")

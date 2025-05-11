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


def setup_database(table_name="summary_aug_keywords"):
    """创建数据库表和索引"""
    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cursor:
        # 执行删除表操作（如果表存在）
        cursor.execute("DROP TABLE IF EXISTS your_table_name")
        conn.commit()  # 提交事务
        print("表已成功删除（如果存在）")

    cursor = conn.cursor()
    # 创建表
    cursor.execute(f"""
    CREATE EXTENSION IF NOT EXISTS vector;  -- 确保向量扩展已安装
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        package_id TEXT,
        package_name TEXT,
        ecosystem TEXT,
        summary TEXT,
        augmented_keywords TEXT,
        summary_embedding VECTOR(768),  -- 使用384维向量
        keywords_embedding VECTOR(768)  -- 使用384维向量
    );
    """)

    cursor.execute(f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_package_id ON {table_name} (package_id);
    """)

    cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_ecosystem ON {table_name} (ecosystem);
    """)

    # 创建全文搜索索引
    cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_summary_search ON {table_name}
    USING gin(to_tsvector('english', summary));
    """)

    # 创建关键词搜索索引
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_keywords_search ON {table_name}
        USING gin(to_tsvector('english', augmented_keywords));
        """)

    # 创建向量索引 (使用IVFFlat或HNSW)
    cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_summary_embedding ON {table_name}
    USING ivfflat (summary_embedding vector_l2_ops) WITH (lists = 100);
    """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_keywords_embedding ON {table_name}
        USING ivfflat (keywords_embedding vector_l2_ops) WITH (lists = 100);
        """)

    conn.commit()
    cursor.close()
    conn.close()
    print("数据库表和索引创建完成")

# 样例文档数据
sample_documents = [
    {
        "package_id": "fast-copy@@@@$$@@@@npm",
        "package_name": "fast-copy",
        "ecosystem": "npm",
        "summary": "The library is a high-performance deep object copier designed to create deep copies of complex objects efficiently. It supports a wide range of object types, including arrays, maps, sets, dates, custom constructors, and more, while handling circular references automatically. The default copier prioritizes speed and uses the original object's constructor for replication. Key features include: Deep Copying: Creates independent copies of nested objects, ensuring no shared references. Strict Mode: An optional strict mode copies non-enumerable properties and retains property descriptors, though it is slower. Custom Copiers: Allows creating tailored copiers for specific use cases or performance optimization. Circular Reference Support: Detects and handles circular references without infinite recursion. Type Preservation: Maintains the original object's type, including custom subclasses and prototypes. The library outperforms alternatives like Lodash's cloneDeep and Ramda in benchmarks, especially for large or nested objects. It avoids copying certain types like errors, functions, and promises, as they typically don't require deep copying. The focus is on balancing correctness with speed for common use cases.",
        "augmented_keywords": "circular clone copi custom deep fast object refer fast-copy",
        "summary_embedding": np.random.rand(768).astype(np.float32).tolist(), #np.repeat(0.1, 768).astype(np.float32).tolist(),
        "keywords_embedding": np.random.rand(768).astype(np.float32).tolist()
    },
    {
        "package_id": "markdown-it-emoji@@@@$$@@@@npm",
        "package_name": "markdown-it-emoji",
        "ecosystem": "npm",
        "summary": "This library is a plugin for the markdown-it markdown parser that adds support for emoji and emoticon syntax. It allows users to include emojis in their markdown text using standard emoji codes (e.g., :satellite:) and emoticon shortcuts (e.g., :), :-(). The plugin provides three configuration presets: full (all emojis), light (a small subset of commonly used emojis), and bare (no default emojis). Users can customize emoji definitions, enable or disable specific emojis, and override default shortcuts. By default, emojis are rendered as Unicode characters, but the renderer can be customized to output emojis as HTML spans (e.g., for custom fonts) or using external libraries like Twemoji for image-based emojis. The plugin is compatible with both Node.js and browser environments.",
        "augmented_keywords": "default e.g. emoji emoticon markdown markdown-it markdown-it-plugin plugin markdown-it-emoji",
        "summary_embedding": np.random.rand(768).astype(np.float32).tolist(),
        "keywords_embedding": np.random.rand(768).astype(np.float32).tolist()
    },
    {
        "package_id": "flatten@@@@$$@@@@npm",
        "package_name": "flatten",
        "ecosystem": "npm",
        "summary": "The library is a small utility designed to flatten nested arrays into a single, non-nested list of elements. It handles arrays nested to any depth, recursively unwrapping them to produce a flat array. Users can optionally specify a depth limit to control how many levels of nesting are flattened. For example, if the depth is set to 2, only the first two levels of nested arrays are flattened, leaving deeper structures intact. The tool works with arrays containing mixed data types, including non-array elements, and processes them in the order they appear. It is maintained primarily for backward compatibility with older projects. The core functionality simplifies working with deeply nested arrays by converting them into a linear sequence, making it easier to iterate or process the elements. The library does not modify the original input arrays but returns a new flattened array. It is lightweight and focused solely on array flattening, with no additional dependencies or extended features.",
        "augmented_keywords": "array depth element flatten level nest",
        "summary_embedding": np.random.rand(768).astype(np.float32).tolist(),
        "keywords_embedding": np.random.rand(768).astype(np.float32).tolist()
    },
    {
        "package_id": "punycode@@@@$$@@@@npm",
        "package_name": "punycode",
        "ecosystem": "npm",
        "summary": "The library is a robust Punycode converter that fully complies with RFC 3492 and RFC 5891 standards, designed to work across nearly all JavaScript platforms. It converts between Unicode symbols and Punycode, a specialized encoding used to represent Unicode characters within the limited ASCII character set of domain names and email addresses. Key functions include: decode(): Converts a Punycode string of ASCII symbols to Unicode symbols. encode(): Converts a Unicode string to a Punycode ASCII string. toUnicode(): Converts Punycode domain names or email addresses to Unicode, leaving already Unicode parts unchanged. toASCII(): Converts Unicode domain names or email addresses to Punycode, leaving ASCII parts unchanged. Additionally, it provides UCS-2/UTF-16 utilities: ucs2.decode(): Converts a string into an array of Unicode code points, handling surrogate pairs. ucs2.encode(): Converts an array of code points back into a string. The library is optimized for modern JavaScript environments but also supports older runtimes via a legacy version. Its core purpose is enabling accurate and standards-compliant IDN (Internationalized Domain Name) processing.",
        "augmented_keywords": "ascii convert dn domain idn idna punycod string unicod url punycode",
        "summary_embedding": np.random.rand(768).astype(np.float32).tolist(),
        "keywords_embedding": np.random.rand(768).astype(np.float32).tolist()
    },
    {
        "package_id": "@pulumi/docker@@@@$$@@@@npm",
        "package_name": "@pulumi/docker",
        "ecosystem": "npm",
        "summary": "The library is a Pulumi package designed to facilitate interactions with Docker within Pulumi programs. It enables users to manage Docker resources, such as containers, images, networks, and volumes, programmatically using infrastructure-as-code principles. By leveraging this package, developers can define, deploy, and manage Docker infrastructure alongside other cloud or on-premises resources in a unified Pulumi workflow. The package abstracts Docker operations, allowing users to declare desired states for Docker resources, which Pulumi then translates into API calls to the Docker daemon. This simplifies tasks like building and pushing Docker images, running containers, and configuring networks, while ensuring consistency and reproducibility across environments. The integration with Pulumi's broader ecosystem also enables combining Docker resources with other infrastructure components, such as cloud services or Kubernetes clusters, in a single deployment. The package is particularly useful for automating Docker workflows, managing containerized applications, and maintaining infrastructure configurations as code. It supports common Docker features and aligns with Pulumi's declarative approach to infrastructure management.",
        "augmented_keywords": "docker enabl infrastructur manag packag pulumi resourc user @pulumi/docker",
        "summary_embedding": np.random.rand(768).astype(np.float32).tolist(),
        "keywords_embedding": np.random.rand(768).astype(np.float32).tolist()
    },
]

def insert_sample_data(table_name="summary_aug_keywords"):
    """插入样例数据"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()


    try:
        # 插入数据
        execute_values(
            cursor,
            f"""
            INSERT INTO {table_name} (package_id, package_name, ecosystem, summary, augmented_keywords, summary_embedding, keywords_embedding)
            VALUES %s
            """,
            [
                (doc["package_id"], doc["package_name"], doc["ecosystem"], doc["summary"], doc["augmented_keywords"],
                 doc["summary_embedding"], doc["keywords_embedding"])
                for doc in sample_documents
            ]
        )
    except psycopg2.errors.UniqueViolation:
        print("psycopg2.errors.UniqueViolation")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"插入了 {len(sample_documents)} 条样例数据")


def hybrid_search(weight_summary_vector, query_summary_vector,
                  weight_keywords_vector, query_keywords_vector,
                  weight_summary_text, query_summary_text,
                  weight_keywords_text, query_keywords_text, top_k=5, table_name="summary_aug_keywords"):
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

    # 执行混合查询，不再需要显式类型转换
    # cursor.execute(f"""
    # SELECT
    #     package_id,
    #     summary,
    #     augmented_keywords,
    #     %s * (1 - (summary_embedding <=> %s))
    #     + %s * (1 - (keywords_embedding <=> %s))
    #     + %s * ts_rank(to_tsvector('english', summary), plainto_tsquery('english', %s))
    #     + %s * ts_rank(to_tsvector('english', augmented_keywords), plainto_tsquery('english', %s))  AS combined_score,
    #     (1 - (summary_embedding <=> %s)) AS summary_embedding_score,
    #     (1 - (keywords_embedding <=> %s)) AS keywords_embedding_score,
    #     ts_rank(to_tsvector('english', summary), plainto_tsquery('english', %s)) as summary_text_score,
    #     ts_rank(to_tsvector('english', augmented_keywords), plainto_tsquery('english', %s)) as keywords_text_score,
    #     summary_embedding,
    #     keywords_embedding,
    # FROM {table_name}
    # ORDER BY combined_score DESC
    # LIMIT %s;
    # """, (weight_summary_vector, query_summary_vector,
    #       weight_keywords_vector, query_keywords_vector,
    #       weight_summary_text, query_summary_text,
    #       weight_keywords_text, query_keywords_text,
    #       query_summary_vector, query_keywords_vector, query_summary_text, query_keywords_text,
    #       top_k))

    cursor.execute(f"""
        SELECT
            package_id,
            summary,
            augmented_keywords,
            {weight_summary_vector} * (1 - (summary_embedding <=> '{query_summary_vector}'))
            + {weight_keywords_vector} * (1 - (keywords_embedding <=> '{query_keywords_vector}'))
            + {weight_summary_text} * ts_rank(to_tsvector('english', summary), plainto_tsquery('english', '{query_summary_text}'))
            + {weight_keywords_text} * ts_rank(to_tsvector('english', augmented_keywords), plainto_tsquery('english', '{query_keywords_text}'))  AS combined_score,
            (1 - (summary_embedding <=> '{query_summary_vector}')) AS summary_embedding_score,
            (1 - (keywords_embedding <=> '{query_keywords_vector}')) AS keywords_embedding_score,
            ts_rank(to_tsvector('english', summary), plainto_tsquery('english', '{query_summary_text}')) as summary_text_score,
            ts_rank(to_tsvector('english', augmented_keywords), plainto_tsquery('english', '{query_keywords_text}')) as keywords_text_score,
            summary_embedding,
            keywords_embedding
        FROM {table_name}
        ORDER BY combined_score DESC
        LIMIT {top_k};
        """)

    results = cursor.fetchall()

    print(f"\n混合检索结果:")
    print("=" * 80)
    for i, row in enumerate(results):
        print(f"\n结果 {i + 1} (综合得分: {row[3]:.4f})")
        print(f"Package ID: {row[0]}")
        print(f"Summary: {row[1][:100]}...")
        print(f"Augmented Keywords: {row[2]}")  # 只显示前100个字符
        print(f"Summary Embedding: {row[8][:5]}...")
        print(f"Keywords Embedding: {row[9][:5]}...")
        print(f"Summary Embedding 得分: {row[4]:.4f}")
        print(f"Keywords Embedding 得分: {row[5]:.4f}")
        print(f"Summary Text 得分: {row[6]:.4f}")
        print(f"Keywords Text 得分: {row[7]:.4f}")
        print(f"综合得分: {row[3]:.4f}")

    cursor.close()
    conn.close()
    return results


if __name__ == "__main__":
    # 初始化数据库
    setup_database()
    insert_sample_data()

    weight_summary_vector = 0.25
    query_summary_vector = sample_documents[2]["summary_embedding"]
    weight_keywords_vector = 0.25
    query_keywords_vector = sample_documents[2]["keywords_embedding"]
    weight_summary_text = 0.25
    query_summary_text = "flatten nested arrays into a single, non-nested list of element"
    weight_keywords_text = 0.25
    query_keywords_text = "array flatten"

    # 执行查询
    hybrid_search(weight_summary_vector, query_summary_vector,
                  weight_keywords_vector, query_keywords_vector,
                  weight_summary_text, query_summary_text,
                  weight_keywords_text, query_keywords_text)

    # 尝试不同的权重组合
    # print("\n尝试不同的权重组合:")
    # hybrid_search(query_text, example_query_vector, vector_weight=0.9, text_weight=0.1)
    # hybrid_search(query_text, example_query_vector, vector_weight=0.1, text_weight=0.9)

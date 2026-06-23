import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

from src.configuration.config import MYSQL_CONFIG, NEO4J_CONFIG


# 读取MySQL工具类
class MySQLReader:
    def __init__(self):
        self.connection = pymysql.connect(**MYSQL_CONFIG)
        #  {'id': 1, 'name': '麦德龙', 'origin': '德国'} 让返回数据更好操作
        self.cursor = self.connection.cursor(DictCursor)

    # 查询mysql，读取数据
    def query(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    # 释放资源
    def close(self):
        self.cursor.close()
        self.connection.close()


# neo4j 写入工具类
class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    # 写入节点 固定同一个标签 批量写入多个节点
    def write_node(self, label: str, properties: [dict]):
        # 写入Neo4j
        cypher = f"""
                UNWIND $properties as item
                MERGE (:{label} {{id: item.id, name: item.name}})
            """
        self.driver.execute_query(cypher, properties=properties)

    # 批量写入关系
    def write_relationship(self, type: str, start_label: str, end_label: str, relationships: [dict]):
        # 写入关系
        cypher = f"""
                        UNWIND $relationships as item
                        MATCH (start:{start_label} {{id: item.start_id}}), (end:{end_label} {{id: item.end_id}})
                        MERGE (start)-[:{type}]->(end)
                    """
        self.driver.execute_query(cypher, relationships=relationships)


if __name__ == '__main__':
    reader = MySQLReader()

    # 1. 读取category1
    sql = """
          SELECT id, name
          FROM gmall.base_category1 \
          """

    category1 = reader.query(sql)
    print(category1)

    # 2. 写入Neo4j
    writer = Neo4jWriter()
    writer.write_node("Category1", category1)

    # 3. 读取category2
    sql = """
          SELECT id, name \
          FROM gmall.base_category2 \
          """
    category2 = reader.query(sql)
    writer.write_node("Category2", category2)

    # 4. 写入创建关系
    sql = """
        select id as start_id, category1_id as end_id
        from gmall.base_category2
    
    """
    relationships = reader.query(sql)

    writer.write_relationship("Belong", "Category2", "Category1", relationships)

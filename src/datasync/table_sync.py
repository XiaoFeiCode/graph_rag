from src.datasync.utils import MySQLReader, Neo4jWriter


# 构建一个表数据的同步器
class TableSync:
    def __init__(self):
        self.reader = MySQLReader()
        self.writer = Neo4jWriter()

    # 1. Category1
    def sync_category1(self):
        sql = """
              SELECT id, name
              FROM base_category1 \
              """
        # 读取数据
        properties = self.reader.query(sql)
        # 写入数据
        self.writer.write_node("Category1", properties)

    # 1. Category2
    def sync_category2(self):
        sql = """
              SELECT id, name
              FROM base_category2 \
              """
        # 读取数据
        properties = self.reader.query(sql)
        # 写入数据
        self.writer.write_node("Category2", properties)

    # 1. Category3
    def sync_category3(self):
        sql = """
              SELECT id, name
              FROM base_category3 \
              """
        properties = self.reader.query(sql)
        self.writer.write_node("Category3", properties)

    # 1. Category2 to Category1
    def sync_category2_to_category1(self):
        sql = """
              SELECT id AS start_id, category1_id as end_id
              FROM base_category2 \
              """
        # 读取数据
        relationship = self.reader.query(sql)
        # 写入数据
        self.writer.write_relationship("Belong", "Category2", "Category1", relationship)

    # 1. Category3 to Category2
    def sync_category3_to_category2(self):
        sql = """
              SELECT id AS start_id, category2_id as end_id
              FROM base_category3 \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Belong", "Category3", "Category2", relationship)

    # 2. 平台属性
    def sync_base_attr_name(self):
        sql = """
              SELECT id, attr_name as name
              FROM base_attr_info
              """
        properties = self.reader.query(sql)
        self.writer.write_node("BaseAttrName", properties)

    # 2. 平台属性值
    def sync_base_attr_value(self):
        sql = """
              SELECT id, value_name as name
              FROM base_attr_value
              """
        properties = self.reader.query(sql)
        self.writer.write_node("BaseAttrValue", properties)

    # 2. 平台属性关系
    def sync_base_attr_name_value(self):
        sql = """
              SELECT id AS end_id, attr_id AS start_id
              FROM base_attr_value
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "BaseAttrName", "BaseAttrValue", relationship)

    # 3. 类别和平台属性关系
    def sync_category1_to_base_attr_name(self):
        # 利用category_level
        sql = """
              SELECT category_id AS start_id, id AS end_id
              FROM base_attr_info
              where category_level = 1
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "Category1", "BaseAttrName", relationship)

    def sync_category2_to_base_attr_name(self):
        # 利用category_level
        sql = """
              SELECT category_id AS start_id, id AS end_id
              FROM base_attr_info
              where category_level = 2
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "Category2", "BaseAttrName", relationship)

    def sync_category3_to_base_attr_name(self):
        # 利用category_level
        sql = """
              SELECT category_id AS start_id, id AS end_id
              FROM base_attr_info
              where category_level = 3
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "Category3", "BaseAttrName", relationship)

    # 4. 商品信息
    def sync_spu(self):
        sql = """
              SELECT id, spu_name as name
              FROM spu_info
              """
        properties = self.reader.query(sql)
        self.writer.write_node("SPU", properties)

    def sync_sku(self):
        sql = """
              SELECT id, sku_name as name
              FROM sku_info
              """
        properties = self.reader.query(sql)
        self.writer.write_node("SKU", properties)

    def sync_sku_to_spu(self):
        sql = """
              SELECT id AS start_id, spu_id AS end_id
              FROM sku_info
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Belong", "SKU", "SPU", relationship)

    # spu到category3
    def sync_spu_to_category3(self):
        sql = """
              SELECT id AS start_id, category3_id AS end_id
              FROM spu_info \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Belong", "SPU", "Category3", relationship)

    # 品牌信息
    def sync_trademark(self):
        sql = """
              SELECT id, tm_name as name
              FROM base_trademark
              """
        properties = self.reader.query(sql)
        self.writer.write_node("Trademark", properties)

    def sync_spu_to_trademark(self):
        sql = """
              SELECT id AS start_id, tm_id AS end_id
              FROM spu_info \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Belong", "SPU", "Trademark", relationship)

    # 销售属性
    def sync_sale_attr(self):
        sql = """
              SELECT id, sale_attr_name as name
              FROM gmall.spu_sale_attr
              """
        properties = self.reader.query(sql)
        self.writer.write_node("SaleAttrName", properties)

    def sync_sale_attr_value(self):
        sql = """
              SELECT id, sale_attr_value_name as name
              FROM spu_sale_attr_value
              """
        properties = self.reader.query(sql)
        self.writer.write_node("SaleAttrValue", properties)

    def sync_sale_attr_name_to_value(self):
        sql = """
              SELECT a.id AS start_id, v.id AS end_id
              FROM gmall.spu_sale_attr a
                       JOIN gmall.spu_sale_attr_value v
                            ON a.spu_id = v.spu_id
                                AND a.base_sale_attr_id = v.base_sale_attr_id
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "SaleAttrName", "SaleAttrValue", relationship)

    def sync_spu_to_sale_attr_name(self):
        sql = """
              select spu_id as start_id, id as end_id
              from gmall.spu_sale_attr \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "SPU", "SaleAttrName", relationship)

    def sync_sku_to_sale_attr_value(self):
        sql = """
              select sku_id as start_id, sale_attr_value_id as end_id
              from gmall.sku_sale_attr_value \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "SKU", "SaleAttrValue", relationship)

    def sync_sku_to_base_attr_value(self):
        sql = """
              select sku_id as start_id, value_id as end_id
              from gmall.sku_attr_value \
              """
        relationship = self.reader.query(sql)
        self.writer.write_relationship("Have", "SKU", "BaseAttrValue", relationship)



_SYNC_STEPS = [
    # 1. 同步类别
    "sync_category1", "sync_category2", "sync_category3",
    "sync_category2_to_category1", "sync_category3_to_category2",
    # 2. 同步平台属性
    "sync_base_attr_name", "sync_base_attr_value", "sync_base_attr_name_value",
    "sync_category1_to_base_attr_name", "sync_category2_to_base_attr_name", "sync_category3_to_base_attr_name",
    # 3. 同步商品信息
    "sync_spu", "sync_sku", "sync_sku_to_spu", "sync_spu_to_category3",
    # 4. 同步品牌信息
    "sync_trademark", "sync_spu_to_trademark",
    # 5. 同步销售属性
    "sync_sale_attr", "sync_sale_attr_value", "sync_sale_attr_name_to_value",
    "sync_spu_to_sale_attr_name", "sync_sku_to_sale_attr_value", "sync_sku_to_base_attr_value",
]

if __name__ == '__main__':
    sync = TableSync()
    for method_name in _SYNC_STEPS:
        getattr(sync, method_name)()
    sync.reader.close()
    sync.writer.close()


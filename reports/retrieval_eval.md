# Retrieval Evaluation — Multi-Method Comparison

**Top-K:** 5

## Summary

| Method | Recall | Precision | MRR |
| --- | --- | --- | --- |
| bm25_fulltext | 12.50% | 5.00% | 0.0833 |
| neo4j_vector | 12.50% (+0.0%) | 5.00% | 0.1250 |
| hybrid_rrf | 12.50% (+0.0%) | 5.00% | 0.1250 |
| hybrid_rrf_rerank | 0.00% (+-100.0%) | 0.00% | 0.0000 |
| milvus_vector | 12.50% (+0.0%) | 5.00% | 0.1250 |

## BM25 Fulltext

| Question | Expected | Top Hits | Recall | Precision | MRR |
| --- | --- | --- | --- | --- | --- |
| 有没有适合拍照的 Apple 手机？ | Apple, iPhone 15 | 手机, 拍照配件, 手机支架, 手机电池, 手机饰品 | 0.00% | 0.00% | 0.0000 |
| 华为 Mate 系列有什么卖点？ | 华为, HUAWEI Mate 60 | 华为Mate 40 pro, 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器, 华为, 华为HUAWEI二手笔记本MateBook13触屏2K全面屏 便携二手笔记本电脑 Magic R5-2500-8G-256G-高分屏 95成新, 华为智慧屏V65i 65英寸 HEGE-560B 4K全面屏智能电视机 多方视频通话 AI升降摄像头 4GB+32GB 星际黑 | 50.00% | 20.00% | 0.3333 |
| 推荐一个适合少油烹饪的厨房电器 | 美的 5L 空气炸锅 | 其它厨房电器, 烹饪美食, 充电器, 厨房DIY/小工具, 电子秤/厨房秤 | 0.00% | 0.00% | 0.0000 |
| iPhone 15 有哪些 SKU？ | iPhone 15 | iPhone 配件, Apple iPhone 12, Apple iPhone 16 Pro, H&U&W国行【2025新英特尔+酷睿i7】笔记本电脑轻薄本大学生办公便携高性能游戏本商务设计AI标压手提本 【15代14英寸】英特尔IPS护眼+9D蓝光全面屏 8G内存/128G超速硬盘, Apple iPhone 12 (A2404) 64GB 蓝色 支持移动联通电信5G 双卡双待手机 | 0.00% | 0.00% | 0.0000 |

## Neo4j Vector

| Question | Expected | Top Hits | Recall | Precision | MRR |
| --- | --- | --- | --- | --- | --- |
| 有没有适合拍照的 Apple 手机？ | Apple, iPhone 15 | Apple iPhone 12, Apple iPhone 16 Pro, 苹果, Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机, Apple iPhone 12 (A2404) 64GB 红色 支持移动联通电信5G 双卡双待手机 | 0.00% | 0.00% | 0.0000 |
| 华为 Mate 系列有什么卖点？ | 华为, HUAWEI Mate 60 | 华为Mate 40 pro, 华为, 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器, 华为HUAWEI二手笔记本MateBook13触屏2K全面屏, 华为HUAWEI二手笔记本MateBook13触屏2K全面屏 便携二手笔记本电脑 Magic R5-2500-8G-256G-高分屏 95成新 | 50.00% | 20.00% | 0.5000 |
| 推荐一个适合少油烹饪的厨房电器 | 美的 5L 空气炸锅 | 厨房小电, 其它厨房电器, 油烟机, 厨具, 电子秤/厨房秤 | 0.00% | 0.00% | 0.0000 |
| iPhone 15 有哪些 SKU？ | iPhone 15 | Apple iPhone 16 Pro, iPhone 配件, Apple/苹果 iPhone 16 Pro（A3294）256GB 黑色钛金属 支持移动联通电信5G 双卡双待手机, 苹果配件, 苹果周边 | 0.00% | 0.00% | 0.0000 |

## RRF (FT + Vector)

| Question | Expected | Top Hits | Recall | Precision | MRR |
| --- | --- | --- | --- | --- | --- |
| 有没有适合拍照的 Apple 手机？ | Apple, iPhone 15 | 手机, 苹果, 拍照配件, 华为, 手机支架 | 0.00% | 0.00% | 0.0000 |
| 华为 Mate 系列有什么卖点？ | 华为, HUAWEI Mate 60 | 华为Mate 40 pro, 华为, 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器, 苹果, 华为 | 50.00% | 20.00% | 0.5000 |
| 推荐一个适合少油烹饪的厨房电器 | 美的 5L 空气炸锅 | 其它厨房电器, 小米, 烹饪美食, 华为, 充电器 | 0.00% | 0.00% | 0.0000 |
| iPhone 15 有哪些 SKU？ | iPhone 15 | iPhone 配件, 苹果, Apple iPhone 12, VIVO, Apple iPhone 16 Pro | 0.00% | 0.00% | 0.0000 |

## RRF + BGE-Reranker

| Question | Expected | Top Hits | Recall | Precision | MRR |
| --- | --- | --- | --- | --- | --- |
| 有没有适合拍照的 Apple 手机？ | Apple, iPhone 15 | Apple iPhone 12 (A2404) 64GB 红色 支持移动联通电信5G 双卡双待手机, Apple iPhone 12 (A2404) 64GB 黑色 支持移动联通电信5G 双卡双待手机, Apple iPhone 12 (A2404) 64GB 蓝色 支持移动联通电信5G 双卡双待手机, Apple iPhone 12 (A2404) 64GB 白色 支持移动联通电信5G 双卡双待手机, Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机 | 0.00% | 0.00% | 0.0000 |
| 华为 Mate 系列有什么卖点？ | 华为, HUAWEI Mate 60 | 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器, 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器, 华为HUAWEI二手笔记本MateBook13触屏2K全面屏 便携二手笔记本电脑 Magic R5-2500-8G-256G-高分屏 95成新, 华为HUAWEI二手笔记本MateBook13触屏2K全面屏 便携二手笔记本电脑 Magic R5-2500-8G-256G-高分屏 95成新, 华为Mate 40 pro | 0.00% | 0.00% | 0.0000 |
| 推荐一个适合少油烹饪的厨房电器 | 美的 5L 空气炸锅 | 油烟机, 其它厨房电器, 其它厨房电器, 电饭煲, 厨房小电 | 0.00% | 0.00% | 0.0000 |
| iPhone 15 有哪些 SKU？ | iPhone 15 | 机身附件, iPhone 配件, iPhone 配件, 苹果周边, 苹果配件 | 0.00% | 0.00% | 0.0000 |

## Milvus Vector

| Question | Expected | Top Hits | Recall | Precision | MRR |
| --- | --- | --- | --- | --- | --- |
| 有没有适合拍照的 Apple 手机？ | Apple, iPhone 15 | Apple iPhone 12, Apple iPhone 16 Pro, 苹果, 苹果, Apple iPhone 12 (A2404) 64GB 红色 支持移动联通电信5G 双卡双待手机 | 0.00% | 0.00% | 0.0000 |
| 华为 Mate 系列有什么卖点？ | 华为, HUAWEI Mate 60 | 华为Mate 40 pro, 华为, 性价比, 颜值, 华为Mate 40 pro 5G全网通麒麟9000芯片鸿蒙国货NFC双卡游戏学生 黑色 8+128 送适用100w充电器 | 50.00% | 20.00% | 0.5000 |
| 推荐一个适合少油烹饪的厨房电器 | 美的 5L 空气炸锅 | 厨房小电, 其它厨房电器, 油烟机, 厨具, 电子秤/厨房秤 | 0.00% | 0.00% | 0.0000 |
| iPhone 15 有哪些 SKU？ | iPhone 15 | Apple iPhone 16 Pro, iPhone 配件, Apple/苹果 iPhone 16 Pro（A3294）256GB 黑色钛金属 支持移动联通电信5G 双卡双待手机, 苹果配件, 苹果周边 | 0.00% | 0.00% | 0.0000 |

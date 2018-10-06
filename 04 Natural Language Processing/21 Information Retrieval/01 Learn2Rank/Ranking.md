#  Ranking 

+ 数据获得：
  + 爬取当前音乐排行榜的数据
  + 当前文本数据（金辉）查看
  + 查找媒资库数据
  + 使用其他Ranking数据检查模型搭建情况
    + https://www.microsoft.com/en-us/research/project/mslr/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fmslr%2F
+ 查询词
  + 有
    + 歌曲
    + 歌手
  + 无
    + 利用其他信息Ranking
+ 特征：（文档转化未特征向量）
  + 查询词在文档中的词频信息
  + 查询词的IDF信息
  + 文档长度
  + 网页的入链数量
  + 网页的出链数量
  + 网页的PageRank值
  + 网页的URL松度
  + Proximity ：即在文档中多大的窗口内可以出现所有査询词
  + 其他特征
    + 播放量
    + 收藏量
    + 评论量
    + 分享量

#  Ranking 

+ 其他数据集
  + [LETOR, ](http://blog.crackcell.com/posts/2011/12/17/a_short_intro_2_ltr.html#sec-7)<http://research.microsoft.com/en-us/um/beijing/projects/letor/>
  + Microsoft Learning to Rank Dataset, <http://research.microsoft.com/en-us/projects/mslr/>
  + Yahoo Learning to Rank Challenge, <http://webscope.sandbox.yahoo.com/>

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
+ 学习分类函数
  + 输入：<文档，标签> 或 <文档，得分>
  + Pointwise
    + 一篇文档：分类，打分
  + Pairwise
    + <Doc1, Doc2> 是否满足顺序关系
    + 二分类判断分类顺序是否正确
    + 缺点：
      + 只考虑了两个文档对的先后顺序，却没有考虑文档在搜索列表中的位置（？？？）
      + 不同的查询，转化后的文档对数量不同
    + 实现：
      + SVM Rank
      + RankNet
      + FRank
      + RankBoost
      + LambdaRank
  + Listwise
    + 每一个查询结果的返回的列表作为训练实例
    + <query, list>
    + KL 散度

+ 效果评估
  + NDGG(Normalized Disconted Cumulative Gain)
  + MAP(Mean Average Precision)
+ 工具
  + SVM Ranking : https://www.microsoft.com/en-us/research/project/mslr/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fmslr%2F
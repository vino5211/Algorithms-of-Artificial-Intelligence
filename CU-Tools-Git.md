+ 版本恢复
	+ git reset --hard commitid  //本地代码回到指定的commitid
	+ git push -f origin branchname//git服务器代码回到指定的commitid
	+ git reflog命令可以对git误操作进行数据恢复。
	如不小心用git commit --amend当成git commit覆盖当前的commit，或不小心把当前的commit给搞没了（reset --hard）。 都可以通过git reflog恢复。
	Git记录每次修改HEAD的操作，git reflog/git log -g可以查看所有的历史操作记录，然后通过git reset命令进行恢复。
	+ example
	```
	apollo@Mars:~/craft/projects/QA-Craft$ git reflog 
	ab284bb HEAD@{0}: reset: moving to HEAD^
	2cdfce4 HEAD@{1}: reset: moving to 2cdfce4
	564d350 HEAD@{2}: reset: moving to 564d350
	8149c69 HEAD@{3}: reset: moving to 8149c69
	7339ef0 HEAD@{4}: reset: moving to 7339ef0
	da52f90 HEAD@{5}: reset: moving to da52f90
	ad62b1a HEAD@{6}: reset: moving to ad62b1a
	c3a7e85 HEAD@{7}: reset: moving to c3a7e85
	317c4c8 HEAD@{8}: reset: moving to 317c4c8f57cbdd9f0e56f6fb52b6f945162f11c1
	c3a7e85 HEAD@{9}: commit (amend): complete data preprocess
	ad62b1a HEAD@{10}: commit: complete data preprocess
	317c4c8 HEAD@{11}: reset: moving to 317c4c8f57cbdd9f0e56f6fb52b6f945162f11c1
	ab284bb HEAD@{12}: reset: moving to ab284bbf13526ebebaa4e6ac9af3df8046bcb651
	da52f90 HEAD@{13}: commit: complete data preprocess
	7ae4bca HEAD@{14}: reset: moving to 7ae4bca2a9f157aa1154026a01fcdbfb122d018d
	8ff5db7 HEAD@{15}: commit (amend): update corpus
	987073a HEAD@{16}: commit (amend): update corpus
	8b43b26 HEAD@{17}: commit (amend): update corpus
	0e142ce HEAD@{18}: commit (amend): update corpus
	7339ef0 HEAD@{19}: commit: update corpus
	8149c69 HEAD@{20}: commit (amend): update for data preprocess
	6587c3a HEAD@{21}: commit (amend): update for data preprocess
	e635bb1 HEAD@{22}: commit (amend): update for data preprocess
	5d40164 HEAD@{23}: commit: update for data preprocess
	564d350 HEAD@{24}: commit: complete preprocess data for reader
	7ae4bca HEAD@{25}: commit: update data/glove
	7dba404 HEAD@{26}: commit: prepare reader
	2cdfce4 HEAD@{27}: commit: complete SQuAD data preprocess
	ab284bb HEAD@{28}: commit: update
	c8d3fdc HEAD@{29}: pull: Fast-forward
	78411cc HEAD@{30}: pull: Fast-forward
	bde2501 HEAD@{31}: commit: add SQuAD and update its README.md
	317c4c8 HEAD@{32}: commit (initial): first commit
	
	apollo@Mars:~/craft/projects/QA-Craft$ git reset --hard ad62b1a
	HEAD is now at ad62b1a complete data preprocess
	apollo@Mars:~/craft/projects/QA-Craft$ 
	apollo@Mars:~/craft/projects/QA-Craft$ git status
	On branch master
	Your branch and 'origin/master' have diverged,
	and have 1 and 6 different commits each, respectively.
	  (use "git pull" to merge the remote branch into yours)
	nothing to commit, working directory clean
	apollo@Mars:~/craft/projects/QA-Craft$ ll
	total 36
	drwxrwxr-x 7 apollo apollo 4096 3月   9 14:52 ./
	drwxrwxr-x 9 apollo apollo 4096 3月   9 14:22 ../
	drwxrwxr-x 3 apollo apollo 4096 3月   9 14:52 craft/
	drwxrwxr-x 4 apollo apollo 4096 3月   9 14:52 drqa/
	drwxrwxr-x 8 apollo apollo 4096 3月   9 14:52 .git/
	-rwxrwxr-x 1 apollo apollo   16 3月   9 14:52 .gitignore*
	drwxrwxr-x 2 apollo apollo 4096 3月   8 18:46 .idea/
	-rw-rw-r-- 1 apollo apollo  740 3月   9 14:52 README.md
	drwxrwxr-x 3 apollo apollo 4096 3月   9 14:52 script/

	```
	
+ 清理缓存
	+ apollo@Mars:~/craft/projects/QA-Craft$ git clean -df
	Removing .idea/
	Removing drqa/__pycache__/
	Removing drqa/reader/__pycache__/
	Removing drqa/tokenizers/__pycache__/

+ git rm 
  + git rm  bnlp-zh-hk-trainer/resources/ -rf
  + 当我们需要删除暂存区或分支上的文件, 同时工作区也不需要这个文件了, 可以使用
      git rm file_path
      git commit -m 'delete somefile'
      git push
  + 当我们需要删除暂存区或分支上的文件, 但本地又需要使用, 只是不希望这个文件被版本控制, 可以使用
      git rm --cached file_path
      git commit -m 'delete remote somefile'
      git push
  
+ gitignore
  ```
  在项目工程中，常会生成一些运行缓存，而这些是不能 push 的，所以需要 .gitignore 配置规则来过滤。

配置规则：

以 # 开头行为注释  

以斜杠“/”开头表示目录；

　　/target/ 过滤根目录的 target 文件夹

　　target/ 过滤任何目录包含的 target 文件夹

以星号“*”通配多个字符；

　　*.zip 过滤所有.zip文件

以问号“?”通配单个字符；

 

以方括号“[]”包含单个字符的匹配列表；

　　

以叹号“!”表示不忽略(跟踪)匹配到的文件或目录；

　　/doc/

　　!/doc/common.doc

此外，git 对于 .ignore 配置文件是按行从上到下进行规则匹配的，意味着如果前面的规则匹配的范围更大，则后面的规则将不会生效；

被过滤掉的文件就不会出现在你的GitHub库中了，当然本地中还有，只是push的时候不会上传。

 

如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。那么解决方法就是先把本地缓存删除（改变成未track状态），然后再提交：

    git rm -r --cached .
    git add .
    git commit -m ‘update .gitignore‘

---

    经测试发现，若要忽略一个文件夹下的部分文件夹，应该一个一个的标示。可能有更好的方法。
    若test下有多个文件和文件夹。若要ignore某些文件夹，应该这个配置.gitignore文件。若test下有test1，test2,test3文件。要track test3，则.gitignore文件为：
    test/test1
    test/test2
    !test/test3
    若为：
    test/
    !test/test3 ，则不能track test3。
    Git 中的文件忽略
    1. 共享式忽略新建 .gitignore 文件，放在工程目录任意位置即可。.gitignore 文件可以忽略自己。忽略的文件，只针对未跟踪文件有效，对已加入版本库的文件无效。
    2. 独享式忽略针对具体版本库 ：.git/info/exclude针对本地全局：  git config --global core.excludefile ~/.gitignore
    忽略的语法规则：
    (#)表示注释
    (*)  表示任意多个字符; 
    (?) 代表一个字符;
     ([abc]) 代表可选字符范围
    如果名称最前面是路径分隔符 (/) ，表示忽略的该文件在此目录下。
    如果名称的最后面是 (/) ，表示忽略整个目录，但同名文件不忽略。
    通过在名称前面加 (!) ，代表不忽略。
    例子如下：
    # 这行是注释
    *.a                   # 忽略所有 .a 伟扩展名的文件
    !lib.a                # 但是 lib.a 不忽略，即时之前设置了忽略所有的 .a
    /TODO            # 只忽略此目录下 TODO 文件，子目录的 TODO 不忽略 
    build/               # 忽略所有的 build/ 目录下文件
    doc/*.txt           # 忽略如 doc/notes.txt, 但是不忽略如 doc/server/arch.txt 
  ```

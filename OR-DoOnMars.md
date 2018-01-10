Do on Mars 

Make this to a scripts

+ CUDA install
	+ Three Monitor Displays
	```
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
 	sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb
 	sudo apt-get update
 	sudo apt-get install cuda
    ```
+ (Undo) cudnn
+ Andconda3
+ Haroopad
	+ http://www.bubuko.com/infodetail-2013827.html
	+ 在线Markdown 编辑器, 可用于测试markdown 格式（列表，图片等）的正确写法
		+ https://jbt.github.io/markdown-editor/
	+ markdown 常见数学符号的写法
		+ http://mohu.org/info/symbols/symbols.htm
	+ 在线Markdown 公式编辑器，可用于产生正确的 markdown 公式（操作类似 Word 的公式编辑器），复制其结果到Markdown中即可，而不用记忆Markdown公式编辑器的语法
		+ http://latex.codecogs.com/eqneditor/editor.php
+ pycharm
	+ tar -zxvf pycharm-communi
	+ pycharm settings
		+ File-Settings-Project:Name-Project Interpreter-Select Python 3
		+ 改变编辑器颜色
			+ Settngs-Editor-Colors&Fonts-Change Scheme to Darcula 
+ Shackshaows
	+ http://blog.csdn.net/sinat_32292481/article/details/78597067 
	+ sslocal -c /home/apollo/settings/shadowsocks/config.json &
+ terminator
	+ sudo apt-get install terminator
	+ http://blog.csdn.net/loveaborn/article/details/21511869
	+ search it in Left top
+ Add Disks && Mount
	+ Add Disk : https://jingyan.baidu.com/article/2f9b480d5c67dd41cb6cc2ef.html
	+ Mount : http://winhyt.iteye.com/blog/980749
		+ sudo vim /etc/fstab
		+ /dev/sdb /craft ext4 defaults 0 0
		```
        第一列为设备号或该设备的卷标
        第二列为挂载点
        第三列为文件系统
        第四列为文件系统参数
        第五列为是否可以用demp命令备份。0：不备份，1：备份，2：备份，但比1重要性小。设置了该参数后，Linux中使用dump命令备份系统的时候就可以备份相应设置的挂载点了。
        第六列为是否在系统启动的时候，用fsck检验分区。因为有些挂载点是不需要检验的，比如：虚拟内存swap、/proc等。0：不检验，1：要检验，2要检验，但比1晚检验，一般根目录设置为1，其他设置为2就可以了。 
        ```
    + sudo chown -R  apollo:apollo craft/
    + suod chmod 775 craft -R
+ 网易云音乐
	+ http://music.163.com/#/download
	+ sudo dpkg -i netease-cloud-music_1.0.0_amd64_ubuntu16.04.deb Or Install in Ubuntu Software Center
	+ weibo login : 7241070@qq.com  925199...sina

+ Chrome
	+ https://jingyan.baidu.com/article/335530da98061b19cb41c31d.html 
+ kanbanflow
	+  gmail kanban123456
+ Ubuntu 16.04 安装 google 输入法
	+ http://blog.csdn.net/striker_v/article/details/51914637
		+ 配置输入法需要重启 
	+ 切换输入法 Ctrl + Shift 或者 Shift
---
+ 在线UML作图
	+ 介绍：http://www.heqiangfly.com/2017/07/08/development-tool-markdown-plant-uml/ 
	+ 网址：http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000
+ 在线作图
	+ Draw Diagrams With Markdown
		+ http://support.typora.io/Draw-Diagrams-With-Markdown/
	+ js-sequence-diagrams 
		+ https://bramp.github.io/js-sequence-diagram
+ Markdown UML 作图

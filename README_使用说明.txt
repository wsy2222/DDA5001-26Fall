1. 把 run_defaults.sh 和 run_alternatives.sh 复制到机房 ~/work/project1。
2. 在机房 Ubuntu 终端执行 cd ~/work/project1 ; bash run_defaults.sh。
3. 等前三组跑完，查看 runs/p1_comparison/ 下的曲线。
4. 在 VS Code 打开 run_alternatives.sh，将顶部三个 CHANGE_ME 改为自己选的正数学习率，保存。
5. 执行 bash run_alternatives.sh，后面三组会依次自动训练，生成比较图、文本样例和 ZIP。
6. 重启会清空本地文件，离开前必须分别备份运行结果、ZIP 和 checkpoint。

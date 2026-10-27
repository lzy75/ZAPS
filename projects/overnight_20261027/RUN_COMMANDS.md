# 上午运行命令：恢复点

A56—A60五包均已实现并推送。10/27中午更新：A57—A60摘要已在GitHub核验；A56未完成。A56选样修订（f1ecf49）已通过服务器19项检查并冻结20例；随后12GiB显存准入未过、尚未启动分片。当前入口648030e按数据集拆为四个工作组，沿用该冻结计划，双卡最多两组并行；A57保留已完成的算子接口修复入口，旧失败状态保留。历史夜间PDF不改写，最新解读见projects/reports/20261027_a57_a60_readout/README.md。
下列五个完整代码块已填写各包真实最低SHA；本轮逐一核对入口/祖先提交和shell语法，不改各包冻结源文件。

## 1. A56：固定规则的新图/双数据集确认（代码已推）

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh &&
conda activate base &&
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor 648030e HEAD &&
cd projects &&
python -u utils/a56_dataset_queue_session.py
```

A56尚未完成。9624前向/最多1204输入反向，6FFHQ+4ImageNet×SR/IP。
四组为FFHQ各6例、ImageNet各4例。FFHQ准入6GiB（工程门槛，新增峰值记录），ImageNet仍12GiB；每张GPU一个工作进程，最多两组并行。显存不足等待；原等待器只有确认零案例attempt且waiting_for_gpu时才接管，已开跑则继续跟随。原20例/9624前向不变，不重选图、不重跑失败案例。断线同入口重连。

## 2. A57：独立无学习30调用与原U等价（代码已推）

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh &&
conda activate base &&
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor cb449fd HEAD &&
cd projects &&
python -u utils/a57_operator_repair_session.py
```

依赖已有A55r2十例完成，不依赖尚未运行的A56。300新方法前向、≤300输入VJP，
0优化更新；10旧例端点/创新配对已通过，最大相对误差2.2951e-7；无新图或10倍墙钟保证。
A56与A57建议顺序运行，避免共享GPU挤占；断线同入口重连，不重做已完成case。

## 3. A58：blur原目标预条件准入（代码已推）

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh &&
conda activate base &&
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor 309e291 HEAD &&
cd projects &&
python -u utils/a58_session.py
```

不依赖A56/A57新结果。先24项0NN数值计量；8项PCG全部过真实残差1e-5才自动跑质量。
若有失败，自动停止NN并归档数值否定，不要通过增加迭代/改eta或阈值重试。
条件质量最多1200新诊断前向、600输入VJP，旧5图×两blur，C/原CG-U/PCG-U/PCG-V。
使用原零边界目标、原核系数提升64bit；周期FFT只作预条件器，成本单列。

## 4. A59：同t方向场几何与终局剂量响应（代码已推）

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh &&
conda activate base &&
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor fa2d7fa HEAD &&
cd projects &&
python -u utils/a59_session.py
```

依赖已完成A55r2，不依赖A56—A58新输出。上限760新诊断前向/760输入VJP；
回放300＋同t双尺度探针120＋γ=.5/1.5单步干预续推340，固定原30点。
全部终点冻结后才评价，比较阶段/物理R与g/新增局部几何的信息；不部署控制器或承诺曲率增益。
建议与A56等大任务顺序运行；失败记录保留，无自动预算重置重跑。

## 5. A60：方向每1/2/4步刷新（代码已推）

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh &&
conda activate base &&
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor 3e855f2 HEAD &&
cd projects &&
python -u utils/a60_session.py
```

900新诊断前向，最多550输入VJP；每步仍计算当前v和K||v||，只滞后u方向。
首末必须刷新。CURRENT同场重跑核对缓存与计时；滞后步不偷偷算当前u。
它是方向复用成本诊断，前向仍30点，不能称采样步数减少或精确冻结Jacobian。

## 顺序与断线恢复

科学优先级A56→A57→A58→A59→A60。A57也可作为300调用的独立短门控先做。
各包不依赖其他新包的运行结果；A58在自身0NN准入后条件触发质量。
建议顺序执行，避免所有launcher同时选择相同空闲GPU导致挤占。

首次完成任一同步后，也可用 `bash /home/lzy/ZAPS/projects/overnight_20261027/run_commands.sh A56`，
将末尾编号换为A57/A58/A59/A60即可；脚本不会默认把五包全部启动。

断线后进入 `/home/lzy/ZAPS/projects`，激活base后再运行对应 `python -u utils/a60_session.py` 即可跟随，
编号按任务替换。加 `--status` 只查看状态/最后日志，不重启GPU。失败记录不自动重跑，先粘贴失败原因。

全队列最多12784次新UNet前向、3414输入VJP及A58的24算子求解；A58未准入则不消耗其质量预算。
所有PT只在/data0，轻量结果自动归档到独立分支，push失败保留结果，不重新算模型。

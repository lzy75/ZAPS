# 上午运行命令：恢复点

当前A56—A59四包已实现并推送。A60未写完前不填假命令。
最终版本需4—5个完整代码块，最低SHA在全部入口推送后填写，不改各包冻结源文件。

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
git merge-base --is-ancestor e60df03 HEAD &&
cd projects &&
python -u utils/a56_session.py
```

实际服务器Torch/GPU尚未报告。9624前向/最多1204输入反向，6FFHQ+4ImageNet×SR/IP。
每GPU至少12GiB空闲，只有一张符合时自动串行。断线重连同入口跟随；不重复启动。

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
git merge-base --is-ancestor e281433 HEAD &&
cd projects &&
python -u utils/a57_session.py
```

依赖已有A55r2十例完成，不依赖尚未运行的A56。300新方法前向、≤300输入VJP，
0优化更新；10旧例端点/创新配对和真实成本核对，无新图或10倍墙钟保证。
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

## 后续包状态

A60方向刷新：待实现/独立卡/检查/推送。
这些不是可执行命令；完成后更新本文件并生成run_commands.sh，依赖与准入失败分支明确。

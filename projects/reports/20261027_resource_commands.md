# 服务器资源核对与可并行队列

先核对进程归属，不终止进程。GPU1快照可容纳ImageNet原生版本，但实时准入仍须通过。

## 1. 精确查看三个占卡PID（只读）

```bash
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
ps -ww -p 1081489,378093,407232 -o user:16,uid,pid,ppid,etime,stat,%cpu,%mem,args
for pid in 1081489 378093 407232; do
  printf '\nPID %s cwd: ' "$pid"
  readlink "/proc/$pid/cwd" 2>/dev/null || true
done
```

PID退出或无权限会读不到cwd，不据此杀进程。不要贴账号/token/订阅信息；不需要环境变量清单。

## 2. 终端一：同步、固定GPU1续跑ImageNet

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy git -c http.proxy=http://127.0.0.1:10809 fetch https://github.com/lzy75/ZAPS.git refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
cd projects &&
/home/lzy/anaconda3/bin/python -m unittest utils.test_a56m_no_interrupt_resume -v &&
/home/lzy/anaconda3/bin/python -u utils/a56m_no_interrupt_resume.py --gpu GPU-e21d8c3c-cde6-c10f-878d-b20a4ded1742
```

旧队列活动时拒绝，不发送任何信号。原M1已活动则跟随；适配器已活动则跟随自己的日志。
完整3例/四条基线/72-call profile复用。快照后资源若减少则原8.652GiB门控继续阻止新采样。
原科学失败或partial不自动重试。30秒进度；断线后同命令重连。

## 3. 终端二：等终端一完成Git同步后，两个blur在CPU并行资格检查

```bash
cd /home/lzy/ZAPS/projects &&
/home/lzy/anaconda3/bin/python -u utils/a61_blur_cpu_session.py
```

不需再次同步Git；终端一正在做科学SHA核对期间不要另一次修改checkout。
CPU各2线程、CUDA隐藏；method/diagnostic NN均0。每任务4个求解，总计8个。
新条件预算可能比旧128迭代更贵，全部算子/FFT/时间收费；真实残差1e-5不变。
这是独立数值资格，不自动启动重建，不等于已有Gaussian/Motion质量结果。
结束自动集中summary/归档/push；进度每30秒输出，详细每solve日志在独立目录内。

## 4. 只看状态

```bash
/home/lzy/anaconda3/bin/python -u /home/lzy/ZAPS/projects/utils/a56m_no_interrupt_resume.py --status
/home/lzy/anaconda3/bin/python -u /home/lzy/ZAPS/projects/utils/a61_blur_cpu_session.py --status
```

没有新增结果就不重跑已完成A57—A60、FFHQ早读或旧数值失败；不要手工清锁/删除失败记录。

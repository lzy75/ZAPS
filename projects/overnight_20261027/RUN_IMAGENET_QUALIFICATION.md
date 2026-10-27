# ImageNet新入口：有上限计量后自动续跑

前提：已完成A56 FFHQ12例。原空闲等待队列会经过身份/子进程检查再接管；若原ImageNet已开算则跟随现任务。
原FP32256模型、30点、4图×2任务不改。计量最多72F/32输入VJP；通过后自动跑原8例，新增总上限3924F/514输入VJP。
只有8GiB才能开始一次计量，质量准入仍需实测峰值+1GiB空间和所有数值门控，不能把8GiB称已认证需求。

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor e7c5749 HEAD &&
cd projects &&
/home/lzy/anaconda3/bin/python -u utils/a56m_imagenet_session.py
```

状态：`/home/lzy/anaconda3/bin/python -u /home/lzy/ZAPS/projects/utils/a56m_imagenet_session.py --status`。
每30秒进度；断线同入口重连。计量OOM/门控失败会停止并归档；不会无限等或自动重撞OOM。
本机没有Torch/CUDA，GPU资格未确认；必须等服务器输出。

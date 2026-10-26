# 服务器系统盘满：先迁移VS Code，再恢复实验

用户回传：/dev/sdb2 879G，可用0，inode仅13%；/data0有12T可用。
用户目录中pip缓存13M、.vscode-server 6.4G、ZAPS/.git 224M。
这是存储容量问题，不是RAM/显存耗尽；当前直连shell未激活Conda，不能依赖python别名。

用户已授权清理和隔离历史内容。采用的具体范围：删除可重建的pip下载缓存以腾出少量元数据空间；
停止当前用户的VS Code远端进程；把完整.vscode-server复制到/data0，逐文件SHA/类型/权限
和源不变检查通过后，原路径切换软链接，只移除系统盘上已核验的重复副本。
原安装、扩展和备份保留在数据盘。ZAPS代码、Git对象、实验PT/模型/数据不在此次清理范围。
过去结果归档明确排除了二进制张量，不能仅凭GitHub有摘要就删除服务器原始证据。

## 独立脚本与离线交付

`utils/relocate_vscode_storage.py`仅用Python3标准库，无Git/网络/Torch依赖，0 UNet。
源目录固定/home/lzy/.vscode-server，目标是/data0/lzy/vscode-server-时间戳-PID。
复制验证不通过不会切换/删除原目录；软链接建立失败会恢复原路径。
已有目标不覆盖；检测到该用户VS Code进程仍运行则停止。迁移清单存目标同级migration.json。

离线粘贴文件：
`/Users/liziyue.22/ZAPS-deliverables/PASTE_SERVER_VSCODE_STORAGE_RECOVERY_20261026.txt`。
它携带同一源码的gzip/base64副本，先校验SHA256再在/data0落盘并用python3运行；
不是新的科学实验或未说明的磁盘扫描/清理。源码SHA见离线文件中的assert。

1. 关闭Windows上连到该服务器的VS Code窗口，保留独立PowerShell SSH终端。
2. 在服务器清除pip缓存，停止该用户VS Code进程，等待退出：

```bash
rm -rf -- /home/lzy/.cache/pip
pkill -TERM -u lzy -f '/home/lzy/\.vscode-server/' || true
sleep 2
```

3. 整块执行离线迁移命令。需看到1/4至4/4和MIGRATION_PASS；不要中断复制/校验。
4. 核验系统盘可用空间及软链接：

```bash
df -hT / /home/lzy/.vscode-server /data0
ls -ld /home/lzy/.vscode-server
```

5. 成功后从Windows重新连接VS Code，恢复其正常RemoteForward10809代理连接。
在新的服务器终端用以下方式恢复实验Python，再取RUN_A50R2_SERVER命令。

```bash
source /home/lzy/anaconda3/etc/profile.d/conda.sh
conda activate base
python --version
```

本机4项真实临时目录检查通过：复制/软链接/幂等、内容变化保留源、link失败回滚、已有目标不覆盖。
服务器迁移尚未执行，不保证释放量恰等于6.4G（共享盘其他写入/硬链接/打开文件会影响显示）。
如果出错，保留错误和已复制目录；不盲删残留副本，不杀其他用户进程或所有python。

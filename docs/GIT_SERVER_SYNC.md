# Git 代码同步与服务器实验流程

## 基本约定

- 本地是唯一代码修改源，服务器不直接编辑代码。
- Git 只同步代码、配置、脚本和文档。
- 数据集、模型权重和实验结果不通过代码分支同步。
- 服务器按完整 commit 检出代码，并在已跟踪文件干净时运行实验。

## 首次配置

创建私有远程仓库后，在本地执行：

```powershell
git remote add origin <REMOTE_URL>
git push -u origin master
```

服务器首次部署：

```bash
git clone <REMOTE_URL> ZAPS
cd ZAPS
```

数据集和模型权重单独放回约定目录。

## 本地开发与发布

使用功能分支修改和验证：

```powershell
git switch -c codex/<topic>
git diff --check
python -m py_compile projects/configs/config.py projects/modules/*.py projects/utils/*.py
python projects/modules/main_batch.py --help
```

明确暂存代码文件并提交。合并至 `master` 后发布：

```powershell
git switch master
git merge --ff-only codex/<topic>
powershell -ExecutionPolicy Bypass -File scripts/publish_code.ps1
```

发布脚本拒绝已跟踪文件存在未提交修改的情况，推送后会输出服务器要检出的完整 commit。

## 服务器检出指定版本

```bash
cd /path/to/ZAPS
bash scripts/server_checkout.sh <FULL_COMMIT>
```

脚本会先拒绝服务器上的已跟踪文件修改，然后获取远程版本并以 detached HEAD 检出指定 commit。

## 服务器验证

先做静态检查和任务预览：

```bash
python -m py_compile projects/configs/config.py projects/modules/*.py projects/utils/*.py
python projects/modules/main_batch.py --data_dir datasets --max_images 1 --tasks all
```

再运行一张图的四任务 smoke test：

```bash
python projects/modules/main_batch.py \
  --data_dir datasets \
  --max_images 1 \
  --tasks all \
  --device cuda \
  --run
```

确认全部成功，并检查新实验 `run.json` 中 `git.commit` 不是 `nogit`、`git.dirty` 为 `false` 后，才能扩大批量：

```bash
python projects/modules/main_batch.py \
  --data_dir datasets \
  --start_index 0 \
  --max_images 100 \
  --tasks all \
  --device cuda \
  --quiet \
  --run
```

## 结果回传与回滚

结果从服务器单向拉回本地：

```bash
rsync -av --ignore-existing \
  <SERVER>:/path/to/ZAPS/projects/experiments/ \
  projects/experiments/
```

新版本验证失败时直接检出上一个已验证 commit：

```bash
bash scripts/server_checkout.sh <LAST_GOOD_COMMIT>
```

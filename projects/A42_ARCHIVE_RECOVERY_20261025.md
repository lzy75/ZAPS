# A42归档恢复：不重跑模型

用户报告git add退出128，失败位置在全部计算/绘图之后的结果Git归档。
128本身不能区分空间、过滤器、权限等原因，不据此宣称某一根因已确定。
原完整结果保留在 /data0/lzy/ZAPS-results/diag_uncertainty_feedback/A42_20261025。

新入口 `bash run_a42_archive_recovery_20261025.sh`：

1. 使用原launcher锁，核验4个completed case和签名/张量SHA，先重新打印已有summary。
2. 复用已导出的bundle，逐文件检查manifest字节/哈希，并核对完整源目录指纹。
3. 用hash-object --no-filters和临时index的update-index，绕开git add及clean/LFS过滤器。
   从原提交树开始，只添加归档路径；逐blob复核SHA后用compare-and-swap更新results分支。
   不切换checkout，不碰主index，不删现有锁或用户文件，不改实验指标/原源码/plan哈希。
4. 临时索引在/data0。完整捕获Git stderr，不再只输出CalledProcessError退出码。
5. 仅明确No space left on device时，才尝试将新增结果Git对象写入
   /data0/lzy/ZAPS-results/diag_uncertainty_feedback/git_objects，并登记Git alternate。
   原对象不搬不删，已有alternates保留；此目录需要长期保留以供Git读取/推送。
   如果根盘连很小的refs/alternate元数据也无法写入，则保留结果并报告实际错误，不自动清理磁盘。

不加载模型，不调用UNet，不重渲染、不重新优化。新增实验NFE=0。
实际已完成的科学实验是否正向，仍取决于summary内容，归档成功不代表科学假设通过。

回归检查在临时真实Git仓库执行：故障clean filter绕过、Git blob原始字节校验、
已有staged/unstaged工作和主index不变、重复提交复用、外部对象正常Git可读、
空间错误只按明确消息识别，不猜测权限/锁错误。

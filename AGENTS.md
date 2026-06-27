# AGENTS.md

## 项目概述

本项目重点在基于论文《Zero-Shot Adaptation for Approximate Posterior Sampling of Diffusion Models in Inverse Problems》（缩写 ZAPS）进行实验代码复现。

## 规则

1. 避免过度读取上下文（包含数据集、模型参数、日志、CSV），注意 token 消耗
2. 精简的模块化代码，便于复用和修改，减少非必要嵌套和重复
3. 只在关键位置做提示性注释并使用中文，变量名遵循原文
4. 做好环境判断、缓存管理和张量维度对齐
5. 第一性原理思考，分析和解决问题优先，先构思后执行再复盘
6. 结构性大批量修改和编写需用户先审批再执行

   <br />


# methods
# 第一步：Empirical Roofline Toolkit 

文献paper reference\Hierarchical Roofline analysis\Hierarchical Roofline analysis.pdf中提到ERT

The Empirical Roofline Toolkit (ERT) [12] was developed
to characterize multicore, manycore, and GPU-accelerated
systems. It was written in MPI+OpenMP+CUDA in order to
replicate the most common programming environments on
DOE (Department of Energy) supercomputers. 
ERT（Empirical Roofline Toolkit） 是一个基于 MPI、OpenMP 和 CUDA 的开源实测基准工具，用于测量多核、多核加速器及 GPU 系统的实际计算性能与内存带宽，为构建 Roofline 模型提供硬件性能上限。

代码仓库为：https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/
【该 Bitbucket 仓库为伯克利实验室的 cs‑roofline‑toolkit（ERT 实证屋顶线工具）源码仓库，基于 MPI+OpenMP+CUDA 实现微基准测试，用于实测多核、众核及 GPU 加速系统的算力与各级内存带宽，输出屋顶线模型所需硬件表征参数。】

下载方法 git clone https://bitbucket.org/berkeleylab/cs-roofline-toolkit.git
已经下载到了Hierachy-Roofline-Model\cs-roofline-toolkit中

@todo agent你需要将这个代码文件运行起来，获得Nvidia A6000GPU的roofline模型，并将运行的脚本写入Hierachy-Roofline-Model\run_ERT.md中，比如说我在A6000的环境上跑，应该能得到A6000的硬件参数，将硬件参数保存在resluts\A6000.md中，如果型号是H100，则保存到resluts\H100.md中，以此类推

# 第二步：算子测试

https://github.com/z11s11y11/FP16_GEMM_TensorCore.git这是需要测试的GPU 矩阵乘法算子代码

已经下载到了文件夹中，目录为Hierachy-Roofline-Model\FP16_GEMM_TensorCore

@todo agent你需要将这个代码文件运行起来，获得文件夹里面十个算子的不同层级的计算访存比，并将运行的脚本写入Hierachy-Roofline-Model\run_operator.md中。结合第一步，应该能得到这10个矩阵乘法算子在L1,L2和HBM的计算访存比以及算力，将得到的参数保存在resluts文件夹中

# 第三步：画图
参考Example Scripts for Plotting Roofline. [Online]. Available:
https://github.com/cyanguwa/nersc-roofline 里面的画图方法，相关代码已经下载到本地，目录为Hierachy-Roofline-Model\nersc-roofline，可以修改一些代码

@todo agent你需要稍微修改一下这个代码，画出roofline model的图片，将第二步和第一步得到的结果都画出来，图片保存在Hierachy-Roofline-Model\resluts中

# 提醒
如果是第一次下载git，需要用Remove-Item -Recurse -Force将git删除，再上传github
Remove-Item -Recurse -Force FP16_GEMM_TensorCore\.git, cs-roofline-toolkit\.git, nersc-roofline\.git
git add -A
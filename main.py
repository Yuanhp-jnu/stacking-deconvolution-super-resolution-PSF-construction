from matplotlib import pyplot as plt
from stack_deconvolution_PSF import StackDeconvolutionPSF

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 创建堆叠-去卷积对象（已经执行堆叠操作）
    ibp = StackDeconvolutionPSF('../../Data/LORRI/FIT/lor_0019314810_0x630_sci.fit',
                                '../../Data/LORRI/QMPF/lor_0019314810_0x630_sci.QMPF',
                                10, 4, 3, 51, 25, 0)
    # IBP去卷积
    f_rld, Ps = ibp.iterative_back_project_deconvolution()
    plt.title(r'$F_{RLD}')
    plt.imshow(f_rld)
    plt.show()

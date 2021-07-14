#coding=utf8
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Line


def logistic_plot(loss_list, acc_list,epoch,mode):
    x = [i+1 for i in range(epoch+1)]
    title = "Logistic Regression "+mode+" Accuracy and Loss Curves"
    l =[]
    a = []
    for i in loss_list:
        l.append(round(i,2))
    for i in acc_list:
        a.append(round(i,2))

    line1 = (
        Line()
            .add_xaxis(x)
            .add_yaxis("Accuracy", a)  # 图例
            .extend_axis(
            yaxis=opts.AxisOpts(
                name='Loss',
            ),
            xaxis=opts.AxisOpts(
                name=' Epoch'
            )
        )
            .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
              markpoint_opts=opts.MarkPointOpts(
              data=[opts.MarkPointItem(type_="max", name="最大值")]
                )
            )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
        )
    )

    line2 = (
        Line()
             .add_xaxis(x)
             .add_yaxis("Loss", l,yaxis_index=1)
             .extend_axis(
            yaxis=opts.AxisOpts(
                name='Loss',
            )

        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),  # 是否在曲线上方标数值
        )
             )
    line1.overlap(line2)
    line1.render("Logistic Regression "+mode+".html")

def softmax_plot(acc_list, epoch, mode):
    x = [i+1 for i in range(epoch+1)]
    title ="Softmax Regression "+mode+" Accuracy Curve"
    a = []

    for i in acc_list:
        a.append(round(i,2))

    line = (
        Line()
            .add_xaxis(x)
            .add_yaxis("Accuracy", a,yaxis_index=0)  # 图例
            .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
              markpoint_opts=opts.MarkPointOpts(
              data=[opts.MarkPointItem(type_="max", name="最大值")]
                )
            )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
        )
    )
    line.render("Softmax Regression "+mode+".html")

if __name__=='__main__':
    l = [5,4,3,2,1]
    a = [10,11,12,15,16]
    plot_pye(l,a,5,'a')








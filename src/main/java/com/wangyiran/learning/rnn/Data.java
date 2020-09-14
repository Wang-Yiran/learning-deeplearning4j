package com.wangyiran.learning.rnn;

import java.time.LocalDateTime;
import java.util.Arrays;

/**
 * @program: learning-deeplearning4j
 * @description: 用于存储
 * @author: Wang Yiran
 * @create: 2020-09-14 00:02
 **/
public class Data {
    /**
     * 测点数据
     */
    private double[] datas;
    /**
     * 标签
     * 可以用于分类 0-正常 1-异常
     * 也可以用于回归 label即为数值
     */
    private double label;

    public Data() {
    }

    public Data(double[] datas, double label) {
        this.datas = datas;
        this.label = label;
    }

    public double[] getDatas() {
        return datas;
    }

    public void setDatas(double[] datas) {
        this.datas = datas;
    }

    public double getLabel() {
        return label;
    }

    public void setLabel(double label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "Data{" +
                "datas=" + Arrays.toString(datas) +
                ", label=" + label +
                '}';
    }
}

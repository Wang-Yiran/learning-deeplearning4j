package com.wangyiran.learning.base;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @program: learning-deeplearning4j
 * @description:
 * @author: Wang Yiran
 * @create: 2020-09-12 20:06
 **/
public class Nd4jCreate {
    public static void ndArrayCreate() {
        System.out.println("3x5 all 0 ndarray");
        INDArray zeros = Nd4j.zeros(3, 5);
        System.out.println(zeros);

        System.out.println("3x5 all 1 ndarray");
        INDArray ones = Nd4j.ones(3, 5);
        System.out.println(ones);

        System.out.println("3x5 all random ndarray");
        INDArray rands = Nd4j.rand(3, 5);
        System.out.println(rands);

        System.out.println("3x5 all random gaosi ndarray");
        INDArray randsn = Nd4j.randn(3, 5); // 高斯分布
        System.out.println(randsn);

        // 根据一维数组创建shape对应的ndarray
        INDArray array1 = Nd4j.create(new float[]{2,2,2,2}, new int[]{1,4});
        System.out.println(array1);
        INDArray array2 = Nd4j.create(new float[]{2,2,2,2}, new int[]{2,2});
        System.out.println(array2);
    }
}

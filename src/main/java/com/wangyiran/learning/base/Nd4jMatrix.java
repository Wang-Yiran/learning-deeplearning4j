package com.wangyiran.learning.base;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * @program: learning-deeplearning4j
 * @description:
 * @author: Wang Yiran
 * @create: 2020-09-13 00:42
 **/
public class Nd4jMatrix {
    public static void matrix(){
        // 1x2 行向量
        INDArray nd = Nd4j.create(new float[]{1, 2}, new int[]{1, 2});
        // 2x1 列向量
        INDArray nd2 = Nd4j.create(new float[]{3, 4}, new int[]{2, 1});
        // 2x2 matrix
        INDArray nd3 = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        INDArray nd4 = Nd4j.create(new float[]{3, 4, 5, 6}, new int[]{2, 2});
        System.out.println(nd);
        System.out.println(nd2);
        System.out.println(nd3);

        // 1x2 multiply 2x1 = 1x1
        INDArray ndv = nd.mmul((nd2));
        System.out.println(ndv + ", shape = " + Arrays.toString(ndv.shape()));
        // 1x2 and 2x2 = 1x2
        ndv = nd.mmul(nd4);
        System.out.println(ndv + ", shape = " + Arrays.toString(ndv.shape()));
        // 2x2 and 2x2 = 2x2
        ndv = nd3.mmul(nd4);
        System.out.println(ndv + ", shape = " + Arrays.toString(ndv.shape()));
    }

}

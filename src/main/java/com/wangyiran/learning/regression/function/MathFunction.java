package com.wangyiran.learning.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @program: learning-deeplearning4j
 * @description:
 * @author: Wang Yiran
 * @create: 2020-09-13 02:56
 **/

public interface MathFunction {

    INDArray getFunctionValues(INDArray x);

    String getName();
}

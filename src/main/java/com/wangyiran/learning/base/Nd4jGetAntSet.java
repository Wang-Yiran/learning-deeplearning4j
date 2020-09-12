package com.wangyiran.learning.base;

import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @program: learning-deeplearning4j
 * @description:
 * @author: Wang Yiran
 * @create: 2020-09-13 00:30
 **/
public class Nd4jGetAntSet {
    public static void getAndSet(){
        INDArray nd = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9,10,11,12}, new int[]{2,6});
        System.out.println("print original array");
        System.out.println(nd);

        System.out.println("get coordinate 0,3 value");
        double value = nd.getDouble(0, 3);
        System.out.println(value);

        System.out.println("modify coordinate 0,3 value");
        // scalar 标量
        nd.putScalar(0, 3, 100);
        System.out.println(nd);

        // 使用索引迭代器遍历
        System.out.println("use index iterator");
        NdIndexIterator iterator = new NdIndexIterator(2, 6);
        while(iterator.hasNext()){
            int[] nextIndex = iterator.next();
            double nextVal = nd.getDouble(nextIndex);
            System.out.println(nextVal);
        }

        System.out.println("get one row");
        INDArray singleRow = nd.getRow(0);
        System.out.println(singleRow);

        System.out.println("get many row");
        INDArray multiRow = nd.getRows(0, 1);
        System.out.println(multiRow);

        System.out.println("replace one row");
        INDArray replaceRow = Nd4j.create(new float[]{1, 3, 5, 7, 9, 11});
        nd.putRow(0, replaceRow);
        System.out.println(nd);

    }
}

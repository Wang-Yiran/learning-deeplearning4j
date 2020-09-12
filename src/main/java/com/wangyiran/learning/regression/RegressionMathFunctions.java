package com.wangyiran.learning.regression;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import com.wangyiran.learning.regression.function.MathFunction;
import com.wangyiran.learning.regression.function.SinXDivXMathFunction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;
/**
 * @program: learning-deeplearning4j
 * @description:
 * @author: Wang Yiran
 * @create: 2020-09-13 03:00
 **/
public class RegressionMathFunctions {
    //Random number generator seed, for reproducability
    private static final int seed = 12345;
    //Number of epochs (full passes of the data)
    private static final int nEpochs = 2000;
    //How frequently should we plot the network output?
    private static final int plotFrequency = 500;
    //Number of data points
    private static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    private static final int batchSize = 100;
    //Network learning rate
    private static final double learningRate = 0.01;
    private static final Random rng = new Random(seed);
    private static final int numInputs = 1;
    private static final int numOutputs = 1;


    public static void main(final String[] args){

        //Switch these two options to do different functions with different networks
        final MathFunction fn = new SinXDivXMathFunction();
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();

        //Generate the training data
        final INDArray x = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1);
        final DataSetIterator iterator = getTrainingData(x,fn,batchSize,rng);

        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        //Train the network on the full data set, and evaluate in periodically
        final INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
            if((i+1) % plotFrequency == 0) networkPredictions[i/ plotFrequency] = net.output(x, false);
        }

        //Plot the target data and the network predictions
        plot(fn,x,fn.getFunctionValues(x),networkPredictions);
    }

    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 100;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .learningRate(learningRate)
                .list()
                /*
                // SIGMOID 模型的拟合 最后图形相差多可以增加层数或者更换Activation
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.SIGMOID).build())
//                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
//                        .activation(Activation.SIGMOID).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                        */
                /* SIGMOID TANH RELU
                第一，采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。
                第二，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失，参见
                 */
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }

    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    private static DataSetIterator getTrainingData(final INDArray x, final MathFunction function, final int batchSize, final Random rng) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x,y);

        final List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        return new ListDataSetIterator(list,batchSize);
    }

    //Plot the data
    private static void plot(final MathFunction function, final INDArray x, final INDArray y, final INDArray... predicted) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.getName(),      // chart title
                "X",                        // x axis label
                function.getName() + "(X)", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}

package common;

import java.util.Map;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class XORDebug {
    private static final Logger logger = LoggerFactory.getLogger( XORDebug.class);
    
    public static void main(String[] args){
        
        int seed = 1234567;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .seed(seed)
                                .iterations(iterations)
                                //.learningRate(0.01)
                                .miniBatch(false)
                                .useDropConnect(false)
                                .weightInit(WeightInit.XAVIER)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(Updater.SGD)
                                .list()
                                .layer(0, new DenseLayer.Builder()
                                                        .nIn(2)
                                                        .nOut(2)
                                                        .activation(Activation.RELU)//Activation.IDENTITY will not work
                                                                                    //since non-linear transformation
                                                                                    //is needed here
                                                        .learningRate(0.01)
                                                        .build())
                                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                                         .activation(Activation.IDENTITY)
                                                         .learningRate(0.01)
                                                         .nIn(2).nOut(1).build())
                                .backprop(true).pretrain(false)
                                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();        
        //
        double[][] feature = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
        double[][] label = {{0.0},{1.0},{1.0},{0.0}};
        //
        INDArray ndFeature = Nd4j.create(feature);
        INDArray ndLabel = Nd4j.create(label);
        //
        //DataSet ds = new DataSet(ndFeature, ndLabel);
        System.out.println(model.summary());
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage(); 
//        model.setListeners(new StatsListener(statsStorage, 1));
//        uiServer.attach(statsStorage);
        System.out.println("Before Fit Model Param: ");
        Map<String,INDArray> param = model.paramTable();
        for( Map.Entry<String, INDArray> entry : param.entrySet() ){
            System.out.println(entry.getKey() + "," + entry.getValue());
        }
        for(int i = 1; i <= 500; ++i) {
            for( int j = 0; j < 4; ++j ){
                model.fit(ndFeature.getRow(j), ndLabel.getRow(j));
                if( i == 1 ){
                    System.out.println("Iter Training Finish: " + j);
                    System.out.println("Feature: " + ndFeature.getRow(j));
                    Map<String,INDArray> _param = model.paramTable();
                    for( Map.Entry<String, INDArray> entry : _param.entrySet() ){
                        System.out.println(entry.getKey() + "," + entry.getValue());
                    }
                    Pair<Gradient, Double> gra_score = model.gradientAndScore();
                    System.out.println("gradient: " + gra_score.getFirst());
                    System.out.println("score: " + gra_score.getSecond());
                    System.out.println();
                }
            }
            
            
        }
        INDArray output = model.output(ndFeature);
        System.out.println(output);
//        uiServer.stop();
        //
    }
}

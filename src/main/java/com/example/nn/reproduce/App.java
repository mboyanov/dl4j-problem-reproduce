package com.example.nn.reproduce;

import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {

        ComputationGraph cg = Model.getComputationGraph();
        for (int i =0; i< 10; i++) {
            try {
                INDArray[] input = new INDArray[3];
                getC(input);

            //    getFortran(input);
                cg.clear();
                cg.rnnClearPreviousState();
                Map<String, INDArray> activations = cg.feedForward(input, false);
                System.out.println(activations.get("deep0").toString());
            } catch(Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    private static void getFortran(INDArray[] input) {
        input[0] = Nd4j.zeros( new int[] {1,1,40}, 'f');
        input[0].putScalar(0,0,39, 5);
        input[1] = Nd4j.zeros(new int[] {1,19}, 'f');
        input[2] = Nd4j.zeros(1).dup('f');
    }

    private static void getC(INDArray[] input) {
        input[0] = Nd4j.zeros( new int[] {1,1,40});
        input[0].putScalar(0,0,39, 5);
        input[1] = Nd4j.zeros(new int[] {1,19});
        input[2] = Nd4j.zeros(1);
    }
}

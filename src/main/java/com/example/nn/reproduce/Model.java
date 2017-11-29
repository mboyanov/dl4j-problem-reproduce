package com.example.nn.reproduce;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class Model {


    public static ComputationGraph getComputationGraph() {
        ComputationGraphConfiguration config = getConfig();
        config.addPreProcessors(InputType.recurrent(128), InputType.feedForward(19), InputType.feedForward(1));
        ComputationGraph graph = new ComputationGraph(config);
        graph.init();
        return graph;
    }

    private static ComputationGraphConfiguration getConfig() {
        double regularization = 0.0;
        int deepLayers = 1;
        int textEmbeddingSize= 128;
        int textRnnSize= 64;
        int deepLayerSize = 64;
        GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .regularization(true)
                .learningRate(0.001)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(getOptimizer()).graphBuilder();
        EmbeddingLayer embeddingLayer = new EmbeddingLayer.Builder().nIn(40000)
                .weightInit(WeightInit.UNIFORM).activation(Activation.IDENTITY).nOut(textEmbeddingSize)
                .biasInit(0)
                .biasLearningRate(0)
                .name("text_embedding")
                .build();
        EmbeddingLayer channelEmbedding = new EmbeddingLayer.Builder().nIn(400)
                .activation(Activation.IDENTITY)
                .biasInit(0)
                .biasLearningRate(0)
                .name("channel_embedding")
                .nOut(10).build();
        builder.addInputs("text", "meta", "channel")
                .addLayer("text_embedding", embeddingLayer, "text")
                .addLayer("text_rnn", new GravesLSTM.Builder().nIn(textEmbeddingSize).nOut(textRnnSize)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .biasInit(0)
                        .l2(regularization)
                        .l2Bias(regularization)
                        .build(),
                        "text_embedding")
                .addVertex("rnn_out", new LastTimeStepVertex("text"), "text_rnn")
                .addLayer("channel_embedding", channelEmbedding, "channel")
                .addVertex("merge", new MergeVertex(), "rnn_out", "meta", "channel_embedding")
                .addLayer("deep0", new DenseLayer.Builder().nOut(deepLayerSize).activation(Activation.RELU)
                        .dropOut(1)
                        .l2(regularization)
                        .l2Bias(regularization).build(), "merge");
        String lastDeepLayer = "deep0";
        for (int i = 1; i < deepLayers; i++) {
            lastDeepLayer = "deep" + Integer.toString(i);
            builder.addLayer(lastDeepLayer, new DenseLayer.Builder().nOut(deepLayerSize)
                    .l2(regularization)
                    .l2Bias(regularization)
                    .dropOut(0.0).activation(Activation.RELU).build(),
                    "deep" + Integer.toString(i - 1));
        }
        builder.addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nOut(3).l2(regularization).build(), lastDeepLayer).setOutputs("out");
        return builder.build();
    }

    private static Adam getOptimizer() {
        return new Adam();
    }


}

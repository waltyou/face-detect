package com.walt.face.utils;

import org.tensorflow.Graph;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * @author waltyou
 * @date 2018/12/18
 */
public class TensorflowUtils {

    public static Graph getGraph(String pdPath) throws Exception {
        Graph graph = new Graph();
        byte[] graphDef = Files.readAllBytes(Paths.get(pdPath));
//        graphDef = modifyGraphDef(graphDef, "/gpu:0");
        graph.importGraphDef(graphDef);
        return graph;
    }
}

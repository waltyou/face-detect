package com.walt.face.facenet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import static com.walt.face.utils.ResourceUtils.getResourceFilePath;
import static com.walt.face.utils.TensorflowUtils.getGraph;

/**
 * @author waltyou
 * @date 2018/12/18
 */
public class FaceNet {

    private static Logger logger = LoggerFactory.getLogger(FaceNet.class);

    private static final String PD_PATH = "/models/facenet/20180408-102900.pb";

    private Session session;

    public FaceNet() throws Exception {
        Graph graph = getGraph(getResourceFilePath(PD_PATH));
        byte[] config = setConfig();
        session = new Session(graph, config);
    }

    private byte[] setConfig() {
        GPUOptions gpuOptions = GPUOptions.newBuilder()
                .setVisibleDeviceList("0")
                .setPerProcessGpuMemoryFraction(0.25)
                .setAllowGrowth(true)
                .build();
        return ConfigProto.newBuilder()
                .setGpuOptions(gpuOptions)
                .setAllowSoftPlacement(true)
                .build().toByteArray();
    }

    //todo
    public void extractFeature() {
    }
}

package com.walt.face;

import com.walt.face.mtcnn.Box;
import com.walt.face.mtcnn.Mtcnn;
import com.walt.face.opencv.FaceDetection;
import com.walt.face.opencv.ImageData;
import com.walt.face.utils.ImageUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.bytedeco.javacpp.opencv_core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static com.walt.face.utils.ImageUtils.toBufferedImage;

/**
 * @author waltyou
 * @date 2018/12/11
 */
public class Main {

    private static Logger logger = LoggerFactory.getLogger(Main.class);
    private static Mtcnn mtcnn;

    public static void main(String[] args) throws Exception {
        String inputPath = args[0];

        FaceDetection.initClassifier();
        mtcnn = new Mtcnn();

        SparkConf conf = new SparkConf().setAppName("face Count");

        JavaSparkContext sc = new JavaSparkContext(conf);
        byte[] bytes = getBytes(sc, inputPath);
        ImageData output = detectFace(inputPath, bytes);
        logger.info(output.toString());
    }

    private static byte[] getBytes(JavaSparkContext sc, String inputPath) {
        JavaPairRDD<String, PortableDataStream> pairRDD = sc.binaryFiles(inputPath);
        PortableDataStream portableDataStream = pairRDD.take(1).get(0)._2;
        return portableDataStream.toArray();
    }

    private static ImageData detectFace(String inputPath, byte[] bytes) {
        return processByOpencv(inputPath, bytes);
//        return processByMtcnn(inputPath, bytes);
    }

    private static ImageData processByOpencv(String inputPath, byte[] bytes) {
        ImageData imageData = new ImageData(inputPath, bytes);
        opencv_core.Rect[] faces = FaceDetection.getFaces(imageData.getImage());
        List<String> list = new ArrayList<>(faces.length);
        for (opencv_core.Rect rect : faces) {
            list.add(rect.x() + " " + rect.y() + " " + rect.width() + " " + rect.height());
        }
        imageData.setImage(null);
        imageData.setFaceAreas(list);
        imageData.setFaceCount(faces.length);
        return imageData;
    }

    private static ImageData processByMtcnn(String inputPath, byte[] bytes) {
        BufferedImage img = ImageUtils.toBufferedImage(bytes);
        ImageData imageData = new ImageData(inputPath);
        Vector<Box> boxes = null;
        try {
            boxes = mtcnn.detectFaces(img, 20);
            for (Box box : boxes){
                System.out.println(box.left() + " " + box.right() + " " + box.width() + " " + box.height());
            }
            imageData.setFaceCount(boxes.size());
        } catch (Exception e) {
            logger.error(inputPath + " failed: ", e);
        }
        return imageData;
    }


}

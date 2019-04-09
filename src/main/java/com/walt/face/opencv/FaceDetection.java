package com.walt.face.opencv;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_objdetect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;

import static com.walt.face.utils.ResourceUtils.getResourceFilePath;

/**
 * @author waltyou
 * @date 2018/12/10
 */
public class FaceDetection {

    private static Logger logger = LoggerFactory.getLogger(FaceDetection.class);

    private static opencv_objdetect.CascadeClassifier face;
    private static opencv_objdetect.CascadeClassifier eyes;

    public static void initClassifier() throws Exception {
        if (face == null) {
            String faceXML = "/face/haarcascades/haarcascade_frontalface_alt.xml";
            String eyesXML = "/face/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

            String faceXMLPath = getResourceFilePath(faceXML);
            String eyesXMLPath = getResourceFilePath(eyesXML);
            face = new opencv_objdetect.CascadeClassifier(faceXMLPath);
            if (face.empty()) {
                throw new Exception("initClassifier failed! path: " + faceXMLPath
                        + ", exist: " + new File(faceXMLPath).exists());
            }
            eyes = new opencv_objdetect.CascadeClassifier(eyesXMLPath);
            if (eyes.empty()) {
                throw new Exception("initClassifier failed! path: " + eyesXMLPath
                        + ", exist: " + new File(eyesXMLPath).exists());
            }
        }
    }



    public static Mat[] getFacesMat(Mat source) {
        Rect[] rects = getFaces(source);
        Mat[] faces = new Mat[rects.length];
        for (int i = 0; i < rects.length; i++) {
            faces[i] = new Mat(source, rects[i]);
        }
        return faces;
    }

    public static Rect[] getFaces(Mat source) {
        Mat target = new Mat();
        opencv_imgproc.cvtColor(source, target, opencv_imgproc.COLOR_BGR2GRAY);

        RectVector faces = new RectVector();
        face.detectMultiScale(target, faces);
        Rect[] rects = faces.get();
        ArrayList<Rect> list = new ArrayList<Rect>();
        for (Rect rect : rects) {
            Rect[] eyes = getEyes(new Mat(target, rect));
            if (eyes.length != 0) {
                list.add(rect);
            }
        }
        return list.toArray(new Rect[list.size()]);
    }

    private static Rect[] getEyes(Mat source) {
        RectVector faces = new RectVector();
        eyes.detectMultiScale(source, faces);
        return faces.get();
    }


}

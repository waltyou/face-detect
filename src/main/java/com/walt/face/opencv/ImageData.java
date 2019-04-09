package com.walt.face.opencv;

import com.walt.face.utils.ImageUtils;
import org.bytedeco.javacpp.opencv_core.Mat;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static com.walt.face.utils.ImageUtils.toMat;

/**
 * @author waltyou
 * @date 2018/12/17
 */
public class ImageData implements Serializable {

    transient private String path;
    transient private Mat image;
    transient private int faceCount = 0;
    transient private List<String> faceAreas = new ArrayList<>(0);

    public ImageData(Mat image) {
        this.image = image.clone();
    }

    public ImageData(String inputPath, byte[] data) {
        this.image = ImageUtils.toMat(data);
        path = inputPath;
    }

    public ImageData(String inputPath) {
        path = inputPath;
    }

    ImageData(BufferedImage bi) {
        this.image = ImageUtils.toMat(bi);
    }

    public Mat getImage() {
        return image;
    }

    public void setImage(Mat image) {
        this.image = image;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public int getFaceCount() {
        return faceCount;
    }

    public void setFaceCount(int faceCount) {
        this.faceCount = faceCount;
    }

    public List<String> getFaceAreas() {
        return faceAreas;
    }

    public void setFaceAreas(List<String> faceAreas) {
        this.faceAreas = faceAreas;
    }

    @Override
    public String toString() {
        return path + " | " + faceCount + " | " + faceAreas.toString();
    }
}

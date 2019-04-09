package com.walt.face.utils;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * @author waltyou
 * @date 2018/12/17
 */
public class ImageUtils {

    public static opencv_core.Mat toMat(BufferedImage bi) {
        OpenCVFrameConverter.ToIplImage cv = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter jcv = new Java2DFrameConverter();
        return cv.convertToMat(jcv.convert(bi));
    }

    public static opencv_core.Mat toMat(byte[] bs) {
        return toMat(toBufferedImage(bs));
    }

    public static BufferedImage toBufferedImage(byte[] bs) {
        try {
            return ImageIO.read(new ByteArrayInputStream(bs));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return null;
    }

    public static BufferedImage toBufferedImage(opencv_core.Mat mat) {
        OpenCVFrameConverter.ToMat matConverter = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter bimConverter = new Java2DFrameConverter();
        org.bytedeco.javacv.Frame frame = matConverter.convert(mat);
        BufferedImage img = bimConverter.convert(frame);
        img.flush();
        return img;
    }
}

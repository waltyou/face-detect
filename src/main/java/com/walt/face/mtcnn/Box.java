package com.walt.face.mtcnn;

import static java.lang.Math.max;

/**
 * @author waltyou
 * @date 2018/12/06
 */
public class Box {
    // left:box[0],top:box[1],right:box[2],bottom:box[3]
    public int[] box;
    public float score;
    // bounding box regression
    public float[] bbr;
    public boolean deleted;

    Box() {
        box = new int[4];
        bbr = new float[4];
        deleted = false;
    }

    public int left() {
        return box[0];
    }

    public int right() {
        return box[2];
    }

    public int top() {
        return box[1];
    }

    public int bottom() {
        return box[3];
    }

    public int width() {
        return box[2] - box[0] + 1;
    }

    public int height() {
        return box[3] - box[1] + 1;
    }

    //面积
    public int area() {
        return width() * height();
    }

    //Bounding Box Regression
    public void calibrate() {
        int w = box[2] - box[0] + 1;
        int h = box[3] - box[1] + 1;
        box[0] = (int) (box[0] + w * bbr[0]);
        box[1] = (int) (box[1] + h * bbr[1]);
        box[2] = (int) (box[2] + w * bbr[2]);
        box[3] = (int) (box[3] + h * bbr[3]);
        for (int i = 0; i < 4; i++) {
            bbr[i] = 0.0f;
        }
    }

    //当前box转为正方形
    public void toSquareShape() {
        int w = width();
        int h = height();
        if (w > h) {
            box[1] -= (w - h) / 2;
            box[3] += (w - h + 1) / 2;
        } else {
            box[0] -= (h - w) / 2;
            box[2] += (h - w + 1) / 2;
        }
    }

    //防止边界溢出，并维持square大小
    public void limit_square(int w, int h) {
        if (box[0] < 0 || box[1] < 0) {
            int len = max(-box[0], -box[1]);
            box[0] += len;
            box[1] += len;
        }
        if (box[2] >= w || box[3] >= h) {
            int len = max(box[2] - w + 1, box[3] - h + 1);
            box[2] -= len;
            box[3] -= len;
        }
    }

    public void limit_square2(int w, int h) {
        if (width() > w) {
            box[2] -= width() - w;
        }
        if (height() > h) {
            box[3] -= height() - h;
        }
        if (box[0] < 0) {
            int sz = -box[0];
            box[0] += sz;
            box[2] += sz;
        }
        if (box[1] < 0) {
            int sz = -box[1];
            box[1] += sz;
            box[3] += sz;
        }
        if (box[2] >= w) {
            int sz = box[2] - w + 1;
            box[2] -= sz;
            box[0] -= sz;
        }
        if (box[3] >= h) {
            int sz = box[3] - h + 1;
            box[3] -= sz;
            box[1] -= sz;
        }
    }
}

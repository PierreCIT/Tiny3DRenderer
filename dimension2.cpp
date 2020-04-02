//
// Created by noeim on 30/03/2020.
//

#include "dimension2.h"


bool line(int x0, int y0, int x1, int y1, TGAImage &image, const TGAColor &color) {
    bool steep = false;
    if (abs(x1 - x0) < abs(y1 - y0)) { //If the slope is to steep we invert y and x
        swap(x0, y0);
        swap(x1, y1);
        steep = true;
    }
    if (x0 > x1) { //The point 0 must have the smaller abscissa
        swap(x0, x1);
        swap(y0, y1);
    }
    float slope = (y1 - y0) / (float) (x1 - x0);
    float origin_point = y0 - slope * x0;
    for (int x = x0; x < x1; x++) {
        int y = slope * x + origin_point;
        if (steep) {
            image.set(y, x, color);
        } else {
            image.set(x, y, color);
        }
    }
    return true;
}

bool line(Vec2i t0, Vec2i t1, TGAImage &image, const TGAColor &color) {
    return line(t0.x, t0.y, t1.x, t1.y, image, color);
}

bool triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor &color) {
    line(t0, t1, image, color);
    line(t1, t2, image, color);
    line(t2, t0, image, color);
    return true;
}

bool fill_triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor &color) {
    if ((t0.y == t1.y && t0.y == t2.y) || (t0.x == t1.x && t0.x == t2.x)) return false; // we won't care for a line
    if (t0.y > t1.y) swap(t0, t1);
    if (t0.y > t2.y) swap(t0, t2);
    if (t1.y > t2.y) swap(t1, t2);
    float slopeA = t2.x != t0.x ? (float) (t2.y - t0.y) / (float) (t2.x - t0.x) : 0; // t2.y != t0.y so never null
    int origin_point_A = t0.y - t0.x * slopeA;
    for (int i = 0; i <= t2.y - t0.y; i++) {
        bool second_half = i > t1.y - t0.y || t0.y == t1.y;
        Vec2i a = second_half ? t2 : t0;
        Vec2i b = t1;
        if (a.x > b.x) swap(a, b);
        float slopeB = a.x != b.x ? (float) (b.y - a.y) / (float) (b.x - a.x) : 0;
        int borderA_x = slopeA != 0 ? ((t0.y + i) - origin_point_A) / slopeA : t0.x;
        int borderB_x = slopeB != 0 ? ((t0.y + i) - (a.y - a.x * slopeB)) / slopeB : b.x;
        if (borderA_x > borderB_x) swap(borderA_x, borderB_x);
        for (int j = borderA_x; j <= borderB_x; j++) {
            image.set(j, t0.y + i, color);
        }
    }
    return true;
}
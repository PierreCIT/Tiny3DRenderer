#include "main.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);

int main(int argc, char **argv) {
    TGAImage image(100, 100, TGAImage::RGB);
    image.set(52, 41, red);
    line(13, 20, 80, 40, image, white);
    line(20, 13, 40, 80, image, red);
    line(80, 40, 13, 20, image, red);
    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");
    return 0;
}

bool line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
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

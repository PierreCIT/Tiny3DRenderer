#include "main.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
Model *model = NULL;
const int width  = 800;
const int height = 800;

int main(int argc, char **argv) {
    TGAImage image_line(100, 100, TGAImage::RGB);
    image_line.set(52, 41, red);
    line(13, 20, 80, 40, image_line, white);
    line(20, 13, 40, 80, image_line, red);
    line(80, 40, 13, 20, image_line, red);
    image_line.flip_vertically(); // i want to have the origin at the left bottom corner of the image_wireframe
    image_line.write_tga_file("output.tga");

    TGAImage image_wireframe(width, height, TGAImage::RGB);
    model = new Model("./obj/african_head.obj");
    for (int i=0; i<model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        for (int j=0; j<3; j++) {
            Vec3f v0 = model->vert(face[j]);
            Vec3f v1 = model->vert(face[(j+1)%3]);
            int x0 = (v0.x+1.)*width/2.;
            int y0 = (v0.y+1.)*height/2.;
            int x1 = (v1.x+1.)*width/2.;
            int y1 = (v1.y+1.)*height/2.;
            line(x0, y0, x1, y1, image_wireframe, white);
        }
    }
    image_wireframe.flip_vertically();
    image_wireframe.write_tga_file("african_head_wireframe.tga");

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

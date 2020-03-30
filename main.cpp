#include "main.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);
Model *model = NULL;
const int width = 800;
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
    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        for (int j = 0; j < 3; j++) {
            Vec3f v0 = model->vert(face[j]);
            Vec3f v1 = model->vert(face[(j + 1) % 3]);
            int x0 = (v0.x + 1.) * width / 2.;
            int y0 = (v0.y + 1.) * height / 2.;
            int x1 = (v1.x + 1.) * width / 2.;
            int y1 = (v1.y + 1.) * height / 2.;
            line(x0, y0, x1, y1, image_wireframe, white);
        }
    }
    image_wireframe.flip_vertically();
    image_wireframe.write_tga_file("african_head_wireframe.tga");

    TGAImage image_wireframe_randomColordFilled(width, height, TGAImage::RGB);
    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec2i screen_coords[3];
        for (int j = 0; j < 3; j++) {
            Vec3f world_coords = model->vert(face[j]);
            screen_coords[j] = Vec2i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.);
        }
        fill_triangle(screen_coords[0], screen_coords[1], screen_coords[2],
                      image_wireframe_randomColordFilled,
                      TGAColor(rand() % 254 + 1, rand() % 254 + 1, rand() % 254 + 1, 255));
    }
    image_wireframe_randomColordFilled.flip_vertically();
    image_wireframe_randomColordFilled.write_tga_file("WireFrame_triangles_filled.tga");

    TGAImage image_wireframe_light_intensity(width, height, TGAImage::RGB);
    Vec3f light_dir(0, 0, -1);
    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec2i screen_coords[3];
        Vec3f worldCoordonate[3];
        for (int j = 0; j < 3; j++) {
            Vec3f world_coords = model->vert(face[j]);
            screen_coords[j] = Vec2i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.);
            worldCoordonate[j] = world_coords;
        }
        Vec3f face_normal = (worldCoordonate[2] - worldCoordonate[0])^(worldCoordonate[1] - worldCoordonate[0]);
        face_normal.normalize();
        int light_intensity = face_normal * light_dir *255;
        if (light_intensity > 0) {
            fill_triangle(screen_coords[0], screen_coords[1], screen_coords[2],
                          image_wireframe_light_intensity,
                          TGAColor(light_intensity, light_intensity, light_intensity, 255));
        }
    }
    image_wireframe_light_intensity.flip_vertically();
    image_wireframe_light_intensity.write_tga_file("WireFrame_light_intensity.tga");

    delete model;

// ---------------
//    Triangle tests
//    TGAImage image_triangles(500, 500, TGAImage::RGB);
//    //Vec2i t0[3] = {Vec2i(10, 70), Vec2i(50, 160), Vec2i(70, 80)};
//    Vec2i t0[3] = {Vec2i(414, 348), Vec2i(429, 353), Vec2i(429, 345)};
//    Vec2i t1[3] = {Vec2i(180, 50), Vec2i(150, 1), Vec2i(70, 180)};
//    Vec2i t2[3] = {Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180)};
//    triangle(t0[0], t0[1], t0[2], image_triangles, red);
//    triangle(t1[0], t1[1], t1[2], image_triangles, white);
//    triangle(t2[0], t2[1], t2[2], image_triangles, green);
//    image_triangles.flip_vertically();
//    image_triangles.write_tga_file("Triangles.tga");
//
//    TGAImage image_filled_triangles(500, 500, TGAImage::RGB);
//    fill_triangle(t0[0], t0[1], t0[2], image_filled_triangles, red);
//    fill_triangle(t1[0], t1[1], t1[2], image_filled_triangles, white);
//    fill_triangle(t2[0], t2[1], t2[2], image_filled_triangles, green);
//    image_filled_triangles.flip_vertically();
//    image_filled_triangles.write_tga_file("Triangles_filled.tga");

    return 0;
}

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




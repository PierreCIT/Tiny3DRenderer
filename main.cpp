#include "main.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);
Model *model = NULL;
const int width = 800;
const int height = 800;
Vec3f light_dir(0, 0, -1);





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
    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec2i screen_coords[3];
        Vec3f worldCoordonate[3];
        for (int j = 0; j < 3; j++) {
            Vec3f world_coords = model->vert(face[j]);
            screen_coords[j] = Vec2i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.);
            worldCoordonate[j] = world_coords;
        }
        Vec3f face_normal = (worldCoordonate[2] - worldCoordonate[0]) ^(worldCoordonate[1] - worldCoordonate[0]);
        face_normal.normalize();
        int light_intensity = face_normal * light_dir * 255;
        if (light_intensity > 0) {
            fill_triangle(screen_coords[0], screen_coords[1], screen_coords[2],
                          image_wireframe_light_intensity,
                          TGAColor(light_intensity, light_intensity, light_intensity, 255));
        }
    }
    image_wireframe_light_intensity.flip_vertically();
    image_wireframe_light_intensity.write_tga_file("WireFrame_light_intensity.tga");

    float zbuffer[width][height];
    for (int i =0; i<width;i++) {
        for( int j = 0; j<height; j++){
            zbuffer[i][j] = -0.4;//-std::numeric_limits<float>::max();

        }
    }

    TGAImage image_wireframe_Zbuffer_light_intensity(width, height, TGAImage::RGB);

    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec3f screen_coords[3];
        Vec3f worldCoordonate[3];
        for (int j = 0; j < 3; j++) {
            Vec3f world_coords = model->vert(face[j]);
            screen_coords[j] = Vec3f((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.,
                                     world_coords.z);
            worldCoordonate[j] = world_coords;
        }
        Vec3f face_normal = (worldCoordonate[2] - worldCoordonate[0]) ^(worldCoordonate[1] - worldCoordonate[0]);
        face_normal.normalize();
        int light_intensity = face_normal * light_dir * 255;
        if (light_intensity > 0) {
            fill_triangle_3d(screen_coords[0], screen_coords[1], screen_coords[2], zbuffer,
                             image_wireframe_Zbuffer_light_intensity,
                             TGAColor(light_intensity, light_intensity, light_intensity, 255));
        }
    }
    image_wireframe_Zbuffer_light_intensity.flip_vertically();
    image_wireframe_Zbuffer_light_intensity.write_tga_file("WireFrame_light_intensity_Zbuffer.tga");

//    for (int i = width * height; i >= 0; i--) zbuffer[i] = -0.4;//-std::numeric_limits<float>::max();
//    for (int x = (width / 2) - 100; x < (width / 2) + 100; x++) {
//        for (int y = (height / 2) - 200; y < (height / 2) + 200; y++) {
//            zbuffer[x + y * width] = 0;
//        }
//    }
    draw_z_buffer(zbuffer, "Test_tedt.tga", width, height);
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






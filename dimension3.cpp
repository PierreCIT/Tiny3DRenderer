//
// Created by noeim on 30/03/2020.
//

#include "dimension3.h"

bool fill_triangle_3d(Vec3f t0, Vec3f t1, Vec3f t2, float zbuffer[][800], TGAImage &image, const TGAColor &color) {
    if ((t0.y == t1.y && t0.y == t2.y) || (t0.x == t1.x && t0.x == t2.x)) return false; // we won't care for a line
    if (t0.y > t1.y) swap(t0, t1);
    if (t0.y > t2.y) swap(t0, t2);
    if (t1.y > t2.y) swap(t1, t2);
    float slopeA = t2.x != t0.x ? (float) (t2.y - t0.y) / (float) (t2.x - t0.x) : 0; // t2.y != t0.y so never null
    int origin_point_A = (int)(t0.y - t0.x * slopeA);
    for (int i = 0; i <= (int) (t2.y - t0.y); i++) {
        bool second_half = i > t1.y - t0.y || t0.y == t1.y;
        Vec3f a = second_half ? t2 : t0;
        Vec3f b = t1;
        if (a.x > b.x) swap(a, b);
        float slopeB = a.x != b.x ? (float) (b.y - a.y) / (float) (b.x - a.x) : 0;
        int borderA_x = slopeA != 0 ? (int)((t0.y + i) - origin_point_A) / slopeA : t0.x;
        int borderB_x = slopeB != 0 ? (int)((t0.y + i) - (a.y - a.x * slopeB)) / slopeB : b.x;
        if (borderA_x > borderB_x) swap(borderA_x, borderB_x);
        for (int j = borderA_x; j <= borderB_x; j++) {
            Vec3f P;
            P.z = 0;
            P.x = j;
            P.y = t0.y + i;
            Vec3f bc_screen = barycentric(t0, t1, t2, P);
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue;
            P.z = t0.z * bc_screen.x + t1.z * bc_screen.y + t2.z * bc_screen.z;
//            P.z = z_interpolation(t0, t1, t2, P);
            float test_value = zbuffer[int(P.x)][int(P.y)];
            if (zbuffer[int(P.x)][int(P.y)] < P.z) {
                zbuffer[int(P.x)][int(P.y)] = P.z;
                image.set(P.x, P.y, color);
            }
        }

    }
    return true;
}

Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2];
    for (int i = 2; i--;) {
        s[i].raw[0] = C.raw[i] - A.raw[i];
        s[i].raw[1] = B.raw[i] - A.raw[i];
        s[i].raw[2] = A.raw[i] - P.raw[i];
    }
    Vec3f u = s[0] ^s[1];
    if (std::abs(u.raw[2]) > 1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
        return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
    return Vec3f(-1, 1, 1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

float z_interpolation(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f bc_screen = barycentric(A, B, C, P);
    float z_test = A.z * bc_screen.x + B.z * bc_screen.y + C.z * bc_screen.z;
    float w1_area = abs((P.x - A.x) * (B.y - A.y) - ((P.y - A.y) * (B.x - A.x)));
    float w2_area = abs((P.x - B.x) * (C.y - B.y) - ((P.y - B.y) * (C.x - B.x)));
    float w3_area = abs((P.x - C.x) * (A.y - C.y) - ((P.y - C.y) * (A.x - C.x)));

    float triangle_area = w1_area + w2_area + w3_area;
    float a_coef = (w1_area / triangle_area);
    float b_coef = (w2_area / triangle_area);
    float c_coef = (w3_area / triangle_area);
    float z = A.z * a_coef + B.z * b_coef + C.z * c_coef;
    return z_test;
}

Vec3f create_vector_from_to(Vec3f A, Vec3f B) {
    Vec3f S(B.x - A.x, B.y - A.y, B.z - A.z);
    return S;
}


void draw_z_buffer(float z_buffer[][800], string filename, int width, int height) {
    TGAImage image(width, height, TGAImage::RGB);
    float min = std::numeric_limits<float>::max();
    float max = -std::numeric_limits<float>::max();
    for (int x = 0; x < width; x++) {
        for(int y=0; y<height; y++){
            if (z_buffer[x][y] > max) {
                max = z_buffer[x][y];
            } else if (z_buffer[x][y] < min) {
                min = z_buffer[x][y];
            }
        }

    }
    float divide = max - min;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int value = (int) (((z_buffer[x][y]- min) / divide) * 230);
            image.set(x, y, TGAColor(value, value, value, 255));
            if (value > 0) {
                int buffer_value = z_buffer[x][y];
                int a = 3;
            }
        }
    }
    image.flip_vertically();
    image.write_tga_file(filename + ".tga");
}
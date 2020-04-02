//
// Created by noeim on 30/03/2020.
//

#ifndef TINY3DRENDERER_DIMENSION3_H
#define TINY3DRENDERER_DIMENSION3_H

#include "geometry.h"
#include "model.h"
#include "tgaimage.h"

using namespace std;

bool depth_interpolation(Vec2i t0, Vec2i t1, Vec2i t2, Vec2i P);

bool fill_triangle_3d(Vec3f t0, Vec3f t1, Vec3f t2, float zbuffer[][800], TGAImage &image, const TGAColor &color);

float z_interpolation(Vec3f A, Vec3f B, Vec3f C, Vec3f P);

void draw_z_buffer(float z_buffer[][800], string filename, int width, int height);

Vec3f create_vector_from_to(Vec3f A, Vec3f B);
Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P);

#endif //TINY3DRENDERER_DIMENSION3_H

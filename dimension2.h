//
// Created by noeim on 30/03/2020.
//

#ifndef TINY3DRENDERER_DIMENSION2_H
#define TINY3DRENDERER_DIMENSION2_H

#include "geometry.h"
#include "model.h"
#include "tgaimage.h"

using namespace std;

bool line(int x0, int y0, int x1, int y1, TGAImage &image, const TGAColor& color);
bool line(Vec2i t0, Vec2i t1, TGAImage &image, const TGAColor& color);
bool triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor& color);
bool fill_triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor& color);

#endif //TINY3DRENDERER_DIMENSION2_H

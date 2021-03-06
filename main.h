//
// Created by noeim on 23/03/2020.
//

#ifndef TINY3DRENDERER_MAIN_H
#define TINY3DRENDERER_MAIN_H

#include <iostream>
#include <string>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"

using namespace std;

bool line(int x0, int y0, int x1, int y1, TGAImage &image, const TGAColor& color);
bool line(Vec2i t0, Vec2i t1, TGAImage &image, const TGAColor& color);
bool triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor& color);
bool fill_triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, const TGAColor& color);
bool equalVec2i(Vec2i v1, Vec2i v2);


#endif //TINY3DRENDERER_MAIN_H

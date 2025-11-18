//
// Created by Daniel Pommer on 12.11.25.
//
#include <datatypes.h>
#include <conversions.h>

#ifndef INTERACTINGMAPS_COST_H
#define INTERACTINGMAPS_COST_H
float costFR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R);

float costFG(Tensor3f &F, const Tensor2f V, const Tensor3f &G);

float costGI(Tensor3f &G, const Tensor3f &I_gradient);

#endif //INTERACTINGMAPS_COST_H
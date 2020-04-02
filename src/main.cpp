//
// Created by sw3390 on 3/11/20.
//

#include <Titan/sim.h>

int main() {
    Simulation sim;

    sim.createMass(Vec(0,0,10));

    sim.createPlane(Vec(0, 0, 1), 0);

    sim.setGlobalAcceleration(Vec(0, 0, -9.8));

    sim.start(); // 10 second runtime.

}

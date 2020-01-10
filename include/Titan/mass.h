//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_MASS_H
#define TITAN_MASS_H

#include "vec.h"
#include "object.h"

#include <algorithm>
using namespace  std;

class Mass;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
struct CUDA_MASS {
    CUDA_MASS() = default;
    CUDA_MASS(Mass & mass);

    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    double damping;
    double extduration; // duration of external force
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2
    Vec extforce; // external force in kg m / s^2
    Vec maxforce; // max force by magnitude since simulation start in kg m / s^2

#ifdef GRAPHICS
    Vec color;
#endif

    bool valid;

#ifdef CONSTRAINTS
    CUDA_LOCAL_CONSTRAINTS constraints;
#endif

};

class Mass {
public:
    //Properties
    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    double damping; // damping mass velocity
    double extduration; // duration of external force
    double density; // density of material
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2
    Vec extforce; // external force in kg m / s^2
    Vec maxforce; // max force by magnitude since simulation start in kg m / s^2
    Vec origpos; // original position in m
    int index; // index in masses array
    int ref_count; // reference count
    int spring_count; // number of attached springs

    bool valid; // false if mass has been deleted from arrays


    Mass(const Mass &other);
    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);
#ifdef CONSTRAINTS
    LOCAL_CONSTRAINTS constraints;

    void addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num);
    void clearConstraints(CONSTRAINT_TYPE type);
    void clearConstraints();

    void setDrag(double C);
    void fix();
    void unfix();
#endif
    
#ifdef GRAPHICS
    Vec color;
#endif

private:

    void decrementRefCount();

    CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

    Mass();
    void operator=(CUDA_MASS & mass);

    friend class Simulation;
    friend class Spring;
    friend struct CUDA_SPRING;
    friend struct CUDA_MASS;
    friend class Container;
    friend class Lattice;
    friend class Beam;
    friend class Cube;


};

#endif //TITAN_MASS_H

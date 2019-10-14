//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_SPRING_H
#define TITAN_SPRING_H

#include "mass.h"
#include "vec.h"

#include <algorithm>

class Mass;
struct CUDA_SPRING;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
const int ACTIVE_CONTRACT_THEN_EXPAND = 0;
const int ACTIVE_EXPAND_THEN_CONTRACT = 1;
const int PASSIVE_SOFT = 2;
const int PASSIVE_STIFF = 3;
const int ACTIVE_EXPAND_THEN_NEUTRAL = 4;
const int ACTIVE_CONTRACT_THEN_NEUTRAL = 5;


class Spring {
public:

    //Properties
    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)
    double _diam; // spring diameter (meters)
    double _break_force; // spring breakage point (N)
    double _curr_force; // stateful force (N)
    double _max_stress; // maximum stress exerienced
    bool _broken; // true when spring is broken
    double _mass; // Contributing mass (only used to update Mass objects)

    // BREATHING
    int _type; // 0-5
    double _period; // time period
    double _offset; // time offset
    double _omega; // frequency
    double _actuation; // actuation amount (normalized)
    
    Mass * _left = nullptr; // pointer to left mass object
    Mass * _right = nullptr; // pointer to right mass object


    //Set
    Spring() { _left = nullptr; _right = nullptr; arrayptr = nullptr; _k = 10000.0; _rest = 1.0; _diam = 0.001;_break_force = 10; _broken = false; _type=PASSIVE_STIFF; _period=1.0; _offset=0.0; _omega=0.0; _actuation=0.0; _max_stress = 0.0; }; //Constructor

    Spring(const Spring &other);

    Spring(const CUDA_SPRING & spr);

    Spring(Mass * left, Mass * right, double k = 10000.0, double rest_len = 1.0, double diam = 0.001):
            _k(k), _rest(rest_len), _diam(diam), _left(left), _right(right), arrayptr(nullptr), _break_force(10), _curr_force(0), _max_stress(0), _broken(false), _actuation(0.0)
    {}

    Spring(double k, double rest_length, Mass * left, Mass * right) :
            _k(k), _rest(rest_length), _diam(0.001), _left(left), _right(right),_break_force(10), _curr_force(0), _max_stress(0), _broken(false), _actuation(0.0)
    {}

    Spring(double k, double rest_length, Mass * left, Mass * right, int type, double omega) :
            _k(k), _rest(rest_length), _diam(0.001), _left(left), _right(right), _type(type), _omega(omega), _break_force(10), _curr_force(0), _max_stress(0), _broken(false), _actuation(0.0)
    {}
	    
    void setForce(); // will be private
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength(); //sets rest length

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right) {
        if (_left != nullptr) {
            _left->ref_count--;
        }
        if (_right != nullptr) {
            _right->ref_count--;
        }

        _left = left;
        _right = right;

    } //sets both right and left masses

    //Get
    Vec getForce(); // computes force on right object. left force is - right force.
    int getLeft();
    int getRight();

private:
    //    Mass * _left; // pointer to left mass object // private
    //    Mass * _right; // pointer to right mass object
    CUDA_SPRING *arrayptr; //Pointer to struct version for GPU cudaMalloc

    void operator=(CUDA_SPRING & spring);

    friend class Simulation;
    friend struct CUDA_SPRING;
    friend class Container;
    friend class Lattice;
    friend class Cube;
    friend class Beam;
};

struct CUDA_SPRING {
  CUDA_SPRING() : _max_stress(0) {}
  CUDA_SPRING(const Spring & s);
  
  CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right);
  
  CUDA_MASS * _left; // pointer to left mass object
  CUDA_MASS * _right; // pointer to right mass object
  
  double _k; // spring constant (N/m)
  double _rest; // spring rest length (meters)
  double _diam; // spring diameter (meters)
  double _break_force; // spring breakage point (N)
  double _curr_force; // stateful force (N)
  double _max_stress; // maximum stress exerienced
  double _broken; // true when spring is broken
  double _mass; // Contributing mass (only used to update Mass objects)

    // Breathing
  int _type;
  double _period;
  double _offset;
  double _omega;
  double _actuation;
};

#endif //TITAN_SPRING_H

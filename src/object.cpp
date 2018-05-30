//
// Created by Jacob Austin on 5/21/18.
//

#include "object.h"

//class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
//public:
//    void setNormal(const Vec & normal) { _normal = normal; }; // normal is (a, b, c)
//    void setOffset(double d) { _offset = d; }; // ax + by + cz < d
//
//private:
//    Vec _normal;
//    double _offset;
//};

Vec Plane::getForce(const Vec & position) { // returns force on an object based on its position, e.g. plane or
    double disp = dot(position, _normal) - _offset;
    return (disp < 0) ? - DISPL_CONST * disp * _normal : 0 * _normal;
}

Plane::Plane(const Vec & normal, double d) {
    _offset = d;
    _normal = normal;
}

void Plane::translate(const Vec & displ) {
    _offset += dot(displ, _normal);
}

void ContainerObject::setMassValue(double m) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> setMass(m);
    }
}

void ContainerObject::setKValue(double k) {
    for (Spring * spring : springs) {
        spring -> setK(k);
    }
}

void ContainerObject::setDeltaTValue(double dt) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> setDeltaT(dt);
    }
}

void ContainerObject::setRestLengthValue(double len) { // set masses for all Mass objects
    for (Spring * spring : springs) {
        spring -> setRestLength(len);
    }
}

void ContainerObject::makeFixed() {
    for (Mass * mass : masses) {
        mass -> makeFixed();
    }
}


Cube::Cube(const Vec & center, double side_length) {
    _center = center;
    _side_length = side_length;

    for (int i = 0; i < 8; i++) {
        masses.push_back(new Mass(1.0, side_length * (Vec(i & 1, (i >> 1) & 1, (i >> 2) & 1) - Vec(0.5, 0.5, 0.5)) + center));
    }

    int count = 0; // debug
    for (int i = 0; i < 8; i++) { // add the appropriate springs
        for (int j = i + 1; j < 8; j++) {
            springs.push_back(new Spring(masses[i], masses[j]));
//            std::cout << count << ": " << i << ", " << j << std::endl; // debug
            count++; // debug
        }
    }
}

void Cube::translate(const Vec & displ) {
    for (Mass * m : masses) {
        m->translate(displ);
    }
}


void Cube::generateBuffers() {
    static const GLfloat g_color_buffer_data[] = { // colors for the cube
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
            1.0f, 0.2f, 0.2f,
    };

    GLuint colorbuffer; // bind cube colors to buffer colorbuffer2
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

    this -> colors = colorbuffer;

    GLuint elementbuffer; // create buffer for main cube object
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

    std::vector<GLubyte> indices; // this contains the order in which to draw the lines between points
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            indices.push_back(i);
            indices.push_back(j);
        }
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size(), &indices[0], GL_STATIC_DRAW);

    this -> indices = elementbuffer;

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer); // bind cube vertex buffer

    this -> vertices = vertexbuffer;
}

void Cube::updateBuffers() {
    GLfloat vertex_data[24];

    for (int i = 0; i < 8; i++) { // populate buffer with position data for cube vertices
        vertex_data[3 * i] = (GLfloat) masses[i] -> getPosition()[0];
        vertex_data[3 * i + 1] = (GLfloat) masses[i]->getPosition()[1];
        vertex_data[3 * i + 2] = (GLfloat) masses[i]->getPosition()[2];
    }
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);
}


void Cube::draw() {
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, this -> vertices);
    glPointSize(10);
    glLineWidth(10);
    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, this -> colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    glDrawArrays(GL_POINTS, 0, 8); // 3 indices starting at 0 -> 1 triangle
    glDrawElements(GL_LINES, 2 * springs.size(), GL_UNSIGNED_BYTE, (void*) 0); // 3 indices starting at 0 -> 1 triangle

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

void Plane::generateBuffers() {

    float length = 5;
    float width = 5;
    float depth = 1;
    glm::vec3 color = {0.2f, 0.3f, 0.5f};

    GLfloat vertex_buffer_platform[108] = {
            -length, -width,-depth,
            -length, -width,0.0f,
            -length, width,0.0f,
            length, width,-depth,
            -length, -width,-depth,
            -length, width,-depth,
            length, -width,0.0f,
            -length, -width,-depth,
            length, -width,-depth,
            length, width,-depth,
            length, -width,-depth,
            -length, -width,-depth,
            -length, -width,-depth,
            -length, width, 0.0f,
            -length, width,-depth,
            length, -width, 0.0f,
            -length, -width, 0.0f,
            -length, -width,-depth,
            -length, width, 0.0f,
            -length, -width, 0.0f,
            length, -width, 0.0f,
            length, width, 0.0f,
            length, -width,-depth,
            length, width,-depth,
            length, -width,-depth,
            length, width, 0.0f,
            length, -width, 0.0f,
            length, width, 0.0f,
            length, width,-depth,
            -length, width,-depth,
            length, width, 0.0f,
            -length, width,-depth,
            -length, width, 0.0f,
            length, width, 0.0f,
            -length, width, 0.0f,
            length, -width, 0.0f
    };

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_platform), vertex_buffer_platform, GL_STATIC_DRAW);

    static const GLfloat g_color_buffer_data[] = {
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
    };

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);
}

void Plane::draw() {
    // 1st attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}
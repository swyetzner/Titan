//
//  shader.hpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include <string>


#ifndef shader_hpp
#define shader_hpp

static const std::string FragmentShaderCode = "#version 330 core\n"
                                              "\n"
                                              "// Interpolated values from the vertex shaders\n"
                                              "in vec3 fragmentColor;\n"
                                              "\n"
                                              "// Ouput data\n"
                                              "out vec3 color;\n"
                                              "\n"
                                              "void main(){\n"
                                              "\n"
                                              "\t// Output color = color specified in the vertex shader, \n"
                                              "\t// interpolated between all 3 surrounding vertices\n"
                                              "\tcolor = fragmentColor;\n"
                                              "\n"
                                              "}";

static const  std::string VertexShaderCode = "#version 330 core\n"
                               "\n"
                               "// Input vertex data, different for all executions of this shader.\n"
                               "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                               "layout(location = 1) in vec3 vertexColor;\n"
                               "\n"
                               "// Output data ; will be interpolated for each fragment.\n"
                               "out vec3 fragmentColor;\n"
                               "// Values that stay constant for the whole mesh.\n"
                               "uniform mat4 MVP;\n"
                               "\n"
                               "void main(){\t\n"
                               "\n"
                               "\t// Output position of the vertex, in clip space : MVP * position\n"
                               "\tgl_Position =  MVP * vec4(vertexPosition_modelspace,1);\n"
                               "\n"
                               "\t// The color of each vertex will be interpolated\n"
                               "\t// to produce the color of each fragment\n"
                               "\tfragmentColor = vertexColor;\n"
                               "}\n";

static const std::string  PlaneVertexShaderCode =
        "#version 330\n"
        "layout (location = 0) in vec3 vertexPosition_modelspace;\n"
        "out vec3 vertexPos;\n"
        "uniform mat4 MVP;\n"
        "void main() {\n"
        "   gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);\n"
        "   vertexPos = vertexPosition_modelspace;\n"
        "}\n";

static const std::string  PlaneFragmentShaderCode =
        "#version 330\n"
        "out vec4 fragColor;\n"
        "in vec3 vertexPos;\n"
        "void main() {"
        "   vec2 coord = vertexPos.xy;\n"
        "   vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);\n"
        "   float line = min(grid.x, grid.y);\n"
        "   fragColor = vec4(vec3(1.0 - min(line, 0.8)), 1.0);\n"
        "}\n";


void CompileShader(GLuint shader, const char *source);
GLuint LoadShaders(const char *vertexShaderSource, const char *fragementShaderSource);

#endif /* shader_hpp */

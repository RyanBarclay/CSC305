// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

// Image height
const int H = 480;

// Camera settings
const double near_plane = 1.5; // AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; // 45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

// Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

// Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

// Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
// Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

// Fills the different arrays
void setup_scene()
{
    // Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    // Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{
    // TODO: setup uniform

    // TODO: setup camera, compute w, u, v

    // w = -g / || g ||
    const Vector3d w = -camera_gaze.normalized();

    // u = (t x w) / ||(t x w)||
    const Vector3d u = camera_top.cross(w).normalized();

    // v = w x u
    const Vector3d v = w.cross(u);

    // TODO: compute the camera transformation

    // Construct the unique transformations that converts world coordinates into camera coordinates
    /*
    M_cam =
    [u v w e
    0 0 0 1]^-1
    */
    Matrix4f Temp_M_cam;

    Temp_M_cam << u(0), v(0), w(0), camera_position(0),
        u(1), v(1), w(1), camera_position(1),
        u(2), v(2), w(2), camera_position(2),
        0, 0, 0, 1;

    Matrix4f M_cam;
    M_cam << Temp_M_cam.inverse();

    // TODO: setup projection matrix

    // From assignment 3 to calc t and l
    const float t = near_plane * tan(field_of_view / 2.0);
    const float r = t * aspect_ratio;

    const float l = -r;
    const float b = -t;
    const float n = -near_plane;
    const float f = -far_plane;

    Matrix4f M_orth;
    M_orth << 2 / (r - l), 0, 0, -(r + l) / (r - l),
        0, 2 / (t - b), 0, -(t + b) / (t - b),
        0, 0, 2 / (n - f), -(n + f) / (n - f),
        0, 0, 0, 1;

    // Now we can transform into the unifrom view

    Matrix4d P;
    if (is_perspective)
    {
        // TODO setup prespective camera
        P << n, 0, 0, 0,
            0, n, 0, 0,
            0, 0, (n + f), (-f * n),
            0, 0, 1, 0;

        uniform.view = M_orth * P.cast<float>() * M_cam;
    }
    else
    {
        // Orthographic
        uniform.view = M_orth * M_cam;
    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        // return va;

        VertexAttributes out;
        out.position = uniform.view * va.position;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        // TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    // TODO: build the vertex attributes from vertices and facets

    // rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);

    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d av, bv, cv;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        av << vertices.row(verticesIndex(0)).transpose();
        bv << vertices.row(verticesIndex(1)).transpose();
        cv << vertices.row(verticesIndex(2)).transpose();

        const VertexAttributes a(av[0], av[1], av[2]);
        const VertexAttributes b(bv[0], bv[1], bv[2]);
        const VertexAttributes c(cv[0], cv[1], cv[2]);

        vertex_attributes.push_back(a);
        vertex_attributes.push_back(b);
        vertex_attributes.push_back(c);
    }

    float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    uniform.view << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    if (aspect_ratio < 1)
        uniform.view(0, 0) = aspect_ratio;
    else
        uniform.view(1, 1) = 1 / aspect_ratio;

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

Matrix4d compute_rotation(const double alpha)
{
    // TODO: Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4d res;
    double c = cos(alpha);
    double s = sin(alpha);

    res << c, 0, s, 0,
        0, 1, 0, 0,
        -s, 0, c, 0,
        0, 0, 0, 1;

    Matrix4d I;
    I.setIdentity();
    res = res * I;
    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4d trafo = compute_rotation(alpha);

    // float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    // uniform.view << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    uniform.view = uniform.view * trafo.cast<float>();

    // uniform.view(1, 1) = 1 / aspect_ratio;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        // return va;
        VertexAttributes out;
        out.position = uniform.view * va.position;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        // TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    // TODO: generate the vertex attributes for the edges and rasterize the lines
    // TODO: use the transformation matrix

    // For each triangle
    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d av, bv, cv;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        av << vertices.row(verticesIndex(0)).transpose();
        bv << vertices.row(verticesIndex(1)).transpose();
        cv << vertices.row(verticesIndex(2)).transpose();

        const VertexAttributes a(av[0], av[1], av[2]);
        const VertexAttributes b(bv[0], bv[1], bv[2]);
        const VertexAttributes c(cv[0], cv[1], cv[2]);

        // line from a -> b
        vertex_attributes.push_back(a);
        vertex_attributes.push_back(b);

        // line from b -> c
        vertex_attributes.push_back(b);
        vertex_attributes.push_back(c);

        // line from c -> a
        vertex_attributes.push_back(c);
        vertex_attributes.push_back(a);
    }

    // float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    // uniform.view << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    uniform.view = uniform.view * trafo.cast<float>();

    // uniform.view(1, 1) = 1 / aspect_ratio;

    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

void get_shading_program(Program &program)
{
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // The only difference lies in the attributes that are sent to the vertex sharer.

        // TODO: transform the position and the normal

        // Transform the position
        VertexAttributes out;
        out.position = uniform.view * va.position;
        out.normal = va.normal;

        Vector3d lights_color(0, 0, 0);
        for (int i = 0; i < light_positions.size(); i++)
        {
            const Vector3d light_position = light_positions[i];
            const Vector3d light_color = light_colors[i];

            // need to make p and N so we can port over old code
            Vector3d p(out.position[0], out.position[1], out.position[2]);
            Vector3d N(va.normal[0], va.normal[1], va.normal[2]);

            const Vector3d Li = (light_position - p).normalized();

            // Diffuse contribution
            const Vector3d diffuse = obj_diffuse_color * std::max(Li.dot(N), 0.0);

            // Specular contribution
            const Vector3d Hi = (Li - p).normalized();
            const Vector3d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);

            // Attenuate lights according to the squared distance to the lights
            const Vector3d D = light_position - p;
            lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
        }
        lights_color += ambient_light;

        Vector4f C(lights_color[0], lights_color[1], lights_color[2], 1);
        out.color = C;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        // TODO: create the correct fragment
        FragmentAttributes out(va.color(0), va.color(1), va.color(2), uniform.color(3));
        Vector4f tmp;
        if (is_perspective)
        {
            tmp << va.position[0], va.position[1], va.position[2], va.position[3];
        }
        else
        {
            tmp << va.position[0], va.position[1], -va.position[2], va.position[3];
        }
        out.position = tmp; // TRY doing out.position = va.position
        return out;
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        // TODO: implement the depth check
        if (fa.position(2) < previous.depth)
        {
            FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
            out.depth = fa.position[2];
            return out;
        }
        else
        {
            return previous;
        }

        // return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;    // same from simple and wireframe
    build_uniform(uniform);       // same from simple and wireframe
    Program program;              // same from simple and wireframe
    get_shading_program(program); // This is new, will update program

    Eigen::Matrix4d trafo = compute_rotation(alpha); // This is new

    std::vector<VertexAttributes> vertex_attributes;
    // TODO: compute the normals

    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d av, bv, cv;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        av << vertices.row(verticesIndex(0)).transpose();
        bv << vertices.row(verticesIndex(1)).transpose();
        cv << vertices.row(verticesIndex(2)).transpose();

        VertexAttributes a(av[0], av[1], av[2]);
        VertexAttributes b(bv[0], bv[1], bv[2]);
        VertexAttributes c(cv[0], cv[1], cv[2]);

        const Vector3d pgram_u = bv - av;
        const Vector3d pgram_v = cv - av;

        Vector3f N = pgram_v.cross(pgram_u).normalized().cast<float>();

        a.normal = N;
        b.normal = N;
        c.normal = N;

        vertex_attributes.push_back(a);
        vertex_attributes.push_back(b);
        vertex_attributes.push_back(c);
    }

    float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    // uniform.view << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    uniform.view *= trafo.cast<float>();

    // if (aspect_ratio < 1)
    //     uniform.view(0, 0) = aspect_ratio;
    // else
    //     uniform.view(1, 1) = 1 / aspect_ratio;

    // TODO: set material colors

    uniform.color = {0, 0, 0, 1};

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4d trafo = compute_rotation(alpha);

    float aspect_ratio = float(frameBuffer.cols()) / float(frameBuffer.rows());

    // uniform.view << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    uniform.view *= trafo.cast<float>();

    // if (aspect_ratio < 1)
    //     uniform.view(0, 0) = aspect_ratio;
    // else
    //     uniform.view(1, 1) = 1 / aspect_ratio;

    // TODO: compute the vertex normals as vertex normal average

    std::vector<VertexAttributes> vertex_attributes;

    std::vector<Vector3f> Normals(vertices.rows(), Vector3f(0, 0, 0));
    // TODO: create vertex attributes

    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d av, bv, cv;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        av << vertices.row(verticesIndex(0)).transpose();
        bv << vertices.row(verticesIndex(1)).transpose();
        cv << vertices.row(verticesIndex(2)).transpose();

        // VertexAttributes a(av[0], av[1], av[2]);
        // VertexAttributes b(bv[0], bv[1], bv[2]);
        // VertexAttributes c(cv[0], cv[1], cv[2]);

        const Vector3d pgram_u = bv - av;
        const Vector3d pgram_v = cv - av;

        Vector3f N = pgram_v.cross(pgram_u).normalized().cast<float>();

        Normals[verticesIndex(0)] += N;
        Normals[verticesIndex(1)] += N;
        Normals[verticesIndex(2)] += N;
        // a.normal = N;
        // b.normal = N;
        // c.normal = N;

        // vertex_attributes.push_back(a);
        // vertex_attributes.push_back(b);
        // vertex_attributes.push_back(c);
    }

    for (int i = 0; i < facets.rows(); ++i)
    {
        Vector3d av, bv, cv;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        av << vertices.row(verticesIndex(0)).transpose();
        bv << vertices.row(verticesIndex(1)).transpose();
        cv << vertices.row(verticesIndex(2)).transpose();

        VertexAttributes a(av[0], av[1], av[2]);
        VertexAttributes b(bv[0], bv[1], bv[2]);
        VertexAttributes c(cv[0], cv[1], cv[2]);

        a.normal = Normals[verticesIndex(0)].normalized();
        b.normal = Normals[verticesIndex(1)].normalized();
        c.normal = Normals[verticesIndex(2)].normalized();

        vertex_attributes.push_back(a);
        vertex_attributes.push_back(b);
        vertex_attributes.push_back(c);
    }

    // TODO: set material colors

    uniform.color = {0, 0, 0, 1};

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    // Simple_render
    // DONE

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    // Wireframe_render
    // Done
    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    // Flat_shading
    // Done
    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    // Pv_shading
    // IP
    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    // TODO: add the animation

    // Wireframe_render
    // Done

    int delay = 25;
    GifWriter g;
    GifBegin(&g, "wireframe_render.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 1; i < 24; i += EIGEN_PI / 12)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        wireframe_render(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    GifBegin(&g, "flat_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 1; i < 24; i += EIGEN_PI / 12)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        flat_shading(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    GifBegin(&g, "pv_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);
    for (float i = 1; i < 24; i += EIGEN_PI / 12)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        pv_shading(i, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);

    return 0;
}

// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

// Custom Functions

Vector3d getPoint(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
{
    // *** Make matrix A ***

    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    // std::cout << "Here is the matrix A:\n"
    //           << A << std::endl;

    // *** Make vector from a-e ***

    Vector3d ae_vector = a_vector - e_vector;
    // std::cout << "Here is the vector a:\n"
    //           << a_vector << std::endl;
    // std::cout << "Here is the vector e:\n"
    //           << e_vector << std::endl;
    // std::cout << "Here is the vector ae:\n"
    //           << ae_vector << std::endl;

    // *** Calc the values for solution ****
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);

    // For some reason the left and right version of the equation are giving different points

    // Vector3d rightside = a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);

    // Vector3d leftside = e_vector + (solution_vector(2) * v_vector);

    // if (leftside != rightside)
    // {
    //     printf("NOT GOOD\n");
    //     std::cout << "Here is the left:\n"
    //               << leftside << std::endl;
    //     std::cout << "Here is the right:\n"
    //               << rightside << std::endl;
    // }

    return a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);
}

bool raytri(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
{
    // *** Make matrix A ***

    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    // std::cout << "Here is the matrix A:\n"
    //           << A << std::endl;

    // *** Make vector from a-e ***

    Vector3d ae_vector = a_vector - e_vector;
    // std::cout << "Here is the vector a:\n"
    //           << a_vector << std::endl;
    // std::cout << "Here is the vector e:\n"
    //           << e_vector << std::endl;
    // std::cout << "Here is the vector ae:\n"
    //           << ae_vector << std::endl;

    // *** Calc the values for solution ****
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);
    // std::cout << "Here is the vector solution:\n"
    //           << solution_vector << std::endl;

    // *** Validate answer ***

    // Check t
    if (solution_vector(2) < 0)
    {
        return false;
    }

    // Check u
    if (solution_vector(0) < 0 || solution_vector(0) > 1)
    {
        return false;
    }

    // Check v
    if (solution_vector(1) <= 0 || solution_vector(1) > 1)
    {
        return false;
    }

    // *** If all checks good, return true. ***
    return true;
}
void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            Vector2d ray_on_xy(ray_origin(0), ray_origin(1));
            const double sphere_radius = 0.9;

            if (ray_on_xy.norm() < sphere_radius)
            {
                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection(
                    ray_on_xy(0), ray_on_xy(1),
                    sqrt(sphere_radius * sphere_radius - ray_on_xy.squaredNorm()));

                // Compute normal at the intersection point
                Vector3d ray_normal = ray_intersection.normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0); // a
    const Vector3d pgram_u(0, 0.7, -10);        // b - a
    const Vector3d pgram_v(1, 0.4, 0);          // c - a

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;             // e
            const Vector3d ray_direction = camera_view_direction; // d

            // TODO: Check if the ray intersects with the parallelogram

            // f(u,v) = a + (b-a)u + (c-a)v  **Will need to confirm if need to make unit vector length or not**
            // ** this is a (x,y,z) point that is inside of the parallelogram **
            // b = pu + a
            // c = pv + a
            // a = origin

            // p(t) = e + (t*d) ** this is a (x,y,z) point **
            // e = origin of ray
            // d = ray direction

            // Intersects if p(t) = f(u,v); {t > 0, u >= 0 , v >= 0, u+v <= 1}

            //

            if (raytri(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin))
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection
                // point

                Vector3d ray_intersection = getPoint(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin);

                // TODO: Compute normal at the intersection point
                // Vector3d ray_normal = ray_intersection.normalized();
                Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // TODO: Check if the ray intersects with the parallelogram
            if (true)
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection
                // point
                Vector3d ray_intersection(0, 0, 0);

                // TODO: Compute normal at the intersection point
                Vector3d ray_normal = ray_intersection.normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    // Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    // material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            Vector2d ray_on_xy(ray_origin(0), ray_origin(1));
            const double sphere_radius = 0.9;

            if (ray_on_xy.norm() < sphere_radius)
            {
                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection(
                    ray_on_xy(0), ray_on_xy(1),
                    sqrt(sphere_radius * sphere_radius - ray_on_xy.squaredNorm()));

                // Compute normal at the intersection point
                Vector3d ray_normal = ray_intersection.normalized();

                // TODO: Add shading parameter here
                const double diffuse = (light_position - ray_intersection).normalized().dot(ray_normal);
                const double specular = (light_position - ray_intersection).normalized().dot(ray_normal);

                // Simple diffuse model
                C(i, j) = ambient + diffuse + specular;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

int main()
{
    // raytrace_sphere();
    raytrace_parallelogram();
    // raytrace_perspective();
    // raytrace_shading();

    return 0;
}
